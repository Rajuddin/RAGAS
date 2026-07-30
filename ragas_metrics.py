"""
ragas_metrics.py
Shared metric registry, RunConfig, and single-row evaluation logic used by both
the Streamlit UI (app_pages/rag_evaluation.py) and the pytest suite
(tests/test_ragas_evaluation.py), so the two can't silently drift out of sync
the way they did before (the UI got RunConfig tuning + config-driven metric
selection + retries; the pytest suite never did).
"""

import asyncio
import logging
import math
import sys
import time

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import ResponseRelevancy, Faithfulness, ContextPrecision, ContextRecall
from ragas.run_config import RunConfig

# ragas.evaluate() calls asyncio.run() fresh on every attempt (see EVAL_RUN_CONFIG
# below), which creates *and closes* a new event loop each time. On Windows, the
# default WindowsProactorEventLoopPolicy's loop.close() can hang forever inside
# _poll() during its own teardown if an async HTTP client's connection (openai's/
# langchain_openai's httpx-based client, reused across these many short-lived
# loops) leaves a pending overlapped I/O operation IOCP never signals as complete.
# This isn't a slow LLM or a bad prompt -- it happens *after* the real work already
# finished, confirmed via live py-spy stack dumps showing worker threads stuck in
# asyncio\windows_events.py's _poll, reached through loop.close(), not through any
# awaited call. WindowsSelectorEventLoopPolicy uses select()-based I/O instead of
# IOCP and doesn't have this hang-on-close behavior; the only capability it lacks
# (subprocess pipes) isn't used anywhere in this project (all LLM/embedding calls
# are plain HTTP), so switching is safe here. Must happen before any event loop is
# created in this process, which is why it's done at import time in this shared
# module rather than deeper in evaluate_single_row.
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# An LLM call is retried at most this many times before a row is treated as failed.
MAX_RETRIES = 1

# context_precision issues one sequential LLM call per retrieved context chunk
# (ragas.metrics._context_precision.LLMContextPrecisionWithReference._ascore loops
# `for context in retrieved_contexts`), so a row's context_precision cost scales
# with chunk count. This is just a sanity ceiling against pathologically large
# context lists (rows with more chunks than this fail fast with a clear error --
# see TooManyContextsError -- instead of burning a full attempt on something that
# was never going to fit any reasonable budget). It is intentionally *not* sized to
# guarantee EVAL_RUN_CONFIG.timeout covers every chunk: the timeout below is kept
# tight so a slow/stuck call fails visibly within minutes rather than reliably
# succeeding no matter how long that takes -- see the comment there.
MAX_CONTEXTS = 15


class TooManyContextsError(ValueError):
    """Raised by evaluate_single_row when a row has more retrieved contexts than
    MAX_CONTEXTS — see the comment above MAX_CONTEXTS for why that limit exists."""


# ragas.evaluate() defaults to RunConfig(timeout=180, max_retries=10, max_wait=60) for
# every metric's internal LLM calls whenever run_config isn't passed explicitly — even
# if a tighter RunConfig was set on the LLM/embeddings wrapper, evaluate() silently
# resets it (see ragas.evaluation.aevaluate: `run_config = run_config or RunConfig()`,
# then `metric.init(run_config)`). Passing this explicitly to every evaluate() call is
# what actually makes a bad/slow call fail fast instead of retrying for minutes.
# timeout=120: deliberately tight, per an explicit requirement that one test case
# should finish within a few minutes *or fail with a clear error* rather than wait
# indefinitely for a possibly-still-working call. response_relevancy/faithfulness/
# context_recall normally finish well inside this (single-digit seconds to ~30s);
# context_precision on a row with many context chunks (see MAX_CONTEXTS) or an
# unlucky run of slow individual chunk calls (~2-30s each observed) can legitimately
# exceed 120s -- when it does, that one metric comes back as a clear TimeoutError
# (surfaced via _JobErrorCapture, not a bare NaN) while the other three metrics for
# that row are unaffected, since only still-missing metrics get retried (see
# evaluate_single_row). Combined with MAX_RETRIES=1 (2 attempts), a row's scoring
# is bounded at ~240s worst case, matching that requirement.
# max_workers=4 (not the default 16): each row only ever has 4 metric-level tasks in
# flight at once (one per configured metric), so 16 workers just means up to 4x more
# concurrent LLM calls hitting the deployment than are actually needed per row, adding
# unnecessary peak load with no speed benefit — this caps it to what's actually used.
EVAL_RUN_CONFIG = RunConfig(timeout=120, max_retries=MAX_RETRIES, max_wait=15, max_workers=4)

# Every metric config.yaml's ragas.metrics list can name, and how to build/label each
# one. context_precision/context_recall are the expensive ones (one sequential LLM
# call per context chunk each).
METRIC_BUILDERS = {
    "response_relevancy": lambda llm, embeddings: ResponseRelevancy(llm=llm, embeddings=embeddings),
    "faithfulness": lambda llm, embeddings: Faithfulness(llm=llm),
    "context_precision": lambda llm, embeddings: ContextPrecision(llm=llm),
    "context_recall": lambda llm, embeddings: ContextRecall(llm=llm),
}
METRIC_LABELS = {
    "response_relevancy": "Response Relevancy",
    "faithfulness": "Faithfulness",
    "context_precision": "Context Precision",
    "context_recall": "Context Recall",
}


class _JobErrorCapture(logging.Handler):
    """Recovers the real reason a metric came back NaN.

    ragas.evaluate() is always called with raise_exceptions=False (see
    evaluate_single_row below) so one bad metric can't abort the others. But
    ragas.executor.Executor implements that by catching *every* per-job exception
    (TimeoutError, a 429 rate-limit error, a malformed-JSON parse failure that
    exhausted ragas's own internal retry, an auth error, ...) and only logging it
    via `logger.error("Exception raised in Job[%s]: %s(%s)", counter, type, msg)`
    before replacing the result with NaN — the exception itself is never re-raised
    or stored anywhere the caller can retrieve it (ragas/executor.py:71-84). Without
    this handler, every one of those distinct failure modes is indistinguishable
    from every other and looks identical to the caller: a bare NaN.

    `counter` is the job's submission index, which — for the single-row Dataset
    evaluate_single_row builds, with a fresh Executor per evaluate() call — lines
    up exactly with the position of each metric in the `metrics` list passed to
    that call (ragas/evaluation.py:253-264: one job submitted per metric, in
    order, for row 0). That lets the caller map a captured error back to the
    specific metric key that produced it.
    """

    def __init__(self):
        super().__init__()
        self.by_job_index = {}

    def emit(self, record):
        if record.name == "ragas.executor" and record.args and len(record.args) == 3:
            counter, exec_name, exec_message = record.args
            self.by_job_index[counter] = f"{exec_name}: {exec_message}" if exec_message else exec_name


_RAGAS_EXECUTOR_LOGGER = logging.getLogger("ragas.executor")


def configured_metric_keys(app_config) -> list:
    """Metric keys to run, from config.yaml's ragas.metrics list. Falls back to all
    four if the list is missing/empty/unrecognized."""
    keys = [k for k in (getattr(app_config, "ragas_metrics", None) or []) if k in METRIC_BUILDERS]
    return keys or list(METRIC_BUILDERS)


def evaluate_single_row(
    llm,
    embeddings,
    metric_keys,
    question: str,
    answer: str,
    contexts: list,
    ground_truth: str,
    run_config: RunConfig = None,
    max_attempts: int = None,
    on_attempt=None,
) -> tuple:
    """Run ragas.evaluate() for one {question, answer, contexts, ground_truth} row,
    retrying up to max_attempts times if any requested metric comes back NaN.

    Retries are scoped to only the metric(s) still missing — a metric that already
    succeeded on an earlier attempt is never recomputed. This matters because
    context_precision/context_recall are the slow ones (one LLM call per context
    chunk each): without this, one flaky context_precision call would force
    response_relevancy and faithfulness to be redone too, even though they already
    had good scores.

    on_attempt, if given, is called as on_attempt(attempt, max_attempts, retry_keys)
    before each attempt (attempt starts at 1; retry_keys is the list of metric keys
    being (re)computed this attempt) — lets callers surface "retrying X..." progress.

    Raises TooManyContextsError if len(contexts) > MAX_CONTEXTS, before making any
    LLM calls — see the comment above MAX_CONTEXTS for why that limit exists.

    Returns (scores, duration_s, errors):
      scores    -- {metric_key: float} for every key in metric_keys (NaN if it
                    could not be computed after all attempts)
      duration_s -- total wall-clock time across all attempts
      errors    -- {metric_key: str} the real reason (exception type + message,
                    e.g. "RateLimitError: Error code: 429 - ...") for every metric
                    key still NaN in `scores` — see _JobErrorCapture. Empty dict
                    when every requested metric computed successfully.
    """
    if len(contexts) > MAX_CONTEXTS:
        raise TooManyContextsError(
            f"{len(contexts)} retrieved contexts exceeds the supported maximum of "
            f"{MAX_CONTEXTS}. context_precision issues one sequential LLM call per "
            f"context chunk, so rows with more chunks than this cannot reliably "
            f"finish within EVAL_RUN_CONFIG's timeout. Reduce the number of contexts "
            f"returned by the RAG API, or raise ragas_metrics.MAX_CONTEXTS (and its "
            f"paired EVAL_RUN_CONFIG.timeout) if you can afford longer waits."
        )

    run_config = run_config or EVAL_RUN_CONFIG
    max_attempts = max_attempts or (MAX_RETRIES + 1)

    dataset = Dataset.from_dict(
        {
            "question": [question],
            "answer": [answer],
            "contexts": [contexts],
            "ground_truth": [ground_truth],
        }
    )

    start_t = time.perf_counter()
    scores = {k: float("nan") for k in metric_keys}
    errors = {}
    remaining_keys = list(metric_keys)
    for attempt in range(1, max_attempts + 1):
        if on_attempt is not None:
            on_attempt(attempt, max_attempts, remaining_keys)
        metrics = [METRIC_BUILDERS[k](llm, embeddings) for k in remaining_keys]
        capture = _JobErrorCapture()
        _RAGAS_EXECUTOR_LOGGER.addHandler(capture)
        try:
            result = evaluate(dataset=dataset, metrics=metrics, run_config=run_config, raise_exceptions=False)
            row = result.to_pandas().iloc[0].to_dict()
        except Exception as exc:
            # evaluate() itself blew up (not a single metric inside it, e.g. executor/
            # dataset setup) -- attribute it to every metric still in play this round
            # so the reason isn't lost, then let the outer loop retry.
            for k in remaining_keys:
                errors[k] = f"{type(exc).__name__}: {exc}"
            continue
        finally:
            _RAGAS_EXECUTOR_LOGGER.removeHandler(capture)

        still_missing = []
        for idx, k in enumerate(remaining_keys):
            fallback = row.get("answer_relevancy") if k == "response_relevancy" else None
            value = row.get(k, fallback)
            value = float(value) if value is not None else float("nan")
            if math.isnan(value):
                still_missing.append(k)
                if idx in capture.by_job_index:
                    errors[k] = capture.by_job_index[idx]
            else:
                scores[k] = value
                errors.pop(k, None)
        remaining_keys = still_missing
        if not remaining_keys:
            break

    duration_s = time.perf_counter() - start_t
    errors = {k: v for k, v in errors.items() if k in remaining_keys}
    return scores, duration_s, errors
