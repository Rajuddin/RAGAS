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
from ragas.cost import TokenUsage
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

# Below this, a metric counts as "low" for diagnose_row(). Matches the existing
# GOOD/MODERATE score-band boundary used for display (_score_label, duplicated in
# tests/test_ragas_evaluation.py and app_pages/rag_evaluation.py).
LOW_SCORE_THRESHOLD = 0.70

# Shown alongside every non-empty diagnose_row() verdict. This is a statistical
# read of 4 metric scores, not a debugger attached to the RAG system -- it can
# point at the right *area*, but confirming the actual root cause (a specific
# chunking bug, a truncated prompt, a missing system-prompt instruction, ...)
# needs a human looking at the real retrieved chunks / assembled prompt / raw
# model output for that row. Phrased for an end user, not just a developer.
DIAGNOSIS_DISCLAIMER = (
    "This is a possible area for improvement based on an automated reading of the RAGAS "
    "metric scores for this row, not a confirmed root cause. Treat it as a starting point: "
    "a developer should debug the RAG system more deeply -- inspect the actual retrieved "
    "chunks, how they were assembled into the prompt, and the raw model output for this "
    "specific case -- before making changes."
)

# Concrete starting points for each focus area diagnose_row() can raise. Intentionally
# a fixed checklist, not tailored per row -- diagnose_row() can name *which* stage looks
# off, not *why*, so these are things to go check, not a prescribed fix.
POSSIBLE_FIXES = {
    "Retrieval": [
        "Review chunking strategy (chunk size/overlap) -- chunks too large dilute relevance; "
        "chunks too small can split a needed fact across multiple chunks",
        "Re-evaluate the embedding model for this domain -- a generic embedding model may not "
        "capture domain-specific terminology well",
        "Tune top_k (number of retrieved chunks) -- too few hurts recall, too many adds noise "
        "that hurts precision",
        "Add a re-ranking step (e.g. a cross-encoder) after initial retrieval to push the most "
        "relevant chunks to the top",
        "Review the search/index configuration -- hybrid search (keyword + vector), metadata "
        "filters, or relevance boosting may help",
        "Confirm the knowledge base actually contains the needed information -- if it's simply "
        "missing, this is a content-coverage gap, not a retrieval-tuning problem",
    ],
    "Augmentation": [
        "Check whether every relevant retrieved chunk is actually making it into the final "
        "prompt, or being dropped/truncated by a context-window or token-limit",
        "Review chunk ordering in the assembled prompt -- placement can affect how much weight "
        "the model gives each chunk",
        "Check for lossy formatting when context is inserted into the prompt (e.g. stripping "
        "table/structured data the model needs to interpret correctly)",
        "Check for duplicate or overlapping chunks that could dilute or conflict with the "
        "genuinely relevant one",
    ],
    "Generation": [
        "Strengthen system-prompt instructions to require grounding answers strictly in the "
        "provided context, with an explicit fallback for when context doesn't contain the answer",
        "Lower the generation temperature to reduce embellishment beyond what the context supports",
        "Evaluate a different/more capable model for instruction-following and grounding",
        "Add explicit instructions (or few-shot examples) demonstrating focused answers to the "
        "specific question asked, to reduce tangential/off-topic content",
    ],
}


def diagnose_row(scores: dict) -> dict:
    """Map a row's 4 RAGAS metric scores onto which stage of the RAG pipeline --
    Retrieval, Augmentation, and/or Generation -- looks like it needs work.

    The 4 metrics split into two groups that judge genuinely different things:
      Retrieval    = context_precision (are relevant chunks ranked near the top?)
                     + context_recall (does the retrieved set cover everything the
                     answer needs, at all?). Both judge the retrieved context
                     itself, independent of what the LLM does with it.
      Generation   = faithfulness (is the answer grounded in whatever context it
                     got?) + response_relevancy (does the answer address the
                     actual question asked?). Both judge the LLM's output, given
                     whatever context it received.

    Augmentation -- how the retrieved context got assembled into the prompt the
    generation LLM actually saw (chunk ordering, truncation, formatting) -- isn't
    directly measured by any single RAGAS metric here, since this project evaluates
    an external RAG system's already-generated answer rather than controlling that
    assembly step itself. Its fingerprint is *good* retrieval scores paired with
    *poor* faithfulness: the right material was available, yet the answer isn't
    grounded in it. That pattern is equally consistent with a botched hand-off into
    the prompt (augmentation) or the model disregarding perfectly good context
    (generation) -- the two can't be told apart from scores alone, so both are
    surfaced together for that specific pattern rather than guessing which one.

    When retrieval is already bad, a low faithfulness score is much less
    diagnostic (hard to stay grounded in context that wasn't good to begin with),
    so it's noted as a likely symptom rather than raising Augmentation/Generation
    again -- fix retrieval first, then re-evaluate. response_relevancy, by
    contrast, is raised independently of retrieval quality: an off-topic answer is
    off-topic regardless of what was retrieved.

    Returns {"focus_areas": [...], "reasons": [...], "summary": str,
             "possible_fixes": {...}, "disclaimer": str}:
      focus_areas    -- ordered subset of ["Retrieval", "Augmentation", "Generation"],
                        empty if every computed metric is >= LOW_SCORE_THRESHOLD.
      reasons        -- one bullet string per metric that drove a call-out, naming
                        the metric and its value, so this is never a black-box verdict.
      summary        -- a short narrative paragraph explicitly cross-referencing the
                        metric values that led to each call-out (e.g. "context_recall
                        is low (0.35) and faithfulness is also low (0.40), which is
                        consistent with...").
      possible_fixes -- {focus_area: [checklist items]} for each area in focus_areas,
                        from POSSIBLE_FIXES -- concrete things to go inspect, not a
                        prescribed fix (see DIAGNOSIS_DISCLAIMER).
      disclaimer     -- DIAGNOSIS_DISCLAIMER, repeated on every non-trivial result so
                        it can't be dropped by a caller that only reads one field.
    """

    def value(key):
        v = scores.get(key)
        return v if v is not None and not math.isnan(v) else None

    precision = value("context_precision")
    recall = value("context_recall")
    faithfulness = value("faithfulness")
    relevancy = value("response_relevancy")

    if precision is None and recall is None and faithfulness is None and relevancy is None:
        return {
            "focus_areas": [],
            "reasons": ["No metrics were computed for this row — cannot diagnose."],
            "summary": "No metrics were computed for this row, so no diagnosis is possible.",
            "possible_fixes": {},
            "disclaimer": DIAGNOSIS_DISCLAIMER,
        }

    retrieval_bad = (precision is not None and precision < LOW_SCORE_THRESHOLD) or (
        recall is not None and recall < LOW_SCORE_THRESHOLD
    )

    focus_areas = []
    reasons = []
    summary_parts = []

    if retrieval_bad:
        focus_areas.append("Retrieval")
        bad_bits = []
        if precision is not None and precision < LOW_SCORE_THRESHOLD:
            reasons.append(
                f"context_precision {precision:.2f} — relevant chunks aren't ranked near the "
                "top (retrieval ranking/noise)"
            )
            bad_bits.append(f"context_precision is low ({precision:.2f})")
        if recall is not None and recall < LOW_SCORE_THRESHOLD:
            reasons.append(
                f"context_recall {recall:.2f} — retrieved context doesn't cover everything the "
                "answer needs (retrieval coverage, chunking, or a knowledge-base gap)"
            )
            bad_bits.append(f"context_recall is low ({recall:.2f})")
        summary_parts.append(
            f"{' and '.join(bad_bits)}, indicating the retrieval step isn't surfacing the right "
            "information — fix this first, since the generation-side scores below are hard to "
            "trust until the retrieved context itself is right."
        )

    if faithfulness is not None and faithfulness < LOW_SCORE_THRESHOLD:
        retrieval_confirmed_good = not retrieval_bad and precision is not None and recall is not None
        if retrieval_bad:
            reasons.append(
                f"faithfulness {faithfulness:.2f} — likely a downstream symptom of the retrieval "
                "issue above rather than a separate root cause; re-check after fixing retrieval"
            )
            summary_parts.append(
                f"faithfulness is also low ({faithfulness:.2f}), which is consistent with — and "
                "likely explained by — the retrieval problem above, rather than being an "
                "independent issue."
            )
        elif retrieval_confirmed_good:
            focus_areas.extend(["Augmentation", "Generation"])
            reasons.append(
                f"faithfulness {faithfulness:.2f} despite good retrieval — the answer isn't "
                "grounded in context that was actually available (augmentation hand-off into "
                "the prompt, or generation grounding)"
            )
            summary_parts.append(
                f"context_precision ({precision:.2f}) and context_recall ({recall:.2f}) are both "
                f"healthy, yet faithfulness is low ({faithfulness:.2f}) — the right material was "
                "retrieved, but the final answer isn't grounded in it. That points at either how "
                "the context was assembled into the prompt (augmentation) or how the model used "
                "it (generation)."
            )
        else:
            # context_precision and/or context_recall weren't computed for this row, so
            # retrieval quality is unverified, not confirmed good -- don't claim otherwise.
            focus_areas.extend(["Augmentation", "Generation"])
            reasons.append(
                f"faithfulness {faithfulness:.2f} — the answer isn't grounded in its context, but "
                "context_precision/context_recall weren't computed for this row, so retrieval "
                "quality can't be ruled out as a contributing cause either"
            )
            summary_parts.append(
                f"faithfulness is low ({faithfulness:.2f}), but context_precision/context_recall "
                "weren't computed for this row, so retrieval quality can't be ruled in or out as "
                "a contributing cause — treat augmentation/generation as the best current guess."
            )

    if relevancy is not None and relevancy < LOW_SCORE_THRESHOLD:
        if "Generation" not in focus_areas:
            focus_areas.append("Generation")
        reasons.append(
            f"response_relevancy {relevancy:.2f} — the answer doesn't directly address the "
            "question asked (generation focus/prompt)"
        )
        if faithfulness is not None and faithfulness >= LOW_SCORE_THRESHOLD:
            summary_parts.append(
                f"The answer is grounded in its context (faithfulness {faithfulness:.2f}) but "
                f"response_relevancy is low ({relevancy:.2f}) — it accurately reflects the "
                "context without actually answering the question that was asked."
            )
        else:
            summary_parts.append(
                f"response_relevancy is low ({relevancy:.2f}) — the answer doesn't directly "
                "address the question asked."
            )

    if not focus_areas:
        return {
            "focus_areas": [],
            "reasons": [f"All computed metrics are >= {LOW_SCORE_THRESHOLD:.2f} — no action needed"],
            "summary": f"All computed metrics are at or above {LOW_SCORE_THRESHOLD:.2f} for this "
                       "row — no specific area needs attention.",
            "possible_fixes": {},
            "disclaimer": DIAGNOSIS_DISCLAIMER,
        }

    possible_fixes = {area: POSSIBLE_FIXES[area] for area in dict.fromkeys(focus_areas)}

    return {
        "focus_areas": focus_areas,
        "reasons": reasons,
        "summary": " ".join(summary_parts),
        "possible_fixes": possible_fixes,
        "disclaimer": DIAGNOSIS_DISCLAIMER,
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


def _parse_token_usage(llm_result) -> TokenUsage:
    """Token usage parser for ragas.evaluate()'s token_usage_parser -- covers every
    provider path config_loader.build_langchain_llm supports. OpenAI-style clients
    (ChatOpenAI/AzureChatOpenAI, used for the openai/azure/foundry-v1 provider paths)
    report `llm_output['token_usage']['prompt_tokens'/'completion_tokens']`; the
    classic Azure AI Foundry serverless path (AzureAIChatCompletionsModel) reports
    `llm_output['token_usage']['input_tokens'/'output_tokens']` instead. Since which
    shape applies isn't known at this call site, this just tries both -- whichever
    isn't populated contributes 0, which is what an absent field would give anyway.
    Verified end-to-end against this project's Azure deployment (see conversation
    history) before wiring this in.
    """
    llm_output = llm_result.llm_output or {}
    usage = llm_output.get("token_usage") or {}
    input_tokens = usage.get("prompt_tokens") or usage.get("input_tokens") or 0
    output_tokens = usage.get("completion_tokens") or usage.get("output_tokens") or 0
    return TokenUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        model=llm_output.get("model_name", ""),
    )


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

    Returns (scores, duration_s, errors, token_usage):
      scores    -- {metric_key: float} for every key in metric_keys (NaN if it
                    could not be computed after all attempts)
      duration_s -- total wall-clock time across all attempts
      errors    -- {metric_key: str} the real reason (exception type + message,
                    e.g. "RateLimitError: Error code: 429 - ...") for every metric
                    key still NaN in `scores` — see _JobErrorCapture. Empty dict
                    when every requested metric computed successfully.
      token_usage -- {"input_tokens": int, "output_tokens": int, "total_tokens": int}
                    summed across every judge-LLM call made for this row, across
                    every metric and every retry attempt (via ragas's built-in
                    token_usage_parser/CostCallbackHandler -- see _parse_token_usage).
                    Covers chat/completion calls only, not embeddings (used by
                    response_relevancy) -- ragas's cost tracking doesn't instrument
                    those. All zero if no LLM call ever completed (e.g. every
                    attempt errored before returning a response).
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
    input_tokens = 0
    output_tokens = 0
    remaining_keys = list(metric_keys)
    for attempt in range(1, max_attempts + 1):
        if on_attempt is not None:
            on_attempt(attempt, max_attempts, remaining_keys)
        metrics = [METRIC_BUILDERS[k](llm, embeddings) for k in remaining_keys]
        capture = _JobErrorCapture()
        _RAGAS_EXECUTOR_LOGGER.addHandler(capture)
        try:
            result = evaluate(
                dataset=dataset,
                metrics=metrics,
                run_config=run_config,
                raise_exceptions=False,
                token_usage_parser=_parse_token_usage,
            )
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

        try:
            usage = result.total_tokens()
            for u in usage if isinstance(usage, list) else [usage]:
                input_tokens += u.input_tokens
                output_tokens += u.output_tokens
        except ValueError:
            pass  # no LLM call completed this attempt (e.g. every metric errored)

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
    token_usage = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }
    return scores, duration_s, errors, token_usage
