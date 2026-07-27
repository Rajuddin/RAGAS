"""
ragas_metrics.py
Shared metric registry, RunConfig, and single-row evaluation logic used by both
the Streamlit UI (app_pages/rag_evaluation.py) and the pytest suite
(tests/test_ragas_evaluation.py), so the two can't silently drift out of sync
the way they did before (the UI got RunConfig tuning + config-driven metric
selection + retries; the pytest suite never did).
"""

import math
import time

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import ResponseRelevancy, Faithfulness, ContextPrecision, ContextRecall
from ragas.run_config import RunConfig

# An LLM call is retried at most this many times before a row is treated as failed.
MAX_RETRIES = 2

# ragas.evaluate() defaults to RunConfig(timeout=180, max_retries=10, max_wait=60) for
# every metric's internal LLM calls whenever run_config isn't passed explicitly — even
# if a tighter RunConfig was set on the LLM/embeddings wrapper, evaluate() silently
# resets it (see ragas.evaluation.aevaluate: `run_config = run_config or RunConfig()`,
# then `metric.init(run_config)`). Passing this explicitly to every evaluate() call is
# what actually makes a bad/slow call fail fast instead of retrying for minutes.
# timeout=60 (not lower): context_precision alone can legitimately take ~50s on a
# typical deployment (one sequential LLM call per context chunk, plus RAGAS's own
# internal self-correction retry on malformed JSON) — cutting it off early just wastes
# the work and forces the caller's own retry loop to restart the whole row from scratch.
# max_workers=4 (not the default 16): each row only ever has 4 metric-level tasks in
# flight at once (one per configured metric), so 16 workers just means up to 4x more
# concurrent LLM calls hitting the deployment than are actually needed per row, adding
# unnecessary peak load with no speed benefit — this caps it to what's actually used.
EVAL_RUN_CONFIG = RunConfig(timeout=60, max_retries=MAX_RETRIES, max_wait=15, max_workers=4)

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

    Returns (scores, duration_s):
      scores    -- {metric_key: float} for every key in metric_keys (NaN if it
                    could not be computed after all attempts)
      duration_s -- total wall-clock time across all attempts
    """
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
    remaining_keys = list(metric_keys)
    for attempt in range(1, max_attempts + 1):
        if on_attempt is not None:
            on_attempt(attempt, max_attempts, remaining_keys)
        metrics = [METRIC_BUILDERS[k](llm, embeddings) for k in remaining_keys]
        try:
            result = evaluate(dataset=dataset, metrics=metrics, run_config=run_config, raise_exceptions=False)
            row = result.to_pandas().iloc[0].to_dict()
        except Exception:
            continue  # nothing computed this attempt; remaining_keys unchanged for the next one

        still_missing = []
        for k in remaining_keys:
            fallback = row.get("answer_relevancy") if k == "response_relevancy" else None
            value = row.get(k, fallback)
            value = float(value) if value is not None else float("nan")
            if math.isnan(value):
                still_missing.append(k)
            else:
                scores[k] = value
        remaining_keys = still_missing
        if not remaining_keys:
            break

    duration_s = time.perf_counter() - start_t
    return scores, duration_s
