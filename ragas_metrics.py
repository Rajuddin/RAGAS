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

# How many times evaluate_single_row redoes a whole metric (all of context_precision's
# chunk calls included) if it's still missing after an attempt. Not the same knob as
# LLM_CALL_MAX_RETRIES below -- see that constant's comment for why they're separate.
MAX_RETRIES = 1

# Passed as RunConfig.max_retries, this governs tenacity's stop_after_attempt() inside
# ragas's own per-LLM-call retry (ragas.run_config.add_async_retry) -- the retry for a
# single call (e.g. one context_precision chunk), not the whole-metric redo above.
# tenacity's stop_after_attempt(n) counts *n* as the total attempts, not retries beyond
# the first: stop_after_attempt(1) stops right after attempt 1, i.e. zero retries.
# This constant used to just reuse MAX_RETRIES (=1), which silently meant every single
# LLM call had NO retry at all -- a lone transient failure (e.g. an APIConnectionError
# from a burst of concurrent batch rows briefly exceeding a connection limit somewhere
# in the network path) failed that call immediately, with no chance to self-heal, and
# forced the expensive whole-metric redo instead (all of context_precision's already-
# succeeded chunk calls recomputed too, not just the one that failed). 2 gives each
# call one real retry (with tenacity's wait_random_exponential backoff, capped by
# max_wait below) at the point of failure, so a transient blip usually recovers without
# ever reaching the outer retry.
LLM_CALL_MAX_RETRIES = 2

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
# evaluate_single_row). Combined with MAX_RETRIES=1 (2 whole-metric attempts), a row's
# scoring is still bounded at ~240s worst case: LLM_CALL_MAX_RETRIES's per-call retry
# happens *inside* a single attempt's 120s timeout budget, not on top of it, so it
# doesn't add a separate multiplier -- a call that needs its one retry just uses more
# of that same 120s before either succeeding or (if it also fails) letting the 120s
# timeout or the outer whole-metric retry take over.
# max_workers=4 (not the default 16): each row only ever has 4 metric-level tasks in
# flight at once (one per configured metric), so 16 workers just means up to 4x more
# concurrent LLM calls hitting the deployment than are actually needed per row, adding
# unnecessary peak load with no speed benefit — this caps it to what's actually used.
EVAL_RUN_CONFIG = RunConfig(timeout=120, max_retries=LLM_CALL_MAX_RETRIES, max_wait=15, max_workers=4)

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

    The 4 metrics split into two groups that judge genuinely different things --
    and, importantly, two different notions of "the answer":
      Retrieval    = context_precision + context_recall. Both judge these against
                     the GROUND-TRUTH answer you supplied, not the RAG system's
                     generated one (that's how ragas itself defines them --
                     LLMContextPrecisionWithReference/LLMContextRecall both score
                     against `reference`, never `response`). context_recall asks:
                     does the retrieved set cover everything the ground-truth
                     answer needs, at all? context_precision asks: are the chunks
                     that were actually useful for the ground-truth answer ranked
                     near the top? Both judge the retrieved context itself,
                     independent of what the LLM's own generated answer said.
      Generation   = faithfulness + response_relevancy. Both judge the RAG
                     system's actual generated answer: faithfulness asks whether
                     it's grounded in whatever context it got; response_relevancy
                     asks whether it addresses the actual question asked.

    Augmentation -- how the retrieved context got assembled into the prompt the
    generation LLM actually saw (chunk ordering, truncation, formatting) -- isn't
    directly measured by any single RAGAS metric here, since this project evaluates
    an external RAG system's already-generated answer rather than controlling that
    assembly step itself. Its fingerprint is *good* retrieval scores paired with
    *poor* faithfulness: the right material was available, yet the generated answer
    isn't grounded in it. That pattern is equally consistent with a botched hand-off
    into the prompt (augmentation) or the model disregarding perfectly good context
    (generation) -- the two can't be told apart from scores alone, so both are
    surfaced together for that specific pattern rather than guessing which one.

    When retrieval is already bad, a low faithfulness score is much less
    diagnostic (hard to stay grounded in context that wasn't good to begin with),
    so it's noted as a likely symptom rather than raising Augmentation/Generation
    again -- fix retrieval first, then re-evaluate. response_relevancy, by
    contrast, is raised independently of retrieval quality: an off-topic generated
    answer is off-topic regardless of what was retrieved.

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
                "ground-truth answer needs (retrieval coverage, chunking, or a knowledge-base gap)"
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
                f"faithfulness {faithfulness:.2f} despite good retrieval — the generated answer "
                "isn't grounded in context that was actually available (augmentation hand-off "
                "into the prompt, or generation grounding)"
            )
            summary_parts.append(
                f"context_precision ({precision:.2f}) and context_recall ({recall:.2f}) are both "
                f"healthy, yet faithfulness is low ({faithfulness:.2f}) — the right material was "
                "retrieved, but the final generated answer isn't grounded in it. That points at "
                "either how the context was assembled into the prompt (augmentation) or how the "
                "model used it (generation)."
            )
        else:
            # context_precision and/or context_recall weren't computed for this row, so
            # retrieval quality is unverified, not confirmed good -- don't claim otherwise.
            focus_areas.extend(["Augmentation", "Generation"])
            reasons.append(
                f"faithfulness {faithfulness:.2f} — the generated answer isn't grounded in its "
                "context, but context_precision/context_recall weren't computed for this row, so "
                "retrieval quality can't be ruled out as a contributing cause either"
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
            f"response_relevancy {relevancy:.2f} — the generated answer doesn't directly address "
            "the question asked (generation focus/prompt)"
        )
        if faithfulness is not None and faithfulness >= LOW_SCORE_THRESHOLD:
            summary_parts.append(
                f"The generated answer is grounded in its context (faithfulness {faithfulness:.2f}) "
                f"but response_relevancy is low ({relevancy:.2f}) — it accurately reflects the "
                "context without actually answering the question that was asked."
            )
        else:
            summary_parts.append(
                f"response_relevancy is low ({relevancy:.2f}) — the generated answer doesn't "
                "directly address the question asked."
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


class _PromptRecorder:
    """Wraps one of a metric's internal PydanticPrompt instances (e.g.
    ContextPrecision.context_precision_prompt) to record every call's (input,
    output) pair, so evaluate_single_row can recover the judge LLM's structured
    reason/verdict output *after* the metric itself reduces it to a single float
    score and discards the rest. Every metric's output schema already includes a
    natural-language "reason" (or, for response_relevancy, the re-derived
    question) -- the LLM already generates and gets billed for these tokens; this
    just stops throwing that text away. Doesn't change behavior and makes no
    extra LLM calls: it only remembers what the wrapped prompt already returned.
    """

    def __init__(self, wrapped):
        self._wrapped = wrapped
        self.calls = []

    async def generate_multiple(self, *args, **kwargs):
        result = await self._wrapped.generate_multiple(*args, **kwargs)
        self.calls.append((kwargs.get("data"), result))
        return result

    async def generate(self, *args, **kwargs):
        result = await self._wrapped.generate(*args, **kwargs)
        self.calls.append((kwargs.get("data"), result))
        return result

    def __getattr__(self, name):
        return getattr(self._wrapped, name)


# Which attribute on each metric instance holds the PydanticPrompt worth wrapping
# with _PromptRecorder. faithfulness has two prompts (statement_generator_prompt
# just splits the answer into plain statement strings, no reasoning -- only
# nli_statements_prompt, which judges each statement against the context, carries
# a "reason"); only that one is recorded.
_METRIC_RECORDER_ATTR = {
    "context_precision": "context_precision_prompt",
    "context_recall": "context_recall_prompt",
    "faithfulness": "nli_statements_prompt",
    "response_relevancy": "question_generation",
}


def _extract_context_precision_reasons(calls):
    # One generate_multiple() call per retrieved context chunk (see
    # LLMContextPrecisionWithReference._ascore's `for context in retrieved_contexts`
    # loop), in the same order as the row's `contexts` list -- so `index` here lines
    # up with that list.
    chunks = []
    for i, (qac, verdicts) in enumerate(calls):
        if not verdicts:
            continue
        v = verdicts[0]
        chunks.append({
            "index": i,
            "context_preview": (qac.context[:200] if qac is not None else ""),
            "verdict": v.verdict,
            "reason": v.reason,
        })
    return {"chunks": chunks}


def _extract_context_recall_reasons(calls):
    # A single generate_multiple() call, joining every retrieved context into one
    # string and classifying each statement in the reference answer against it.
    if not calls or not calls[0][1]:
        return {"statements": []}
    classifications = calls[0][1][0].classifications
    return {
        "statements": [
            {"statement": c.statement, "attributed": c.attributed, "reason": c.reason}
            for c in classifications
        ]
    }


def _extract_faithfulness_reasons(calls):
    # A single generate() call (not generate_multiple -- faithfulness's NLI check
    # returns one NLIStatementOutput directly), judging each statement extracted
    # from the generated answer against the joined retrieved context.
    if not calls:
        return {"statements": []}
    _, result = calls[0]
    return {
        "statements": [
            {"statement": s.statement, "verdict": s.verdict, "reason": s.reason}
            for s in result.statements
        ]
    }


def _extract_response_relevancy_reasons(calls):
    # A single generate_multiple() call producing `strictness` (default 3)
    # re-derived questions from the answer -- no "reason" field exists for this
    # metric (it's scored via embedding similarity, not an LLM verdict), but the
    # re-derived questions themselves show why the answer did/didn't seem to
    # address the actual question, and `noncommittal` flags a vague/evasive answer.
    if not calls:
        return {"generated_questions": []}
    _, outputs = calls[0]
    return {
        "generated_questions": [
            {"question": o.question, "noncommittal": o.noncommittal} for o in outputs
        ]
    }


_METRIC_REASON_EXTRACTORS = {
    "context_precision": _extract_context_precision_reasons,
    "context_recall": _extract_context_recall_reasons,
    "faithfulness": _extract_faithfulness_reasons,
    "response_relevancy": _extract_response_relevancy_reasons,
}


def _build_metric_with_recorder(key, llm, embeddings):
    """Build a metric via METRIC_BUILDERS, then splice a _PromptRecorder into its
    reasoning prompt attribute (see _METRIC_RECORDER_ATTR) so its judge-LLM output
    can be recovered after scoring. Returns (metric, recorder)."""
    metric = METRIC_BUILDERS[key](llm, embeddings)
    attr = _METRIC_RECORDER_ATTR[key]
    recorder = _PromptRecorder(getattr(metric, attr))
    setattr(metric, attr, recorder)
    return metric, recorder


def build_metric_tooltip(key: str, score, reasons: dict, max_items: int = 5):
    """Build short hover-tooltip markdown explaining *why* a metric scored low, from
    the per-chunk/per-statement reason data _build_metric_with_recorder captured.
    Only surfaces the failing sub-verdicts (the chunks/statements that actually
    dragged the score down) -- the passing ones aren't why it's low, and dumping
    all of them (up to MAX_CONTEXTS=15 for context_precision) would bury the signal.

    Returns None when there's nothing worth showing: the score isn't low, is
    missing/NaN, or no reason data was captured for this metric (e.g. it never
    got its own attempt this row, or extraction silently failed -- see the
    try/except around _METRIC_REASON_EXTRACTORS in evaluate_single_row).
    """
    if score is None or (isinstance(score, float) and math.isnan(score)):
        return None
    if score >= LOW_SCORE_THRESHOLD or not reasons:
        return None

    if key == "context_precision":
        bad = [c for c in reasons.get("chunks", []) if not c["verdict"]]
        if not bad:
            return None
        lines = [f"- Context[{c['index']}]: {c['reason']}" for c in bad[:max_items]]
        if len(bad) > max_items:
            lines.append(f"- ...and {len(bad) - max_items} more chunk(s) judged not useful")
        return (
            "**Why this is low — context chunks judged not useful for arriving at the "
            "ground-truth answer:**\n" + "\n".join(lines)
        )

    if key == "context_recall":
        bad = [s for s in reasons.get("statements", []) if not s["attributed"]]
        if not bad:
            return None
        lines = [f"- \"{s['statement']}\": {s['reason']}" for s in bad[:max_items]]
        if len(bad) > max_items:
            lines.append(f"- ...and {len(bad) - max_items} more statement(s) not covered")
        return (
            "**Why this is low — ground-truth statements not covered by retrieved "
            "context:**\n" + "\n".join(lines)
        )

    if key == "faithfulness":
        bad = [s for s in reasons.get("statements", []) if not s["verdict"]]
        if not bad:
            return None
        lines = [f"- \"{s['statement']}\": {s['reason']}" for s in bad[:max_items]]
        if len(bad) > max_items:
            lines.append(f"- ...and {len(bad) - max_items} more statement(s) not grounded")
        return (
            "**Why this is low — generated-answer statements not grounded in retrieved "
            "context:**\n" + "\n".join(lines)
        )

    if key == "response_relevancy":
        questions = reasons.get("generated_questions", [])
        if not questions:
            return None
        lines = [
            f"- Re-derived question: \"{q['question']}\""
            + (" (generated answer flagged as vague/evasive)" if q["noncommittal"] else "")
            for q in questions[:max_items]
        ]
        return (
            "**Why this is low — questions re-derived from the generated answer, compared "
            "against what was actually asked:**\n" + "\n".join(lines)
        )

    return None


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
    rebuild_clients=None,
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

    rebuild_clients, if given, is a zero-arg callable returning a fresh (llm,
    embeddings) pair, called before *every* attempt (including the first) so each
    attempt gets its own never-before-used LLM/embeddings client -- replacing the
    initial `llm`/`embeddings` arguments for that attempt only, not building on top
    of them. This matters on Windows: ragas.evaluate() calls asyncio.run() fresh on
    every attempt (a new event loop each time), and reusing one async HTTP client
    across those separate loops is what triggers a ProactorEventLoop.close() hang
    (see WindowsSelectorEventLoopPolicy above, and ROW_HARD_TIMEOUT_S in
    app_pages/rag_evaluation.py) -- building fresh per row alone still leaves this
    exact reuse *within* a row, across its own retry attempts. Omit this if the
    caller doesn't have a cheap way to rebuild clients (e.g. session-scoped
    fixtures shared across many rows, like tests/test_ragas_evaluation.py) -- the
    passed-in `llm`/`embeddings` are then reused across attempts as before.

    Raises TooManyContextsError if len(contexts) > MAX_CONTEXTS, before making any
    LLM calls — see the comment above MAX_CONTEXTS for why that limit exists.

    Returns (scores, duration_s, errors, token_usage, metric_reasons):
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
      metric_reasons -- {metric_key: {...}} the judge LLM's own per-chunk/per-statement
                    reason data for whichever metrics computed (see
                    _METRIC_REASON_EXTRACTORS for the shape per metric key); use
                    build_metric_tooltip(key, score, reasons) to turn this into
                    display text. Missing entries (not even an empty dict) for a
                    metric key mean recording/extraction found nothing to capture.
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
    metric_reasons = {}
    input_tokens = 0
    output_tokens = 0
    remaining_keys = list(metric_keys)
    for attempt in range(1, max_attempts + 1):
        if rebuild_clients is not None:
            llm, embeddings = rebuild_clients()
        if on_attempt is not None:
            on_attempt(attempt, max_attempts, remaining_keys)
        built = [_build_metric_with_recorder(k, llm, embeddings) for k in remaining_keys]
        metrics = [m for m, _ in built]
        recorders = dict(zip(remaining_keys, (r for _, r in built)))
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
        except (ValueError, IndexError):
            # ValueError: no cost_cb at all (token_usage_parser wasn't honored).
            # IndexError: cost_cb exists but recorded zero calls this attempt (e.g.
            # every metric errored before its LLM call returned) -- ragas's own
            # total_tokens() indexes usage_data[0] unconditionally and doesn't
            # guard against it being empty. Either way this is best-effort token
            # accounting; it must never take down the row's actual scoring.
            pass

        for k, recorder in recorders.items():
            try:
                metric_reasons[k] = _METRIC_REASON_EXTRACTORS[k](recorder.calls)
            except Exception:
                pass  # best-effort only -- never let reason-extraction break scoring

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
    return scores, duration_s, errors, token_usage, metric_reasons
