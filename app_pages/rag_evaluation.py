"""
rag_evaluation.py
Streamlit page for running a single RAGAS evaluation.

- If the RAG API is available (toggle checked), answer + contexts are fetched
  from the configured RAG endpoint.
- If not, the answer and contexts (3-4 chunks, one per line) are entered
  manually in the UI.
"""

import json
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
from datasets import Dataset

from ragas import evaluate

from config_loader import (
    load_config,
    build_langchain_llm,
    build_langchain_embeddings,
    check_llm_credentials,
    save_llm_config,
    save_rag_api_config,
    LLMConfig,
    RAGApiConfig,
)
from rag_client import query_rag_system, build_payload
from ragas_metrics import (
    MAX_RETRIES,
    EVAL_RUN_CONFIG,
    METRIC_BUILDERS,
    METRIC_LABELS,
    configured_metric_keys as _configured_metric_keys,
    evaluate_single_row,
    TooManyContextsError,
)
import allure_utils

ALLURE_REPORT_DIR = Path(__file__).parent.parent / "allure-report"


def _fmt_ms(ms) -> str:
    """Epoch milliseconds (as stored by _process_row/_BatchJob) -> local HH:MM:SS,
    or "—" if not known yet (e.g. a row that hasn't started)."""
    if ms is None:
        return "—"
    return datetime.fromtimestamp(ms / 1000).strftime("%H:%M:%S")


def _score_label(score: float) -> str:
    if score >= 0.85:
        return "EXCELLENT"
    elif score >= 0.70:
        return "GOOD"
    elif score >= 0.50:
        return "MODERATE"
    else:
        return "POOR"


def _has_rag_api(cfg) -> bool:
    endpoint = cfg.rag_api.endpoint or ""
    return bool(endpoint) and "your-rag-api" not in endpoint


def _friendly_llm_error(exc: Exception) -> str:
    """Map common LLM provider errors to an actionable message for the LLM Configuration sidebar."""
    text = str(exc)
    if "401" in text or "AuthenticationError" in text or "invalid subscription key" in text.lower():
        return (
            "The LLM provider rejected your API key (401 Unauthorized). "
            "Check the API Key (and Azure Endpoint, if using Azure) in the LLM Configuration sidebar."
        )
    if "404" in text or "DeploymentNotFound" in text:
        return (
            "The LLM provider could not find the model/deployment (404 Not Found). "
            "Check the Model or Deployment Name in the LLM Configuration sidebar."
        )
    if "429" in text or "RateLimitError" in text:
        return "The LLM provider rate-limited this request (429 Too Many Requests). Wait a moment and try again."
    if "Connection" in text or "Timeout" in text or "timed out" in text.lower():
        return (
            "Could not reach the LLM provider (connection/timeout error). "
            "Check the Azure Endpoint / network connectivity in the LLM Configuration sidebar."
        )
    if _is_transient_json_parse_error(exc):
        return (
            "The LLM's structured-output response could not be parsed as JSON, even after RAGAS's "
            "built-in self-correction retry. This is a known, occasional formatting glitch with some "
            "models/prompts, not a configuration problem. It usually succeeds on a retry."
        )
    return f"The LLM evaluation call failed: {text}"


def _is_transient_json_parse_error(exc: Exception) -> bool:
    text = str(exc)
    return "json_invalid" in text or ("validation error" in text.lower() and "Invalid JSON" in text)


def _validate_batch_items(items, use_api: bool) -> list:
    """Check each batch item has the fields required for the current mode. Returns a list of error strings."""
    errors = []
    for i, item in enumerate(items):
        test_id = item.get("test_id", f"row {i}")
        missing = [k for k in ("query", "ground_truth") if not item.get(k)]
        if not use_api:
            if not item.get("generated_answer"):
                missing.append("generated_answer")
            contexts = item.get("contexts")
            if not contexts or not isinstance(contexts, list) or len(contexts) == 0:
                missing.append("contexts (non-empty list)")
        if missing:
            errors.append(f"{test_id}: missing {', '.join(missing)}")
    return errors


class _BatchJob:
    """Holds a running/completed batch evaluation's state, shared between a
    detached background thread and the Streamlit script.

    Every widget interaction (a checkbox, another button, anything) tears down
    and restarts the whole Streamlit script — code running inline in the script
    gets abandoned mid-way. To survive that, the actual evaluation work must live
    somewhere that isn't tied to any single script execution: this object, held in
    st.session_state, updated by a plain threading.Thread that keeps running
    regardless of how many times the page reruns. The page just reads snapshot()
    on each render and displays whatever's currently true.
    """

    def __init__(self, items, test_ids, metric_cols, metric_labels):
        self.lock = threading.Lock()
        self.items = items  # kept so the live view can show query/ground truth per row
        self.test_ids = test_ids
        self.metric_cols = metric_cols
        self.metric_labels = metric_labels
        self.status = "running"  # running | done | cancelled | error
        self.row_status = ["queued"] * len(test_ids)  # short current-stage text
        self.row_start_ms = [None] * len(test_ids)  # epoch ms, set the moment a row starts
        self.row_answer = [None] * len(test_ids)  # fetched/given answer, once known
        self.row_contexts = [None] * len(test_ids)  # fetched/given contexts, once known
        self.row_scores = [None] * len(test_ids)  # {metric_key: value}, once computed
        self.row_records = [None] * len(test_ids)  # final merged row dict, once the row finishes
        self.completed = 0
        self.error_message = None
        self.cancel_event = threading.Event()
        self.finalized = False  # True once Allure results have been written for this job
        self.failed_case_details = []
        self.allure_write_errors = []  # ["test_id: exc", ...] for cases whose Allure write failed

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "status": self.status,
                "row_status": list(self.row_status),
                "row_start_ms": list(self.row_start_ms),
                "row_answer": list(self.row_answer),
                "row_contexts": list(self.row_contexts),
                "row_scores": list(self.row_scores),
                "row_records": list(self.row_records),
                "completed": self.completed,
                "error_message": self.error_message,
            }

    def set_row_status(self, i: int, message: str) -> None:
        with self.lock:
            self.row_status[i] = message

    def set_row_start(self, i: int, start_ms: int) -> None:
        with self.lock:
            self.row_start_ms[i] = start_ms

    def set_row_fetch(self, i: int, answer, contexts) -> None:
        with self.lock:
            self.row_answer[i] = answer
            self.row_contexts[i] = contexts

    def set_row_scores(self, i: int, scores: dict) -> None:
        with self.lock:
            self.row_scores[i] = dict(scores)

    def set_row_result(self, i: int, row_dict: dict) -> None:
        with self.lock:
            self.row_records[i] = row_dict
            self.completed += 1


def _run_batch_job(job: "_BatchJob", items, use_api, rag_api_cfg, llm_cfg, concurrency) -> None:
    """Runs entirely in a background thread, independent of any Streamlit script
    run — must never call st.* directly (there's no script-run context to route
    to here). All progress is written into `job`; the page polls job.snapshot()."""
    test_ids = job.test_ids
    metric_cols = job.metric_cols
    metric_labels = job.metric_labels
    ROW_STAGGER_SECONDS = 0.4

    def _process_row_inner(i: int) -> None:
        test_id = test_ids[i]
        item = items[i]
        query_i = item["query"]
        ground_truth_i = item["ground_truth"]

        # Built fresh per row rather than once and shared across every concurrently
        # running row: reusing one async HTTP client across the many separate event
        # loops that create/destroy across concurrent rows was a contributor to the
        # Windows asyncio.run()/loop.close() hang (see ROW_HARD_TIMEOUT_S below) --
        # ragas_metrics's WindowsSelectorEventLoopPolicy switch is what actually
        # eliminates that hang at the source, this just removes one more source of
        # cross-row contention as defense in depth.
        llm = build_langchain_llm(llm_cfg)
        embeddings = build_langchain_embeddings(llm_cfg)

        start_ms = int(time.time() * 1000)
        job.set_row_start(i, start_ms)

        if use_api:
            job.set_row_status(i, "Calling RAG API...")
            try:
                answer_i, contexts_i = query_rag_system(item, rag_api_cfg)
            except Exception as exc:
                job.set_row_status(i, f"RAG API call failed: {exc}")
                now_ms = int(time.time() * 1000)
                job.set_row_result(
                    i,
                    {
                        "test_id": test_id,
                        "fetch_error": str(exc),
                        "user_input": query_i,
                        "reference": ground_truth_i,
                        "response": "",
                        "retrieved_contexts": [],
                        "duration_s": 0.0,
                        "start_ms": start_ms,
                        "stop_ms": now_ms,
                        **{k: float("nan") for k in metric_cols},
                    },
                )
                return
        else:
            answer_i = item["generated_answer"]
            contexts_i = item["contexts"]

        job.set_row_fetch(i, answer_i, contexts_i)

        def _on_attempt(attempt, max_attempts, retry_keys):
            if attempt == 1:
                job.set_row_status(i, "Scoring metrics...")
            else:
                labels = ", ".join(metric_labels[k] for k in retry_keys)
                job.set_row_status(i, f"Retrying {labels} (attempt {attempt}/{max_attempts})...")

        try:
            scores, duration_s, errors = evaluate_single_row(
                llm, embeddings, metric_cols, query_i, answer_i, contexts_i, ground_truth_i,
                on_attempt=_on_attempt,
            )
        except TooManyContextsError as exc:
            job.set_row_status(i, f"Scoring failed: {exc}")
            now_ms = int(time.time() * 1000)
            job.set_row_result(
                i,
                {
                    "test_id": test_id,
                    "scoring_error": str(exc),
                    "user_input": query_i,
                    "reference": ground_truth_i,
                    "response": answer_i,
                    "retrieved_contexts": contexts_i,
                    "duration_s": 0.0,
                    "start_ms": start_ms,
                    "stop_ms": now_ms,
                    **{k: float("nan") for k in metric_cols},
                },
            )
            return
        stop_ms = int(time.time() * 1000)

        job.set_row_scores(i, scores)
        missing = [metric_labels[k] for k, v in scores.items() if v != v]
        if missing:
            reasons = "; ".join(f"{metric_labels[k]}: {msg}" for k, msg in errors.items())
            status = f"Missing {', '.join(missing)}" + (f" ({reasons})" if reasons else "") + f" ({duration_s:.1f}s)"
        else:
            status = f"Done ({duration_s:.1f}s)"
        job.set_row_status(i, status)

        row_dict = dict(scores)
        row_dict.update(
            {
                "test_id": test_id,
                "user_input": query_i,
                "response": answer_i,
                "reference": ground_truth_i,
                "retrieved_contexts": contexts_i,
                "duration_s": duration_s,
                "start_ms": start_ms,
                "stop_ms": stop_ms,
                "errors": errors,
            }
        )
        job.set_row_result(i, row_dict)

    def _process_row(i: int) -> None:
        """Catch-all around _process_row_inner: this runs inside a ThreadPoolExecutor
        worker, and results are collected via wait(futures, ...) (see below), not
        future.result() -- so any exception _process_row_inner doesn't already
        handle itself (TooManyContextsError, a RAG API failure) would otherwise be
        silently swallowed inside the Future, never surfaced, and the row would sit
        at whatever status it last had forever (indistinguishable from a genuine
        hang). This guarantees every row ends up with a set_row_result call one way
        or another, with a clear message, no matter what goes wrong."""
        try:
            _process_row_inner(i)
        except Exception as exc:
            now_ms = int(time.time() * 1000)
            item = items[i]
            job.set_row_status(i, f"Unexpected error: {exc}")
            job.set_row_result(
                i,
                {
                    "test_id": test_ids[i],
                    "scoring_error": f"Unexpected error ({type(exc).__name__}): {exc}",
                    "user_input": item.get("query", ""),
                    "reference": item.get("ground_truth", ""),
                    "response": job.row_answer[i] or "",
                    "retrieved_contexts": job.row_contexts[i] or [],
                    "duration_s": 0.0,
                    "start_ms": job.row_start_ms[i],
                    "stop_ms": now_ms,
                    **{k: float("nan") for k in metric_cols},
                },
            )

    # Safety net against a Windows-specific asyncio hang: asyncio.run() (which
    # ragas.evaluate() calls internally, fresh, on every attempt) can leave its
    # ProactorEventLoop stuck forever inside loop.close() -> _poll() during its own
    # teardown -- *after* the actual LLM work already finished -- if an async HTTP
    # client's connection gets reused across the many separate event loops that
    # create/destroy across a row's retry attempts. This happens outside any awaited
    # call, so EVAL_RUN_CONFIG's asyncio.wait_for timeout never sees it and can't
    # protect against it; the affected worker thread is stuck permanently (Python
    # can't force-kill a thread). ragas_metrics now switches Windows to
    # WindowsSelectorEventLoopPolicy specifically to eliminate this at the source --
    # this is a backstop for if it still happens. Per an explicit requirement that
    # one test case should finish within a few minutes or fail with a clear error
    # rather than wait indefinitely: sized just above the legitimate worst case for a
    # row that's genuinely still working, not hung (EVAL_RUN_CONFIG.timeout=120s x up
    # to 2 attempts = 240s of scoring, plus the RAG API call), so it fires only for a
    # real hang while still keeping the *outer* ceiling close to that requirement.
    ROW_HARD_TIMEOUT_S = 5 * 60

    pool = ThreadPoolExecutor(max_workers=concurrency)
    try:
        futures = {}
        for i in range(len(test_ids)):
            if job.cancel_event.is_set():
                job.set_row_status(i, "cancelled (not started)")
                continue
            futures[pool.submit(_process_row, i)] = i
            if i < concurrency - 1:
                time.sleep(ROW_STAGGER_SECONDS)

        _done, not_done = wait(futures, timeout=ROW_HARD_TIMEOUT_S)
        for f in not_done:
            i = futures[f]
            job.set_row_status(
                i,
                f"Stuck — exceeded the {ROW_HARD_TIMEOUT_S // 60}min hard timeout "
                "(a Windows asyncio hang, not a slow API call)",
            )
            job.set_row_result(
                i,
                {
                    "test_id": test_ids[i],
                    "scoring_error": (
                        f"Row exceeded the {ROW_HARD_TIMEOUT_S // 60}-minute hard timeout and was "
                        "abandoned. This is not a slow API call -- its worker thread is stuck inside "
                        "Python's own asyncio event-loop cleanup (a known Windows-specific hang) and "
                        "is leaked in the background; restart the app to fully clear it."
                    ),
                    "user_input": items[i].get("query", ""),
                    "reference": items[i].get("ground_truth", ""),
                    "response": job.row_answer[i] or "",
                    "retrieved_contexts": job.row_contexts[i] or [],
                    "duration_s": 0.0,
                    "start_ms": job.row_start_ms[i],
                    "stop_ms": int(time.time() * 1000),
                    **{k: float("nan") for k in metric_cols},
                },
            )

        with job.lock:
            job.status = "cancelled" if job.cancel_event.is_set() else "done"
    except Exception as exc:
        with job.lock:
            job.status = "error"
            job.error_message = str(exc)
    finally:
        # wait=False: never block here on a leaked/hung worker thread. Exiting via
        # `with ThreadPoolExecutor(...)` instead would call shutdown(wait=True) and
        # reintroduce the exact whole-batch-blocks-forever failure this exists to
        # prevent (see ROW_HARD_TIMEOUT_S above).
        pool.shutdown(wait=False)


@st.cache_resource
def get_app_config():
    return load_config()


if st.button("← Back to Dashboard"):
    st.switch_page("app_pages/home.py")

st.title("RAG Evaluation using RAGAS")

try:
    app_config = get_app_config()
except Exception as exc:
    st.error(
        f"Could not load config.yaml: {exc}. Fix the file (check it's valid YAML and has the "
        "sections config_loader.load_config expects), then reload this page."
    )
    with st.expander("Technical details"):
        st.code(str(exc))
    st.stop()

# ─────────────────────────────────────────────
# LLM provider configuration (sidebar)
# ─────────────────────────────────────────────

st.sidebar.header("LLM Configuration")

provider = st.sidebar.radio(
    "Provider",
    options=["azure"],  # "openai" hidden for now -- re-add here to bring it back
    format_func=lambda p: {"openai": "OpenAI", "azure": "Azure OpenAI"}[p],
)

if provider == "openai":
    openai_api_key = st.sidebar.text_input(
        "OpenAI API Key",
        value=app_config.llm.api_key if app_config.llm.provider == "openai" else "",
        type="password",
    )
    openai_model = st.sidebar.text_input(
        "Model",
        value=app_config.llm.model if app_config.llm.provider == "openai" else "gpt-4o",
    )
    llm_cfg = LLMConfig(provider="openai", api_key=openai_api_key, model=openai_model)
else:
    azure_api_key = st.sidebar.text_input(
        "Azure API Key",
        value=app_config.llm.api_key if app_config.llm.provider == "azure" else "",
        type="password",
    )
    azure_endpoint = st.sidebar.text_input(
        "Azure Endpoint",
        value=app_config.llm.azure_endpoint if app_config.llm.provider == "azure" else "",
        placeholder="https://your-resource.openai.azure.com/",
    )
    azure_deployment = st.sidebar.text_input(
        "Deployment Name (chat model)",
        value=app_config.llm.deployment_name if app_config.llm.provider == "azure" else "",
    )
    azure_embedding_deployment = st.sidebar.text_input(
        "Deployment Name (embedding model)",
        value=app_config.llm.embedding_model_name if app_config.llm.provider == "azure" else "text-embedding-ada-002",
    )
    # No API Version field: auto-detected based on the endpoint shape.
    azure_api_version = "preview" if "/openai/v1" in azure_endpoint else "2024-02-01"
    llm_cfg = LLMConfig(
        provider="azure",
        api_key=azure_api_key,
        azure_endpoint=azure_endpoint,
        deployment_name=azure_deployment,
        embedding_model_name=azure_embedding_deployment,
        api_version=azure_api_version,
    )

if st.sidebar.button("Save Config"):
    try:
        if provider == "openai":
            save_llm_config(provider="openai", openai_fields={"api_key": openai_api_key, "model": openai_model})
        else:
            save_llm_config(
                provider="azure",
                azure_fields={
                    "api_key": azure_api_key,
                    "azure_endpoint": azure_endpoint,
                    "deployment_name": azure_deployment,
                    "embedding_model_name": azure_embedding_deployment,
                    "api_version": azure_api_version,
                },
            )
        get_app_config.clear()
        st.sidebar.success("Saved to config.yaml")
    except Exception as exc:
        st.sidebar.error(f"Could not save config.yaml: {exc}")

st.sidebar.divider()

st.sidebar.header("Allure Report")
allure_col1, allure_col2 = st.sidebar.columns(2)
if allure_col1.button("Generate & Open Report"):
    try:
        with st.spinner("Building Allure report..."):
            allure_utils.generate_report(app_config.allure_results_dir, ALLURE_REPORT_DIR)
            st.session_state["allure_report_url"] = allure_utils.serve_report(ALLURE_REPORT_DIR)
    except Exception as exc:
        st.sidebar.error(f"Could not generate Allure report: {exc}")

if allure_col2.button("Clear Previous Results"):
    try:
        allure_utils.clear_results(app_config.allure_results_dir, ALLURE_REPORT_DIR)
        st.session_state.pop("allure_report_url", None)
        st.sidebar.success("Cleared. Run an evaluation to start a fresh report.")
    except Exception as exc:
        st.sidebar.error(f"Could not clear results: {exc}")

if st.session_state.get("allure_report_url"):
    st.sidebar.link_button("Open Allure Report ↗", st.session_state["allure_report_url"])

st.sidebar.divider()

st.sidebar.header("Metrics")
selected_metric_keys = st.sidebar.multiselect(
    "Metrics to evaluate",
    options=list(METRIC_BUILDERS.keys()),
    default=_configured_metric_keys(app_config),
    format_func=lambda k: METRIC_LABELS[k],
    help="Defaults to config.yaml's ragas.metrics list. Change here to override for this run only "
    "— context_precision and context_recall are the slowest (one LLM call per context chunk each).",
)

st.sidebar.divider()

use_api = st.checkbox(
    "Fetch answer & contexts from RAG API",
    value=_has_rag_api(app_config),
    help=f"Endpoint: {app_config.rag_api.endpoint}",
)

rag_api_cfg = app_config.rag_api

if use_api:
    with st.expander("RAG API Configuration", expanded=True):
        rag_endpoint = st.text_input(
            "RAG API Endpoint",
            value=app_config.rag_api.endpoint or "",
            placeholder="https://your-rag-api/ask",
        )
        rag_timeout = st.number_input(
            "Timeout (seconds)",
            value=int(app_config.rag_api.timeout or 30),
            min_value=1,
            step=5,
        )
        existing_headers = {k: v for k, v in (app_config.rag_api.headers or {}).items() if k.lower() != "content-type"}
        existing_auth_name, existing_auth_value = next(iter(existing_headers.items()), ("Authorization", ""))

        set_custom_header = st.toggle("Set Custom Header", value=bool(existing_headers))
        rag_headers = {"Content-Type": "application/json"}
        if set_custom_header:
            rag_auth_header_name = st.text_input("Header Name", value=existing_auth_name or "Authorization")
            rag_auth_header_value = st.text_input(
                "Header Value",
                value=existing_auth_value,
                type="password",
                placeholder="Bearer your-token",
            )
            if rag_auth_header_name.strip() and rag_auth_header_value.strip():
                rag_headers[rag_auth_header_name.strip()] = rag_auth_header_value.strip()

        st.markdown("**Request Payload**")
        st.caption(
            "Define each key of the JSON body sent to the RAG API. Use **From JSON field** for "
            "values that differ per test case (any key present in that test case's JSON object, "
            "e.g. `query`), and **Hardcoded value** for values that stay the same across every "
            "test case (e.g. payload key `Type` with hardcoded value `text`)."
        )

        if "rag_payload_params" not in st.session_state:
            st.session_state.rag_payload_params = [dict(p) for p in app_config.rag_api.payload_params]

        header_cols = st.columns([3, 3, 4, 1])
        header_cols[0].caption("Payload Key")
        header_cols[1].caption("Source")
        header_cols[2].caption("JSON Field Name / Hardcoded Value")

        remove_idx = None
        for i, param in enumerate(st.session_state.rag_payload_params):
            row_cols = st.columns([3, 3, 4, 1])
            param["name"] = row_cols[0].text_input(
                "Payload Key",
                value=param.get("name", ""),
                key=f"pp_name_{i}",
                placeholder="e.g. question",
                label_visibility="collapsed",
            )
            source_options = ["field", "static"]
            current_source = param.get("source") if param.get("source") in source_options else "field"
            param["source"] = row_cols[1].selectbox(
                "Source",
                options=source_options,
                index=source_options.index(current_source),
                format_func=lambda s: "From JSON field" if s == "field" else "Hardcoded value",
                key=f"pp_source_{i}",
                label_visibility="collapsed",
            )
            if param["source"] == "field":
                param["field"] = row_cols[2].text_input(
                    "JSON Field Name",
                    value=param.get("field", ""),
                    key=f"pp_field_{i}",
                    placeholder="e.g. query",
                    label_visibility="collapsed",
                )
            else:
                param["value"] = row_cols[2].text_input(
                    "Hardcoded Value",
                    value=param.get("value", ""),
                    key=f"pp_value_{i}",
                    placeholder='e.g. text   or   123   or   ["a","b"]',
                    label_visibility="collapsed",
                )
            if row_cols[3].button("✕", key=f"pp_remove_{i}", help="Remove this parameter"):
                remove_idx = i

        if remove_idx is not None:
            st.session_state.rag_payload_params.pop(remove_idx)
            st.rerun()

        if st.button("+ Add Parameter"):
            st.session_state.rag_payload_params.append({"name": "", "source": "field", "field": "", "value": ""})
            st.rerun()

        rag_payload_params = st.session_state.rag_payload_params
        st.caption("Preview (using a sample test case):")
        st.code(
            json.dumps(
                build_payload({"query": "example query", "ground_truth": "example ground truth"}, rag_payload_params),
                indent=2,
            ),
            language="json",
        )

        has_field_overrides = bool(
            app_config.rag_api.answer_field or app_config.rag_api.contexts_field or app_config.rag_api.context_item_field
        )
        set_response_fields = st.toggle("Set Answer / Contexts Field Manually", value=has_field_overrides)
        rag_answer_field = ""
        rag_contexts_field = ""
        rag_context_item_field = ""
        if set_response_fields:
            col1, col2, col3 = st.columns(3)
            with col1:
                rag_answer_field = st.text_input(
                    "Answer Field",
                    value=app_config.rag_api.answer_field or "",
                    placeholder="e.g. answer or message.generated_response",
                )
            with col2:
                rag_contexts_field = st.text_input(
                    "Contexts Field",
                    value=app_config.rag_api.contexts_field or "",
                    placeholder="e.g. retrieved_docs or message.context[0].context",
                )
            with col3:
                rag_context_item_field = st.text_input(
                    "Context Item Field",
                    value=app_config.rag_api.context_item_field or "",
                    placeholder="e.g. page_content",
                    help="Only needed if each context entry is an object, not a plain string.",
                )
        else:
            st.caption(
                "Answer/contexts field names are auto-detected from common shapes "
                "(answer/contexts, result/source_documents, answer/retrieved_docs, etc). "
                "Turn this on only if you see an 'Unexpected RAG API response structure' error."
            )

        with st.expander("RAG API Response Field Definitions"):
            st.markdown(
                "- **Answer Field** — the JSON key holding the model's generated answer "
                "(e.g. `answer`, `result`, `output_text`, `response`).\n"
                "  - Example: `\"answer\": \"Paris is the capital of France.\"`\n"
                "- **Contexts Field** — the JSON key holding the list of retrieved document "
                "chunks (e.g. `contexts`, `retrieved_docs`, `source_documents`, `input_documents`).\n"
                "  - Example: `\"contexts\": [\"France is a country in Europe.\", \"Paris is its capital.\"]`\n"
                "- **Context Item Field** — if each entry in the contexts list is an object "
                "rather than a plain string, this is the key inside that object holding the "
                "chunk text (e.g. `page_content`, `content`, `text`).\n\n"
                "If your answer/contexts are nested inside the response instead of being a "
                "top-level key, use a dotted path with `[index]` for array entries, e.g. "
                "`message.generated_response` or `message.context[0].context` for the response "
                "shape below."
            )
            st.code(
                json.dumps(
                    {
                        "answer": "Paris is the capital of France.",
                        "retrieved_docs": [
                            {"page_content": "France is a country in Europe."},
                            {"page_content": "Paris is its capital."},
                        ],
                    },
                    indent=2,
                ),
                language="json",
            )
            st.caption("Nested example — Answer Field `message.generated_response`, Contexts Field `message.context[0].context`:")
            st.code(
                json.dumps(
                    {
                        "status": 200,
                        "message": {
                            "generated_response": "Paris is the capital of France.",
                            "context": [
                                {
                                    "tool": "azure_ai_search_call_output",
                                    "context": [
                                        {"content": "France is a country in Europe."},
                                        {"content": "Paris is its capital."},
                                    ],
                                }
                            ],
                        },
                    },
                    indent=2,
                ),
                language="json",
            )

        rag_api_cfg = RAGApiConfig(
            endpoint=rag_endpoint,
            timeout=int(rag_timeout),
            headers=rag_headers,
            answer_field=rag_answer_field.strip() or None,
            contexts_field=rag_contexts_field.strip() or None,
            context_item_field=rag_context_item_field.strip() or None,
            payload_params=rag_payload_params,
        )

        if st.button("Save Config", key="save_rag_api_config"):
            try:
                save_rag_api_config(
                    endpoint=rag_endpoint,
                    timeout=int(rag_timeout),
                    headers=rag_headers,
                    answer_field=rag_api_cfg.answer_field,
                    contexts_field=rag_api_cfg.contexts_field,
                    context_item_field=rag_api_cfg.context_item_field,
                    payload_params=rag_payload_params,
                )
                get_app_config.clear()
                st.success("Saved to config.yaml")
            except Exception as exc:
                st.error(f"Could not save config.yaml: {exc}")

st.divider()
eval_mode = st.radio("Evaluation Mode", ["Single Case", "Batch (JSON file)"], horizontal=True)

if eval_mode == "Single Case":
    query = st.text_area("Query", height=80)
    ground_truth = st.text_area("Ground Truth", height=80)

    manual_answer = None
    manual_contexts_raw = None

    if not use_api:
        manual_answer = st.text_area("Generated Answer", height=100)
        manual_contexts_raw = st.text_area(
            "Contexts (one per line — 3 or 4 chunks)",
            height=150,
            placeholder="Context chunk 1\nContext chunk 2\nContext chunk 3",
        )
else:
    if use_api:
        example_schema = [{"test_id": "TC001", "query": "What is X?", "ground_truth": "X is ..."}]
    else:
        example_schema = [
            {
                "test_id": "TC001",
                "query": "What is X?",
                "ground_truth": "X is ...",
                "generated_answer": "X is ...",
                "contexts": ["context chunk 1", "context chunk 2", "context chunk 3"],
            }
        ]
    st.caption(
        "Upload (or point to) a JSON file containing a list of test cases. "
        + ("Since the RAG API is enabled, each item only needs `query` and `ground_truth` "
           "(answer & contexts are fetched automatically)."
           if use_api else
           "Since manual entry is selected, each item must also include `generated_answer` "
           "and `contexts` (a list of 3-4 strings).")
    )
    with st.expander("Expected JSON schema"):
        st.code(json.dumps(example_schema, indent=2), language="json")

    batch_path = st.text_input(
        "JSON File Path (on this machine)",
        placeholder="test_data/my_tests.json",
    )
    uploaded_file = st.file_uploader("...or upload a JSON file", type=["json"])

run_single = eval_mode == "Single Case" and st.button("Run Evaluation", type="primary")
run_batch = eval_mode == "Batch (JSON file)" and st.button("Run Batch Evaluation", type="primary")

if run_single:
    if not selected_metric_keys:
        st.error("Select at least one metric to evaluate in the sidebar.")
        st.stop()

    if not query.strip() or not ground_truth.strip():
        st.error("Query and Ground Truth are required.")
        st.stop()

    if use_api:
        if not rag_api_cfg.endpoint.strip():
            st.error("RAG API Endpoint is required.")
            st.stop()
        with st.spinner("Calling RAG API..."):
            try:
                answer, contexts = query_rag_system({"query": query, "ground_truth": ground_truth}, rag_api_cfg)
            except Exception as exc:
                st.error(f"RAG API call failed: {exc}")
                st.stop()
        st.subheader("RAG API Response")
        st.write("**Generated Answer:**", answer)
        st.write("**Contexts:**")
        for i, c in enumerate(contexts, 1):
            st.text(f"{i}. {c}")
    else:
        answer = (manual_answer or "").strip()
        contexts = [line.strip() for line in (manual_contexts_raw or "").splitlines() if line.strip()]
        if not answer or not contexts:
            st.error("Generated Answer and at least one Context are required.")
            st.stop()

    metric_keys = selected_metric_keys

    MAX_ATTEMPTS = MAX_RETRIES + 1
    scores = None
    last_exc = None
    start_ms = int(time.time() * 1000)
    start_t = time.perf_counter()
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            with st.spinner(f"Running RAGAS evaluation... (attempt {attempt}/{MAX_ATTEMPTS})"):
                llm = build_langchain_llm(llm_cfg)
                embeddings = build_langchain_embeddings(llm_cfg)
                check_llm_credentials(llm)

                dataset = Dataset.from_dict(
                    {
                        "question": [query],
                        "answer": [answer],
                        "contexts": [contexts],
                        "ground_truth": [ground_truth],
                    }
                )

                result = evaluate(
                    dataset=dataset,
                    metrics=[METRIC_BUILDERS[k](llm, embeddings) for k in metric_keys],
                    run_config=EVAL_RUN_CONFIG,
                    raise_exceptions=True,
                )
                scores = result.to_pandas().iloc[0].to_dict()
            break
        except Exception as exc:
            last_exc = exc
            if _is_transient_json_parse_error(exc) and attempt < MAX_ATTEMPTS:
                time.sleep(1)
                continue
            st.error(_friendly_llm_error(exc))
            with st.expander("Technical details"):
                st.code(str(exc))
            st.stop()

    stop_ms = int(time.time() * 1000)
    duration_s = time.perf_counter() - start_t

    metric_values = {}
    for k in metric_keys:
        fallback = scores.get("answer_relevancy") if k == "response_relevancy" else None
        value = scores.get(k, fallback)
        metric_values[METRIC_LABELS[k]] = float(value) if value is not None else float("nan")
    nan_metrics = [name for name, value in metric_values.items() if math.isnan(value)]

    try:
        allure_utils.write_case_result(
            app_config.allure_results_dir,
            test_id="TC_SINGLE",
            query=query,
            ground_truth=ground_truth,
            answer=answer,
            contexts=contexts,
            metrics={k: v for k, v in metric_values.items() if not math.isnan(v)},
            status="failed" if nan_metrics else "passed",
            status_message=f"Could not compute: {', '.join(nan_metrics)}" if nan_metrics else None,
            story="Single Case Evaluation",
            start_ms=start_ms,
            stop_ms=stop_ms,
        )
    except Exception as exc:
        # A report-writing failure is a side effect, not a scoring failure -- don't
        # let it hide the metric scores computed just above; surface it as a
        # non-fatal warning and keep going so the Scores section below still renders.
        st.warning(f"Evaluation succeeded but the Allure result could not be written: {exc}")

    if nan_metrics:
        st.error(
            f"Evaluation failed: could not compute {', '.join(nan_metrics)} after {MAX_ATTEMPTS} "
            f"attempt(s) ({MAX_RETRIES} retries). The LLM call for these metrics failed silently. "
            "Check the API Key / Endpoint / Deployment Name in the LLM Configuration sidebar, then try again."
        )
        st.stop()

    st.subheader("Scores")
    st.caption(f"Started: {_fmt_ms(start_ms)}  |  Completed: {_fmt_ms(stop_ms)}")
    cols = st.columns(len(metric_values) + 1)
    for col, name, value in zip(cols, metric_values.keys(), metric_values.values()):
        if math.isnan(value):
            col.metric(name, "—")
        else:
            col.metric(name, f"{value:.4f}", _score_label(value))
    cols[-1].metric("Time Taken", f"{duration_s:.2f}s")

if run_batch:
    existing_job = st.session_state.get("batch_job")
    if existing_job is not None and existing_job.snapshot()["status"] == "running":
        # No st.stop() here: that would halt the script before it reaches the
        # "active_job" rendering block below, which is where the progress bar
        # and "Stop Evaluation" button actually live — the user would see this
        # warning but never the button that lets them act on it. Falling
        # through (skipping straight to that block instead of starting a new
        # job) is what actually surfaces the Stop button "below" as promised.
        st.warning("A batch evaluation is already running below. Stop it or wait for it to finish first.")
    else:
        if batch_path.strip():
            try:
                with open(batch_path.strip(), "r", encoding="utf-8") as f:
                    items = json.load(f)
            except Exception as exc:
                st.error(f"Could not read JSON file at '{batch_path}': {exc}")
                st.stop()
        elif uploaded_file is not None:
            try:
                items = json.load(uploaded_file)
            except Exception as exc:
                st.error(f"Could not parse uploaded JSON file: {exc}")
                st.stop()
        else:
            st.error("Provide a JSON file path or upload a JSON file.")
            st.stop()

        if not isinstance(items, list) or not items:
            st.error("The JSON file must contain a non-empty list of test cases.")
            st.stop()

        validation_errors = _validate_batch_items(items, use_api)
        if validation_errors:
            st.error("The JSON file is missing required fields for some test cases:")
            for err in validation_errors:
                st.write(f"- {err}")
            st.stop()

        if use_api and not rag_api_cfg.endpoint.strip():
            st.error("RAG API Endpoint is required.")
            st.stop()

        if not selected_metric_keys:
            st.error("Select at least one metric to evaluate in the sidebar.")
            st.stop()

        test_ids = [item.get("test_id", f"TC_{i + 1:03d}") for i, item in enumerate(items)]
        metric_cols = selected_metric_keys
        metric_labels = {k: METRIC_LABELS[k] for k in selected_metric_keys}

        with st.spinner("Checking LLM credentials..."):
            try:
                check_llm_credentials(build_langchain_llm(llm_cfg))
            except Exception as exc:
                st.error(_friendly_llm_error(exc))
                with st.expander("Technical details"):
                    st.code(str(exc))
                st.stop()

        concurrency = max(1, min(app_config.batch_size, len(test_ids)))
        new_job = _BatchJob(items, test_ids, metric_cols, metric_labels)
        st.session_state["batch_job"] = new_job
        threading.Thread(
            target=_run_batch_job,
            args=(new_job, items, use_api, rag_api_cfg, llm_cfg, concurrency),
            daemon=True,
        ).start()
        st.rerun()

# Rendered on every script run, not just the one that clicked "Run Batch Evaluation" —
# this is what makes the batch survive being interrupted by an unrelated widget
# interaction: the actual work runs in the detached background thread started
# above, and this just displays its current state, however this particular
# rerun was triggered.
active_job = st.session_state.get("batch_job")
if active_job is not None:
    snap = active_job.snapshot()
    metric_cols = active_job.metric_cols
    metric_labels = active_job.metric_labels
    total = len(active_job.test_ids)

    if snap["status"] == "running":
        st.subheader("Batch Evaluation — In Progress")
        st.progress(
            snap["completed"] / total if total else 0.0,
            text=f"Evaluated {snap['completed']}/{total} test case(s)...",
        )
        if st.button("Stop Evaluation"):
            active_job.cancel_event.set()
            st.warning("Stopping — test cases already in flight will finish; no new ones will start.")
        # Rebuilt fresh from the job's current snapshot on every poll (not
        # incrementally updated in place), so this is safe regardless of why
        # this particular rerun happened.
        for i, test_id in enumerate(active_job.test_ids):
            stage = snap["row_status"][i]
            if stage.startswith("Done"):
                icon = "✅"
            elif stage.startswith("Missing") or "failed" in stage.lower():
                icon = "❌"
            else:
                icon = "🔄"
            # st.expander (unlike st.status) accepts a `key`, so Streamlit persists
            # whether the user manually expanded/collapsed it across reruns — this
            # page reruns itself every ~1.5s while a batch is running (see the
            # st.rerun() below) purely to poll for progress, and st.status has no
            # way to remember a manual expand across that: every poll rebuilt it
            # from scratch with expanded=False, snapping it shut again.
            with st.expander(f"{icon} {test_id} — {stage}", key=f"batch_row_expander_{i}"):
                item = active_job.items[i]
                st.caption(item.get("query", ""))
                row_record = snap["row_records"][i]
                st.caption(
                    f"Started: {_fmt_ms(snap['row_start_ms'][i])}"
                    + (f"  |  Completed: {_fmt_ms(row_record['stop_ms'])}" if row_record is not None else "")
                )
                answer = snap["row_answer"][i]
                contexts = snap["row_contexts"][i]
                if answer is not None:
                    st.write("**Generated Answer:**", answer)
                    st.write("**Contexts:**")
                    for c in contexts or []:
                        st.text(c if len(c) <= 500 else c[:500] + "…")
                scores = snap["row_scores"][i]
                if scores is not None:
                    for k in active_job.metric_cols:
                        v = scores.get(k, float("nan"))
                        label = active_job.metric_labels[k]
                        st.write(f"**{label}:** {'—' if v != v else f'{v:.4f} ({_score_label(v)})'}")
        time.sleep(1.5)
        st.rerun()
    else:
        if snap["status"] == "cancelled":
            st.warning(f"Evaluation stopped manually after {snap['completed']}/{total} test case(s).")
        elif snap["status"] == "error":
            st.error(f"Batch evaluation crashed: {snap['error_message']}")

        completed_records = [r for r in snap["row_records"] if r is not None]
        fetch_errors = [r for r in completed_records if r.get("fetch_error")]
        if fetch_errors:
            st.warning("Some test cases could not be fetched from the RAG API and were skipped from scoring:")
            for r in fetch_errors:
                st.write(f"- {r['test_id']}: {r['fetch_error']}")

        scoring_errors = [r for r in completed_records if r.get("scoring_error")]
        if scoring_errors:
            st.warning("Some test cases were fetched but could not be scored:")
            for r in scoring_errors:
                st.write(f"- {r['test_id']}: {r['scoring_error']}")

        ok_records = [
            r for r in completed_records if not r.get("fetch_error") and not r.get("scoring_error")
        ]
        if ok_records:
            df = pd.DataFrame(ok_records)

            if not active_job.finalized:
                failed_case_details = []
                allure_write_errors = []
                for _, row in df.iterrows():
                    row_errors = row.get("errors") or {}
                    missing_metrics = [
                        metric_labels[c] for c in metric_cols if isinstance(row[c], float) and math.isnan(row[c])
                    ]
                    row_metrics = {
                        metric_labels[c]: float(row[c])
                        for c in metric_cols
                        if not (isinstance(row[c], float) and math.isnan(row[c]))
                    }
                    has_nan = bool(missing_metrics)
                    if has_nan:
                        reasons = "; ".join(f"{metric_labels[k]}: {msg}" for k, msg in row_errors.items())
                        detail = f"{row['test_id']}: {', '.join(missing_metrics)}"
                        if reasons:
                            detail += f" ({reasons})"
                        failed_case_details.append(detail)
                    try:
                        allure_utils.write_case_result(
                            app_config.allure_results_dir,
                            test_id=row["test_id"],
                            query=row["user_input"],
                            ground_truth=row.get("reference", ""),
                            answer=row.get("response", ""),
                            contexts=list(row.get("retrieved_contexts", [])),
                            metrics=row_metrics,
                            status="failed" if has_nan else "passed",
                            status_message="One or more metrics could not be computed." if has_nan else None,
                            story="Batch Evaluation",
                            start_ms=int(row["start_ms"]),
                            stop_ms=int(row["stop_ms"]),
                        )
                    except Exception as exc:
                        # A report-writing failure is a side effect, not a scoring
                        # failure -- one bad write shouldn't stop the rest of the
                        # cases from being written, or hide the summary/table below.
                        allure_write_errors.append(f"{row['test_id']}: {exc}")
                active_job.failed_case_details = failed_case_details
                active_job.allure_write_errors = allure_write_errors
                active_job.finalized = True

            if active_job.allure_write_errors:
                st.warning("Scores were computed but some Allure results could not be written:")
                for detail in active_job.allure_write_errors:
                    st.write(f"- {detail}")

            if active_job.failed_case_details:
                st.error(
                    f"Evaluation failed: {len(active_job.failed_case_details)} of {len(df)} test case(s) have "
                    f"metrics that could not be computed after {MAX_RETRIES + 1} attempt(s) ({MAX_RETRIES} retries):"
                )
                for detail in active_job.failed_case_details:
                    st.write(f"- {detail}")
            else:
                st.subheader("Summary — Averages Across All Test Cases")
                summary_cols = st.columns(len(metric_cols))
                for col_widget, metric_key in zip(summary_cols, metric_cols):
                    avg_value = df[metric_key].mean(skipna=True)
                    if math.isnan(avg_value):
                        col_widget.metric(metric_labels[metric_key], "—")
                    else:
                        col_widget.metric(metric_labels[metric_key], f"{avg_value:.4f}", _score_label(avg_value))

                st.caption(
                    f"Total evaluation time: {df['duration_s'].sum():.2f}s "
                    f"(average {df['duration_s'].mean():.2f}s per test case)"
                )

                st.subheader(f"Per-Case Results ({len(df)} test case{'s' if len(df) != 1 else ''})")
                display_df = df[["test_id"] + metric_cols + ["duration_s"]].copy()
                display_df.columns = ["Test ID"] + [metric_labels[c] for c in metric_cols] + ["Duration (s)"]
                for col_name in [metric_labels[c] for c in metric_cols]:
                    display_df[col_name] = display_df[col_name].apply(
                        lambda v: "—" if (isinstance(v, float) and math.isnan(v)) else round(v, 4)
                    )
                display_df["Duration (s)"] = display_df["Duration (s)"].round(2)
                st.dataframe(display_df, use_container_width=True, hide_index=True)

                # A checkbox, not st.expander, controls the outer show/hide here: each
                # test case below is its own st.expander (so it can be opened/closed
                # independently, and — via `key` — remembers that across reruns), and
                # Streamlit doesn't allow nesting an expander inside another expander.
                show_full_details = st.checkbox("Show full details per test case (query, ground truth, answer, contexts)")
                if show_full_details:
                    for _, row in df.iterrows():
                        query_preview = row["user_input"] if len(row["user_input"]) <= 80 else row["user_input"][:80] + "…"
                        with st.expander(f"{row['test_id']} — {query_preview}", key=f"batch_final_expander_{row['test_id']}"):
                            st.write("Query:", row["user_input"])
                            st.write("Ground Truth:", row.get("reference", ""))
                            st.write("Generated Answer:", row.get("response", ""))
                            st.write("Contexts:", row.get("retrieved_contexts", []))
                            st.caption(
                                f"Started: {_fmt_ms(row.get('start_ms'))}  |  "
                                f"Completed: {_fmt_ms(row.get('stop_ms'))}"
                            )
        elif snap["status"] != "error":
            st.error("No test cases could be evaluated.")

        if st.button("Clear batch results"):
            del st.session_state["batch_job"]
            st.rerun()
