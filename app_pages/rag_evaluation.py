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
import time
from pathlib import Path

import streamlit as st
from datasets import Dataset

from ragas import evaluate
from ragas.metrics import (
    ResponseRelevancy,
    Faithfulness,
    ContextPrecision,
    ContextRecall,
)

from config_loader import (
    load_config,
    build_langchain_llm,
    build_langchain_embeddings,
    save_llm_config,
    save_rag_api_config,
    LLMConfig,
    RAGApiConfig,
)
from rag_client import query_rag_system, build_payload
import allure_utils

ALLURE_REPORT_DIR = Path(__file__).parent.parent / "allure-report"


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


@st.cache_resource
def get_app_config():
    return load_config()


if st.button("← Back to Dashboard"):
    st.switch_page("app_pages/home.py")

st.title("RAG Evaluation using RAGAS")

app_config = get_app_config()

# ─────────────────────────────────────────────
# LLM provider configuration (sidebar)
# ─────────────────────────────────────────────

st.sidebar.header("LLM Configuration")

provider = st.sidebar.radio(
    "Provider",
    options=["openai", "azure"],
    index=["openai", "azure"].index(app_config.llm.provider)
    if app_config.llm.provider in ("openai", "azure")
    else 0,
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
    allure_utils.clear_results(app_config.allure_results_dir, ALLURE_REPORT_DIR)
    st.session_state.pop("allure_report_url", None)
    st.sidebar.success("Cleared. Run an evaluation to start a fresh report.")

if st.session_state.get("allure_report_url"):
    st.sidebar.link_button("Open Allure Report ↗", st.session_state["allure_report_url"])

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

    MAX_ATTEMPTS = 3
    scores = None
    last_exc = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            with st.spinner(f"Running RAGAS evaluation... (attempt {attempt}/{MAX_ATTEMPTS})"):
                llm = build_langchain_llm(llm_cfg)
                embeddings = build_langchain_embeddings(llm_cfg)

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
                    metrics=[
                        ResponseRelevancy(llm=llm, embeddings=embeddings),
                        Faithfulness(llm=llm),
                        ContextPrecision(llm=llm),
                        ContextRecall(llm=llm),
                    ],
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

    response_relevancy = float(scores.get("response_relevancy", scores.get("answer_relevancy", 0)))
    faithfulness = float(scores.get("faithfulness", 0))
    context_precision = float(scores.get("context_precision", 0))
    context_recall = float(scores.get("context_recall", 0))

    metric_values = {
        "Response Relevancy": response_relevancy,
        "Faithfulness": faithfulness,
        "Context Precision": context_precision,
        "Context Recall": context_recall,
    }
    nan_metrics = [name for name, value in metric_values.items() if math.isnan(value)]
    if nan_metrics:
        st.error(
            f"Could not compute: {', '.join(nan_metrics)}. The LLM call for these metrics failed silently. "
            "Check the API Key / Endpoint / Deployment Name in the LLM Configuration sidebar, then try again."
        )

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
    )

    st.subheader("Scores")
    cols = st.columns(4)
    for col, name, value in zip(cols, metric_values.keys(), metric_values.values()):
        if math.isnan(value):
            col.metric(name, "—")
        else:
            col.metric(name, f"{value:.4f}", _score_label(value))

if run_batch:
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

    test_ids, questions, answers, contexts_list, ground_truths = [], [], [], [], []
    fetch_errors = []

    if use_api:
        progress = st.progress(0.0, text="Calling RAG API...")
        for i, item in enumerate(items):
            test_id = item.get("test_id", f"TC_{i + 1:03d}")
            progress.progress(i / len(items), text=f"Calling RAG API for {test_id}...")
            try:
                answer_i, contexts_i = query_rag_system(item, rag_api_cfg)
            except Exception as exc:
                fetch_errors.append(f"{test_id}: RAG API call failed: {exc}")
                continue
            test_ids.append(test_id)
            questions.append(item["query"])
            answers.append(answer_i)
            contexts_list.append(contexts_i)
            ground_truths.append(item["ground_truth"])
        progress.empty()
    else:
        for i, item in enumerate(items):
            test_ids.append(item.get("test_id", f"TC_{i + 1:03d}"))
            questions.append(item["query"])
            answers.append(item["generated_answer"])
            contexts_list.append(item["contexts"])
            ground_truths.append(item["ground_truth"])

    if fetch_errors:
        st.warning("Some test cases could not be fetched from the RAG API and were skipped:")
        for err in fetch_errors:
            st.write(f"- {err}")

    if not test_ids:
        st.error("No test cases could be evaluated.")
        st.stop()

    metric_cols = ["response_relevancy", "faithfulness", "context_precision", "context_recall"]
    metric_labels = {
        "response_relevancy": "Response Relevancy",
        "faithfulness": "Faithfulness",
        "context_precision": "Context Precision",
        "context_recall": "Context Recall",
    }

    def _run_ragas(ds):
        return evaluate(
            dataset=ds,
            metrics=[
                ResponseRelevancy(llm=llm, embeddings=embeddings),
                Faithfulness(llm=llm),
                ContextPrecision(llm=llm),
                ContextRecall(llm=llm),
            ],
            # A single failed row shouldn't sink the whole batch — surface NaNs per row/metric instead
            # (retried below, since these are usually the same transient JSON-formatting glitches that
            # Single Case mode retries for).
            raise_exceptions=False,
        )

    def _normalize(result_df):
        if "answer_relevancy" in result_df.columns and "response_relevancy" not in result_df.columns:
            result_df["response_relevancy"] = result_df["answer_relevancy"]
        for col in metric_cols:
            if col not in result_df.columns:
                result_df[col] = float("nan")
        return result_df

    with st.spinner(f"Running RAGAS evaluation on {len(test_ids)} test case(s)..."):
        llm = build_langchain_llm(llm_cfg)
        embeddings = build_langchain_embeddings(llm_cfg)

        dataset = Dataset.from_dict(
            {
                "question": questions,
                "answer": answers,
                "contexts": contexts_list,
                "ground_truth": ground_truths,
            }
        )

        try:
            df = _normalize(_run_ragas(dataset).to_pandas())
        except Exception as exc:
            st.error(_friendly_llm_error(exc))
            with st.expander("Technical details"):
                st.code(str(exc))
            st.stop()

        MAX_BATCH_ATTEMPTS = 3
        for attempt in range(2, MAX_BATCH_ATTEMPTS + 1):
            missing_positions = [i for i in range(len(df)) if df.iloc[i][metric_cols].isna().any()]
            if not missing_positions:
                break
            st.caption(
                f"Retrying {len(missing_positions)} test case(s) with missing metrics "
                f"(attempt {attempt}/{MAX_BATCH_ATTEMPTS})..."
            )
            try:
                retry_df = _normalize(_run_ragas(dataset.select(missing_positions)).to_pandas())
            except Exception:
                continue
            for retry_pos, orig_pos in enumerate(missing_positions):
                for col in metric_cols:
                    new_val = float(retry_df.iloc[retry_pos][col])
                    if not math.isnan(new_val) and math.isnan(float(df.iloc[orig_pos][col])):
                        df.iat[orig_pos, df.columns.get_loc(col)] = new_val

    df.insert(0, "test_id", test_ids)

    for _, row in df.iterrows():
        row_metrics = {
            metric_labels[c]: float(row[c])
            for c in metric_cols
            if not (isinstance(row[c], float) and math.isnan(row[c]))
        }
        has_nan = len(row_metrics) < len(metric_cols)
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
        )

    st.subheader("Summary — Averages Across All Test Cases")
    summary_cols = st.columns(4)
    for col_widget, metric_key in zip(summary_cols, metric_cols):
        avg_value = df[metric_key].mean(skipna=True)
        if math.isnan(avg_value):
            col_widget.metric(metric_labels[metric_key], "—")
        else:
            col_widget.metric(metric_labels[metric_key], f"{avg_value:.4f}", _score_label(avg_value))

    n_with_failures = int(df[metric_cols].isna().any(axis=1).sum())
    if n_with_failures:
        st.warning(
            f"{n_with_failures} of {len(df)} test case(s) have at least one metric that could not be "
            "computed (shown as — below). This can happen on transient LLM formatting glitches — try "
            "re-running the batch."
        )

    st.subheader(f"Per-Case Results ({len(df)} test case{'s' if len(df) != 1 else ''})")
    display_df = df[["test_id"] + metric_cols].copy()
    display_df.columns = ["Test ID"] + [metric_labels[c] for c in metric_cols]
    for col_name in display_df.columns[1:]:
        display_df[col_name] = display_df[col_name].apply(
            lambda v: "—" if (isinstance(v, float) and math.isnan(v)) else round(v, 4)
        )
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    with st.expander("Full details per test case (query, ground truth, answer, contexts)"):
        for _, row in df.iterrows():
            st.markdown(f"**{row['test_id']}** — {row['user_input']}")
            st.write("Ground Truth:", row.get("reference", ""))
            st.write("Generated Answer:", row.get("response", ""))
            st.write("Contexts:", row.get("retrieved_contexts", []))
            st.divider()
