# Development Notes

Pointers on what was actually built for each activity, with references to the relevant code.

---

## 1. Framework design — pick RAGAS metrics (relevancy, faithfulness, precision, recall)

- Standardized on four RAGAS metrics per test case: `ResponseRelevancy`, `Faithfulness`, `ContextPrecision`, `ContextRecall` (`tests/test_ragas_evaluation.py:14-20`).
- Defined a shared scoring scale (`EXCELLENT ≥0.85`, `GOOD ≥0.70`, `MODERATE ≥0.50`, `POOR <0.50`) so every report reads consistently (`tests/test_ragas_evaluation.py:252-261`).
- Chose to evaluate `question/answer/contexts/ground_truth` as a single-row `Dataset` per case rather than batching everything, so failures are isolated per test case.

## 2. Config layer & multi-LLM support (OpenAI / Azure OpenAI / Azure AI Foundry)

- Built a `config.yaml`-driven `AppConfig`/`LLMConfig` dataclass model supporting three providers, each with its own required fields (`config_loader.py:44-96`).
- Added environment-variable overrides (`OPENAI_API_KEY`, `AZURE_OPENAI_API_KEY`, etc.) so secrets never have to live only in the YAML file (`config_loader.py:58,68`).
- Handled Azure AI Foundry's two distinct API shapes — classic Azure OpenAI deployments vs. the OpenAI-v1-compatible endpoint — with separate client builders (`_build_foundry_v1_chat`, `AzureAIChatCompletionsModel`) (`config_loader.py:154-224`).
- Wrapped every LLM/embeddings client in RAGAS's `LangchainLLMWrapper`/`LangchainEmbeddingsWrapper` instead of passing raw LangChain objects — needed for RAGAS's markdown-fence-safe JSON parsing and self-correction retry (`config_loader.py:187-195`).
- Added `save_llm_config`/`save_rag_api_config` so the Streamlit sidebar can persist entered credentials back to `config.yaml` without clobbering other sections.

## 3. RAG API client integration (rag_client.py)

- Wrote a generic `query_rag_system()` that posts `{question, chat_history}` and auto-detects the response shape instead of hardcoding one API contract (`rag_client.py:42-67`).
- Auto-detection tries a ranked list of common field names for both the answer (`answer`, `result`, `output_text`, `response`...) and contexts (`contexts`, `source_documents`, `retrieved_docs`...) (`rag_client.py:14-23`).
- Handled context items that are objects rather than strings (e.g. LangChain's `{"page_content": "..."}`) via `_extract_context_text` (`rag_client.py:30-39`).
- Exposed manual overrides (`answer_field`, `contexts_field`, `context_item_field`) for APIs that don't match any known shape, surfaced directly in the Streamlit UI.

## 4. Pytest suite + fixtures for the four RAGAS metrics

- Built a fixture chain in `conftest.py`: config → LLM/embeddings → test dataset → live RAG responses, so tests just consume `rag_responses` (`conftest.py:24-146`).
- Supported three ways to supply test data with a clear priority order: `--test-data-file` JSON, inline `--query`/`--ground-truth`, or a bundled sample file (`conftest.py:83-115`).
- Wrote one evaluation test (`test_evaluate_all_metrics`) that loops every test case, runs all 4 metrics, and fails the run with a summary of exactly which test IDs failed (`tests/test_ragas_evaluation.py:132-249`).
- Made per-row failures non-fatal to the rest of the batch — one bad LLM call is caught, logged, and evaluation continues for the remaining cases.

## 5. Allure reporting integration (auto-fetch CLI, result writer)

- `allure_utils.py` auto-downloads the Allure commandline tool into a local `.tools/` folder on first use if it isn't already on `PATH`, so no manual Allure install is required (just a JRE) (`allure_utils.py:54-91`).
- Implemented `write_case_result()` to write raw Allure result JSON (+ JSON attachments for inputs/scores) directly from the Streamlit app, without going through pytest at all (`allure_utils.py:94-176`).
- Solved the "must serve over http://, not file://" issue by spinning up a lightweight local `ThreadingHTTPServer` per report directory, reusing it across Streamlit reruns (`allure_utils.py:209-227`).
- Added `clear_results()` to wipe accumulated results/report between runs so old test cases don't leak into a fresh report.
- Report merges results from both sources (pytest runs and Streamlit runs) since both write into the same `allure-results/` directory.

## 6. Streamlit UI — single-case & batch evaluation

- Built two evaluation modes on one page: Single Case (manual query/ground truth entry) and Batch (JSON file upload or path) (`app_pages/rag_evaluation.py:336-383`).
- Sidebar lets you switch LLM provider (OpenAI/Azure) and RAG API config at runtime, pre-filled from `config.yaml`, with a "Save Config" button per section.
- Added a retry loop (up to 3 attempts) specifically for transient LLM JSON-parsing glitches during evaluation, rather than failing the whole run on the first hiccup (`app_pages/rag_evaluation.py:412-450`).
- Mapped raw provider errors (401/404/429/timeout) to actionable, human-readable messages instead of raw stack traces (`app_pages/rag_evaluation.py:58-89`).
- Batch mode validates every row up front (missing `query`/`ground_truth`/`generated_answer`/`contexts`) and reports all problems at once instead of failing on the first bad row (`app_pages/rag_evaluation.py:92-106`).
- NaN-safe scoring throughout — a metric that fails to compute shows as "—" instead of crashing the whole batch summary.

## 7. Test data expansion — with/without API, edge cases

- Authored parallel sample datasets for the two operating modes: `sample_test_data - With-API.json` (query + ground truth only, answer/contexts fetched live) and `sample_test_data_Without_API.json` (fully manual, includes `generated_answer` + `contexts`).
- Added a dedicated 2-case file (`sample_test_data_Without_API-2-TestCases.json`) covering distinct domains (geography vs. immunology) to check the framework isn't overfitting to one topic's phrasing/context style.
- Used these variants to exercise both code paths in `rag_evaluation.py` (API-fetch vs. manual entry) and confirm Allure results write correctly from both.
