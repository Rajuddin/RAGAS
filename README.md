# RAGAS Evaluation Framework

Automated QA evaluation for RAG-based systems using the [RAGAS](https://docs.ragas.io) framework with full Allure reporting.

---

## Metrics Evaluated

| Metric | Description |
|--------|-------------|
| **Response Relevancy** | How well the generated answer addresses the query |
| **Faithfulness** | Whether the answer is grounded in the retrieved contexts |
| **Context Precision** | Signal-to-noise ratio of retrieved contexts |
| **Context Recall** | How much of the ground truth is covered by the contexts |

---

## Project Structure

```
ragas_agent/
├── app.py                       ← Streamlit UI (single-case & batch evaluation)
├── allure_utils.py              ← Writes Allure results from the UI, builds/serves the report
├── config.example.yaml          ← Sanitized config template (copy to config.yaml, no real keys)
├── config_loader.py             ← Config parsing + LangChain LLM builder
├── rag_client.py                ← Calls your RAG API, extracts answer + contexts
├── conftest.py                  ← Pytest fixtures (config, test data, RAG responses)
├── requirements.txt
├── pytest.ini
├── run_evaluation.sh            ← One-shot pytest+Allure runner (Linux/macOS)
├── run_evaluation.bat           ← One-shot pytest+Allure runner (Windows)
├── run_app.sh                   ← One-shot Streamlit UI launcher (Linux/macOS)
├── run_app.bat                  ← One-shot Streamlit UI launcher (Windows)
├── package_for_share.sh         ← Zips the minimal files needed to hand this project to a colleague
├── test_data/
│   └── sample_test_data.json   ← Sample test data
└── tests/
    └── test_ragas_evaluation.py ← All RAGAS metric tests with Allure annotations
```

`config.yaml` (with your real API keys) is **not** part of this list and is excluded from
`package_for_share.sh` — see [Sharing this project](#sharing-this-project) below.

---

## Quick Start

### 1. Configure `config.yaml`

Choose your LLM provider and fill in the required fields:

**For OpenAI:**
```yaml
llm_provider: "openai"
openai:
  api_key: "sk-..."
  model: "gpt-4o"
rag_api:
  endpoint: "http://your-rag-api/query"
```

**For Azure GPT:**
```yaml
llm_provider: "azure"
azure:
  api_key: "your-azure-key"
  azure_endpoint: "https://your-resource.openai.azure.com/"
  deployment_name: "gpt-5-4"
  api_version: "2024-02-01"
rag_api:
  endpoint: "http://your-rag-api/query"
```

### 2. Prepare test data

Either a JSON file:
```json
[
  {
    "test_id": "TC001",
    "query": "What is X?",
    "ground_truth": "X is ..."
  }
]
```

Or pass inline via CLI flags.

### 3. Run evaluation

**Linux/macOS:**
```bash
chmod +x run_evaluation.sh
./run_evaluation.sh
# With a JSON file:
./run_evaluation.sh --test-data-file test_data/my_tests.json
# Single inline test:
./run_evaluation.sh --query "What is X?" --ground-truth "X is ..."
```

**Windows:**
```cmd
run_evaluation.bat
run_evaluation.bat --test-data-file test_data\my_tests.json
```

---

## RAG API Contract

Your RAG endpoint is called once per test case with a JSON request body you define
yourself in the UI's "RAG API Configuration → Request Payload" section (persisted to
`rag_api.payload_params` in `config.yaml`). Each payload key is either:
- **From JSON field** — its value is read from that key in the test case's JSON object
  (e.g. payload key `question` sourced from field `query`), so it varies per test case.
- **Hardcoded value** — the same literal value on every request (e.g. payload key
  `Type` hardcoded to `text`).

The default payload (used until you customize it) is:
```json
{"question": "user question here", "chat_history": []}
```

And your endpoint should return one of these shapes:

**Shape A (recommended):**
```json
{"answer": "...", "contexts": ["chunk1", "chunk2"]}
```

**Shape B (LangChain default):**
```json
{"result": "...", "source_documents": [{"page_content": "..."}]}
```

If your API has a different schema, edit the parsing block in `rag_client.py`.

---

## Allure Report

The report shows for each test case:
- Inputs: `query`, `ground_truth`, `generated_answer`, `contexts`
- Scores: all four metric values with quality labels (EXCELLENT / GOOD / MODERATE / POOR)
- Session summary: averages across all test cases

Results accumulate in `allure-results/` from **both** sources:
- `pytest` runs (via `run_evaluation.sh` / `run_evaluation.bat`)
- the Streamlit UI (`app.py`), which writes a result for every evaluation you run

**You don't need to install the Allure CLI yourself** — `allure_utils.py` checks for it on
`PATH`, and if it isn't found, downloads the official Allure commandline tool into a local
`.tools/` folder the first time it's needed (the only prerequisite is a Java runtime, which
most machines already have). This download happens once and is reused afterwards.

**In the Streamlit UI:** click **"Generate & Open Allure Report"** in the sidebar. This builds
the report from `allure-results/`, starts a small local web server for it, and reveals an
**"Open Allure Report ↗"** button that opens the report in a new browser tab (Allure's report
needs to be served over `http://`, not opened as a local file, since it fetches its data via
JS — the in-app button handles that for you).

**From the command line:** the run scripts auto-generate and open the report. To open manually:
```bash
allure open allure-report
```

---

## Streamlit UI (single-case & batch evaluation)

For ad-hoc evaluation without writing a test data file, run the Streamlit app:

```bash
cd ragas_agent                  # if not already in this folder
./run_app.sh                    # macOS/Linux — creates .venv, installs deps, launches the app
run_app.bat                     # Windows
```

`run_app.sh` / `run_app.bat` create the virtual environment, install every dependency from
`requirements.txt`, copy `config.example.yaml` → `config.yaml` if one doesn't exist yet, and
start the app — no manual setup steps required. (You can still do it manually with
`python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt &&
streamlit run app.py` if you prefer.)

- In the sidebar, pick **OpenAI**, **Azure OpenAI**, or **Azure AI Foundry** and fill in the required fields (API key, model / endpoint / deployment). These are pre-filled from `config.yaml` if it already has values for that provider. Azure OpenAI's API version is detected automatically from the endpoint and isn't a field you need to fill in.
- Click **Save Config** to write the sidebar values back into `config.yaml`, so you don't have to re-enter them next time you open the app. Without saving, the values you type are still used for that run only.
- Enter **Query** and **Ground Truth**.
- If "Fetch answer & contexts from RAG API" is checked, a **RAG API Configuration** panel lets you set the Endpoint, Timeout, and an optional Auth Header directly in the UI (pre-filled from `config.yaml`, with its own **Save Config** button). The answer and contexts are then pulled from that endpoint automatically.
- If unchecked (no API available), enter the **Generated Answer** and paste your **Contexts** (3-4 chunks, one per line) directly in the UI.
- Click **Run Evaluation** (or **Run Batch Evaluation**) to see all four RAGAS metric scores. Every run also writes its results to `allure-results/`.
- In the sidebar, click **"Generate & Open Allure Report"** to build the report from everything in `allure-results/` so far, then click the **"Open Allure Report ↗"** button that appears to view it in a new tab.

---

## Sharing this project

`config.yaml` holds real API keys — **never share it**. To hand the project to a colleague:

```bash
./package_for_share.sh                 # writes ragas_agent_share.zip
```

This zips only the files needed to run the project (source code, run scripts, sample test
data, and `config.example.yaml`) — no `.venv`, caches, generated reports, or real credentials.
Your colleague then:
1. Unzips it and runs `./run_app.sh` (or `run_app.bat`) — this auto-creates `config.yaml` from
   the template and installs every Python dependency automatically.
2. Fills in their own API key(s) via the sidebar (and clicks **Save Config**), or edits
   `config.yaml` directly.
3. Optionally installs a Java runtime if one isn't already present — needed only for the
   **Open Allure Report** button, which downloads the Allure CLI itself on first use.

---

## Environment Variable Overrides

| Variable | Purpose |
|----------|---------|
| `OPENAI_API_KEY` | Override OpenAI key from config |
| `AZURE_OPENAI_API_KEY` | Override Azure key from config |
| `AZURE_OPENAI_ENDPOINT` | Override Azure endpoint from config |
| `RAG_API_ENDPOINT` | Override RAG endpoint from config |
| `RAGAS_CONFIG_PATH` | Path to a custom config.yaml |
| `RAGAS_TEST_DATA_FILE` | Path to test data JSON |
| `RAGAS_QUERY` | Single query (inline mode) |
| `RAGAS_GROUND_TRUTH` | Single ground truth (inline mode) |
