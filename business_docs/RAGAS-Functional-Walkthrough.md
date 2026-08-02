# RAG Evaluation Platform — Functional Walkthrough

Every control that exists in the running app today, organized the way the screen
is organized: **left sidebar** (setup) → **main panel** (inputs) → **results**
(scores). Field names below match the UI exactly.

---

## Navigation

- **Dashboard** (home screen) — a card per tool. Today there are two:
  - **RAG Evaluation using RAGAS** — click **Open →** to launch it.
  - **Query Generator** — greyed out, shows "🚧 Coming soon" (see *Roadmap* below).
- **← Back to Dashboard** — top of the evaluation page, returns to the dashboard.

---

## Left Sidebar

### LLM Configuration
- **Provider** — currently fixed to **Azure OpenAI** in the running app. (OpenAI
  support is fully built in the code and can be re-enabled as a second option
  with a one-line change — see *Note on providers* at the end.)
- **Azure API Key** (masked)
- **Azure Endpoint** (e.g. `https://your-resource.openai.azure.com/`)
- **Deployment Name** (chat model)
- **Deployment Name** (embedding model)
- API version is **auto-detected** from the endpoint shape — not a field you fill in.
- **Save Config** — writes these values into `config.yaml` so they're pre-filled
  next time the app opens. Without saving, typed values are still used for that
  run only.

### Allure Report
- **Generate & Open Report** — builds an HTML report from every evaluation run
  so far (single-case and batch, from this session and any pytest runs) and
  starts a local server for it.
- **Clear Previous Results** — wipes accumulated results so the next report
  starts fresh.
- **Open Allure Report ↗** — appears once a report has been generated; opens it
  in a new browser tab.

### Metrics
- Multi-select of the four RAGAS metrics: **Response Relevancy**,
  **Faithfulness**, **Context Precision**, **Context Recall**.
- Defaults to whatever's listed under `ragas.metrics` in `config.yaml`;
  changing it here overrides for this run only.
- Tooltip warns that Context Precision and Context Recall are the slowest —
  each issues one LLM call per retrieved context chunk.

---

## Main Panel

### Data source toggle
- **Fetch answer & contexts from RAG API** (checkbox) — switches the entire
  page between two modes: pull the answer live from your RAG endpoint, or
  enter it manually.

### RAG API Configuration (visible when the checkbox above is on)
- **RAG API Endpoint**, **Timeout (seconds)**
- **Set Custom Header** toggle → **Header Name** / **Header Value** (masked) —
  for APIs that need an `Authorization` header or similar.
- **Request Payload** builder — an editable table of key/value rows sent as the
  JSON body of every API call. Each row is either:
  - **From JSON field** — pulled from that field in the test case (varies per case), or
  - **Hardcoded value** — the same literal value on every call.
  - Rows can be added (**+ Add Parameter**) or removed (**✕**), with a live
    **JSON preview** showing exactly what will be sent, built from a sample test case.
- **Set Answer / Contexts Field Manually** toggle → **Answer Field**,
  **Contexts Field**, **Context Item Field** — for RAG APIs whose response
  shape isn't one of the auto-detected ones. A reference panel
  (**RAG API Response Field Definitions**) shows example flat and nested
  response shapes with dotted-path syntax (e.g. `message.generated_response`).
- **Save Config** — persists this whole section to `config.yaml`, independently
  of the LLM Configuration section.

### Evaluation Mode
Radio toggle between two modes:

- **Single Case**
  - **Query**, **Ground Truth** (free text)
  - If not using the API: **Generated Answer** + **Contexts** (one chunk per
    line, 3–4 recommended)
- **Batch (JSON file)**
  - **JSON File Path (on this machine)** — a path text field, *or*
  - **…or upload a JSON file** — a file picker
  - An **Expected JSON schema** panel shows the exact fields required, which
    adapts automatically depending on whether API mode is on (API mode only
    needs `query` + `ground_truth` per case; manual mode also needs
    `generated_answer` and `contexts`).

### Run button
- **Run Evaluation** (single case) / **Run Batch Evaluation** (batch) — primary
  action button.
- Before running, it validates: at least one metric selected, required fields
  present, and (for batch) every row of the uploaded JSON checked at once —
  every problem is reported together, not one at a time.
- A transient LLM JSON-formatting glitch is retried automatically once before
  a case is marked failed.
- Provider errors are translated into plain language instead of a raw stack
  trace: invalid API key (401), wrong model/deployment name (404), rate limit
  (429), and connection/timeout issues each get their own actionable message
  pointing back at the sidebar field to check.

---

## Results — Single Case

- **RAG API Response** (API mode only) — collapsible **Generated Answer** and
  **Contexts** panels showing exactly what came back from your endpoint.
- **Scores** — one tile per selected metric: the 0–1 score, a quality label
  (**EXCELLENT / GOOD / MODERATE / POOR**), color-coded green/orange/red. A
  low score gets a hover tooltip explaining *why*, generated from the judge
  LLM's own reasoning for that case — not a generic definition.
- **Time Taken** and **judge-LLM token usage** (input/output/total).
- **Suggested Focus** — attributes a weak score to a stage of the RAG pipeline
  (**Retrieval**, **Augmentation**, and/or **Generation**), with a plain-language
  summary and an expandable checklist of concrete things to check for that
  specific failure mode.

## Results — Batch

- **Live progress** — a progress bar plus a status row per test case
  (🔄 running / ✅ done / ❌ failed), updating in real time while the batch runs
  in the background.
- **Stop Evaluation** — cancels cleanly: cases already in flight finish, no new
  ones start.
- Once complete:
  - Warnings (not hard failures) for any case that couldn't be fetched from the
    API or couldn't be scored — everything else still shows.
  - **Summary — Averages Across All Test Cases**: one average per metric, total
    and per-case evaluation time, total tokens used.
  - **Suggested focus areas across this batch** — aggregated across every case
    (e.g. *"Retrieval (4/10 test cases)"*), with a combined checklist of fixes
    by area.
  - **Per-Case Results** table — one row per test case: every metric score,
    duration, tokens used, and its suggested focus.
  - **Show full details per test case** (checkbox) — expands every case into
    its full query / ground truth / generated answer / contexts, each metric's
    tooltip, and its individual diagnosis.
  - **Clear batch results** — resets the view for a fresh run.

---

## Behind the Scenes

- Every evaluation — single case or batch, from the UI or from an automated
  pytest run — writes into the same results folder, so **Generate & Open
  Report** always reflects everything run so far, from any source.
- The LLM Configuration and RAG API Configuration sections save to
  `config.yaml` independently, so changing one never requires re-entering the
  other.

## On the Roadmap

- **Query Generator** — visible on the dashboard today as a disabled card.
  Will auto-generate test questions and ground-truth answers from your own
  content, removing the need to hand-write test cases before evaluating.

## Note on providers

The running app's sidebar currently offers only **Azure OpenAI**. Plain
**OpenAI** and **Azure AI Foundry** are both already implemented at the
configuration/backend level (multi-provider credential handling, client
building, embeddings) — they're just not exposed as sidebar options in the
current build. Worth knowing if a client asks "does it support OpenAI directly"
— the answer is yes, it's one line away from being turned back on.
