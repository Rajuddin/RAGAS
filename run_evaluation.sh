#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run_evaluation.sh
# Install dependencies and execute RAGAS evaluation with Allure reporting.
#
# Usage:
#   ./run_evaluation.sh                              # uses sample data
#   ./run_evaluation.sh --test-data-file my_data.json
#   ./run_evaluation.sh --query "..." --ground-truth "..."
# ─────────────────────────────────────────────────────────────────────────────
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo ""
echo "══════════════════════════════════════════════"
echo "  RAGAS Evaluation Framework"
echo "══════════════════════════════════════════════"
echo ""

# ── 1. Python check ──
if ! command -v python3 &>/dev/null; then
  echo "ERROR: python3 not found. Please install Python 3.9+."
  exit 1
fi

PYTHON=$(command -v python3)
echo "Using Python: $($PYTHON --version)"

# ── 2. Virtual environment ──
if [ ! -d ".venv" ]; then
  echo ""
  echo "▶ Creating virtual environment..."
  $PYTHON -m venv .venv
fi
source .venv/bin/activate

# ── 3. Install / upgrade dependencies ──
echo ""
echo "▶ Installing dependencies (this may take a minute on first run)..."
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

# ── 3b. Config check ──
if [ ! -f "config.yaml" ]; then
  echo ""
  echo "▶ No config.yaml found — creating one from config.yaml.temp."
  echo "  Edit config.yaml with your real API keys before running again."
  cp config.yaml.temp config.yaml
fi

# ── 4. Allure CLI check ──
if ! command -v allure &>/dev/null; then
  echo ""
  echo "⚠  Allure CLI not found. HTML report generation will be skipped."
  echo "   Install Allure: https://docs.qameta.io/allure/#_installing_a_commandline"
  ALLURE_AVAILABLE=false
else
  ALLURE_AVAILABLE=true
fi

# ── 5. Run pytest ──
ALLURE_RESULTS_DIR="./allure-results"
mkdir -p "$ALLURE_RESULTS_DIR"

echo ""
echo "▶ Running RAGAS evaluation..."
echo ""

pytest tests/ \
  --alluredir="$ALLURE_RESULTS_DIR" \
  -v \
  --tb=short \
  "$@"

PYTEST_EXIT_CODE=$?

# ── 6. Generate Allure report ──
if [ "$ALLURE_AVAILABLE" = "true" ]; then
  echo ""
  echo "▶ Generating Allure report..."
  allure generate "$ALLURE_RESULTS_DIR" --clean -o allure-report
  echo ""
  echo "✔ Allure report saved to: $SCRIPT_DIR/allure-report/index.html"
  echo ""
  echo "▶ Opening Allure report in browser..."
  allure open allure-report
fi

echo ""
echo "══════════════════════════════════════════════"
if [ $PYTEST_EXIT_CODE -eq 0 ]; then
  echo "  ✔ Evaluation COMPLETE — all tests passed"
else
  echo "  ✖ Evaluation COMPLETE — some tests failed (exit code: $PYTEST_EXIT_CODE)"
fi
echo "══════════════════════════════════════════════"
echo ""

exit $PYTEST_EXIT_CODE
