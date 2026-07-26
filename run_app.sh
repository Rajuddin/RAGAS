#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run_app.sh
# One-shot launcher for the Streamlit UI. Creates the virtual environment and
# installs all dependencies automatically on first run, then starts the app.
#
# Usage:
#   ./run_app.sh
# ─────────────────────────────────────────────────────────────────────────────
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo ""
echo "══════════════════════════════════════════════"
echo "  RAGAS Evaluation — Streamlit UI"
echo "══════════════════════════════════════════════"
echo ""

# ── 1. Python check ──
if ! command -v python3 &>/dev/null; then
  echo "ERROR: python3 not found. Please install Python 3.9+."
  exit 1
fi
echo "Using Python: $(python3 --version)"

# ── 2. Virtual environment ──
if [ ! -d ".venv" ]; then
  echo ""
  echo "▶ Creating virtual environment..."
  python3 -m venv .venv
fi
source .venv/bin/activate

# ── 3. Install / upgrade dependencies ──
echo ""
echo "▶ Installing dependencies (this may take a minute on first run)..."
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

# ── 4. Config check ──
if [ ! -f "config.yaml" ]; then
  echo ""
  echo "▶ No config.yaml found — creating one from config.example.yaml."
  echo "  Fill in your LLM/API keys (use the sidebar in the app, or edit the file directly)."
  cp config.example.yaml config.yaml
fi

# ── 5. Launch ──
# The Allure CLI (used by the in-app "Open Allure Report" button) is downloaded
# automatically into .tools/ the first time the button is clicked — no separate
# install step needed here, only a Java runtime on the machine.
echo ""
echo "▶ Starting Streamlit app..."
echo ""
streamlit run app.py
