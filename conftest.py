"""
conftest.py
Pytest fixtures that wire together:
  - Config loading
  - Test data (inline or from JSON file)
  - RAG API calls (query → answer + contexts)
  - RAGAS Dataset construction
"""

import json
import os
import pytest
from pathlib import Path
from typing import List, Dict, Any

from config_loader import load_config, build_langchain_llm, build_langchain_embeddings
from rag_client import query_rag_system


# ─────────────────────────────────────────────
# 1. Configuration fixture
# ─────────────────────────────────────────────

@pytest.fixture(scope="session")
def app_config():
    """Load and expose the full application config once per session."""
    cfg_path = os.environ.get("RAGAS_CONFIG_PATH", None)
    return load_config(cfg_path)


# ─────────────────────────────────────────────
# 2. LLM + Embeddings fixtures
# ─────────────────────────────────────────────

@pytest.fixture(scope="session")
def llm(app_config):
    """LangChain LLM instance (OpenAI or Azure)."""
    return build_langchain_llm(app_config.llm)


@pytest.fixture(scope="session")
def embeddings(app_config):
    """LangChain Embeddings instance (OpenAI or Azure)."""
    return build_langchain_embeddings(app_config.llm)


# ─────────────────────────────────────────────
# 3. Test-data fixtures
# ─────────────────────────────────────────────

def _load_test_data_from_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    required_keys = {"query", "ground_truth"}
    for i, item in enumerate(data):
        missing = required_keys - set(item.keys())
        if missing:
            raise ValueError(f"Test data item {i} is missing keys: {missing}")
    return data


def pytest_addoption(parser):
    parser.addoption(
        "--test-data-file",
        action="store",
        default=None,
        help="Path to JSON file with test data (list of {query, ground_truth} objects)",
    )
    parser.addoption(
        "--query",
        action="store",
        default=None,
        help="Single query for inline test mode",
    )
    parser.addoption(
        "--ground-truth",
        action="store",
        default=None,
        help="Single ground truth for inline test mode",
    )


@pytest.fixture(scope="session")
def test_dataset(pytestconfig) -> List[Dict[str, Any]]:
    """
    Returns the full list of {query, ground_truth} test cases.
    Priority:
      1. --test-data-file CLI argument
      2. RAGAS_TEST_DATA_FILE env variable
      3. --query + --ground-truth CLI arguments
      4. Default sample data
    """
    data_file = (
        pytestconfig.getoption("--test-data-file")
        or os.environ.get("RAGAS_TEST_DATA_FILE")
    )

    if data_file:
        return _load_test_data_from_json(data_file)

    single_query = pytestconfig.getoption("--query") or os.environ.get("RAGAS_QUERY")
    single_gt = pytestconfig.getoption("--ground-truth") or os.environ.get("RAGAS_GROUND_TRUTH")

    if single_query and single_gt:
        return [{"test_id": "TC_INLINE", "query": single_query, "ground_truth": single_gt}]

    # Fallback: sample data bundled with the project
    default_path = Path(__file__).parent / "test_data" / "sample_test_data - With-API.json"
    if default_path.exists():
        return _load_test_data_from_json(str(default_path))

    raise RuntimeError(
        "No test data provided. Use --test-data-file, --query + --ground-truth, "
        "or set RAGAS_TEST_DATA_FILE env variable."
    )


# ─────────────────────────────────────────────
# 4. RAG response fixture (per test case)
# ─────────────────────────────────────────────

@pytest.fixture(scope="session")
def rag_responses(test_dataset, app_config) -> List[Dict[str, Any]]:
    """
    Calls the RAG API for every test case and collects:
      query, ground_truth, generated_answer, contexts
    These are returned as a list; each pytest test parametrizes over this list.
    """
    results = []
    for item in test_dataset:
        query = item["query"]
        ground_truth = item["ground_truth"]
        test_id = item.get("test_id", f"TC_{len(results) + 1:03d}")

        generated_answer, contexts = query_rag_system(item, app_config.rag_api)

        results.append(
            {
                "test_id": test_id,
                "query": query,
                "ground_truth": ground_truth,
                "generated_answer": generated_answer,
                "contexts": contexts,
            }
        )
    return results
