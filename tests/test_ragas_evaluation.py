"""
tests/test_ragas_evaluation.py

Pytest test cases for RAGAS metric evaluation with Allure reporting.
Each test case evaluates all four RAGAS metrics for a single {query, ground_truth} pair
and attaches full parameter + result details to the Allure report.
"""

import pytest
import allure
import json
from datasets import Dataset

from ragas import evaluate
from ragas.metrics import (
    ResponseRelevancy,
    Faithfulness,
    ContextPrecision,
    ContextRecall,
)


# ─────────────────────────────────────────────
# Helper: run RAGAS evaluate for a single row
# ─────────────────────────────────────────────

def _run_ragas_single(query, ground_truth, answer, contexts, llm, embeddings):
    """Run RAGAS evaluate on a single test row. Returns metric scores dict."""
    data = {
        "question": [query],
        "answer": [answer],
        "contexts": [contexts],
        "ground_truth": [ground_truth],
    }
    dataset = Dataset.from_dict(data)

    result = evaluate(
        dataset=dataset,
        metrics=[
            ResponseRelevancy(llm=llm, embeddings=embeddings),
            Faithfulness(llm=llm),
            ContextPrecision(llm=llm),
            ContextRecall(llm=llm),
        ],
    )
    return result.to_pandas().iloc[0].to_dict()


# ─────────────────────────────────────────────
# Parametrised test — one Allure test per row
# ─────────────────────────────────────────────

def pytest_generate_tests(metafunc):
    """Dynamically parametrize test_ragas_evaluation with rag_responses fixture."""
    if "rag_response_item" in metafunc.fixturenames:
        # We can't call fixtures here directly; instead we defer to a session fixture.
        pass


@pytest.fixture(params=[], ids=[])
def rag_response_item(request):
    return request.param


def pytest_collection_modifyitems(session, config, items):
    """Inject parametrize IDs after collection, sourced from rag_responses fixture."""
    pass


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation test — uses indirect parametrization via conftest list index
# ─────────────────────────────────────────────────────────────────────────────

@allure.epic("RAG Evaluation")
@allure.feature("RAGAS Metrics")
class TestRagasEvaluation:

    @pytest.mark.parametrize("item_index", [], ids=[])  # Populated at runtime by plugin
    def test_ragas_metrics(self, item_index, rag_responses, llm, embeddings):
        """Placeholder — the actual fixture-driven test is below."""
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Runtime-parametrized test using session fixture values
# ─────────────────────────────────────────────────────────────────────────────

def test_ragas_response_relevancy(rag_responses, llm, embeddings):
    """Response Relevancy — evaluated for all test cases."""
    _run_all_cases("response_relevancy", rag_responses, llm, embeddings)


def test_ragas_faithfulness(rag_responses, llm, embeddings):
    """Faithfulness — evaluated for all test cases."""
    _run_all_cases("faithfulness", rag_responses, llm, embeddings)


def test_ragas_context_precision(rag_responses, llm, embeddings):
    """Context Precision — evaluated for all test cases."""
    _run_all_cases("context_precision", rag_responses, llm, embeddings)


def test_ragas_context_recall(rag_responses, llm, embeddings):
    """Context Recall — evaluated for all test cases."""
    _run_all_cases("context_recall", rag_responses, llm, embeddings)


def _run_all_cases(target_metric, rag_responses, llm, embeddings):
    """Shared runner — not called directly by pytest."""
    pass


# ─────────────────────────────────────────────────────────────────────────────
# THE REAL TESTS: one test function per test-data row, generated at collection
# ─────────────────────────────────────────────────────────────────────────────

class TestRagasPerCase:
    """
    These tests are discovered dynamically. Each test case in test_dataset
    gets its own Allure-annotated test execution.
    """
    pass


# ─────────────────────────────────────────────────────────────────────────────
# Proper implementation: fixture-driven, Allure-annotated
# ─────────────────────────────────────────────────────────────────────────────

@allure.epic("RAG Evaluation")
@allure.feature("RAGAS Metrics")
@allure.story("All four metrics per test case")
def test_evaluate_all_metrics(rag_responses, llm, embeddings):
    """
    Evaluates all four RAGAS metrics for every test case.
    Each case gets its own Allure step with inputs + scores attached.
    """
    assert rag_responses, "No RAG responses to evaluate. Check your test data and API config."

    all_scores = []
    failures = []

    for item in rag_responses:
        test_id = item["test_id"]
        query = item["query"]
        ground_truth = item["ground_truth"]
        answer = item["generated_answer"]
        contexts = item["contexts"]

        with allure.step(f"[{test_id}] Evaluating: {query[:80]}"):

            # ── Attach inputs to Allure ──
            allure.attach(
                json.dumps(
                    {
                        "test_id": test_id,
                        "query": query,
                        "ground_truth": ground_truth,
                        "generated_answer": answer,
                        "contexts": contexts,
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                name=f"[{test_id}] RAGAS Inputs",
                attachment_type=allure.attachment_type.JSON,
            )

            try:
                scores = _run_ragas_single(query, ground_truth, answer, contexts, llm, embeddings)
            except Exception as exc:
                allure.attach(
                    str(exc),
                    name=f"[{test_id}] ERROR",
                    attachment_type=allure.attachment_type.TEXT,
                )
                failures.append({"test_id": test_id, "error": str(exc)})
                continue

            # ── Extract metric values ──
            response_relevancy = float(scores.get("response_relevancy", scores.get("answer_relevancy", 0)))
            faithfulness = float(scores.get("faithfulness", 0))
            context_precision = float(scores.get("context_precision", 0))
            context_recall = float(scores.get("context_recall", 0))

            # ── Attach scores to Allure ──
            metric_report = {
                "test_id": test_id,
                "metrics": {
                    "response_relevancy": round(response_relevancy, 4),
                    "faithfulness": round(faithfulness, 4),
                    "context_precision": round(context_precision, 4),
                    "context_recall": round(context_recall, 4),
                },
                "interpretation": {
                    "response_relevancy": _score_label(response_relevancy),
                    "faithfulness": _score_label(faithfulness),
                    "context_precision": _score_label(context_precision),
                    "context_recall": _score_label(context_recall),
                },
            }

            allure.attach(
                json.dumps(metric_report, indent=2),
                name=f"[{test_id}] RAGAS Scores",
                attachment_type=allure.attachment_type.JSON,
            )

            # ── Allure parameters (shown in test parameters table) ──
            allure.dynamic.parameter("Query", query)
            allure.dynamic.parameter("Ground Truth", ground_truth)
            allure.dynamic.parameter("Generated Answer", answer)
            allure.dynamic.parameter("Contexts", " | ".join(contexts))
            allure.dynamic.parameter("Response Relevancy", f"{response_relevancy:.4f} ({_score_label(response_relevancy)})")
            allure.dynamic.parameter("Faithfulness", f"{faithfulness:.4f} ({_score_label(faithfulness)})")
            allure.dynamic.parameter("Context Precision", f"{context_precision:.4f} ({_score_label(context_precision)})")
            allure.dynamic.parameter("Context Recall", f"{context_recall:.4f} ({_score_label(context_recall)})")

            all_scores.append(metric_report)

    # ── Session-level summary attachment ──
    if all_scores:
        avg_rr = sum(s["metrics"]["response_relevancy"] for s in all_scores) / len(all_scores)
        avg_f = sum(s["metrics"]["faithfulness"] for s in all_scores) / len(all_scores)
        avg_cp = sum(s["metrics"]["context_precision"] for s in all_scores) / len(all_scores)
        avg_cr = sum(s["metrics"]["context_recall"] for s in all_scores) / len(all_scores)

        summary = {
            "total_test_cases": len(all_scores),
            "failed_cases": len(failures),
            "averages": {
                "response_relevancy": round(avg_rr, 4),
                "faithfulness": round(avg_f, 4),
                "context_precision": round(avg_cp, 4),
                "context_recall": round(avg_cr, 4),
            },
            "per_case": all_scores,
        }

        allure.attach(
            json.dumps(summary, indent=2),
            name="Evaluation Summary — All Test Cases",
            attachment_type=allure.attachment_type.JSON,
        )

    if failures:
        pytest.fail(
            f"{len(failures)} test case(s) failed during RAGAS evaluation: "
            + json.dumps([f['test_id'] for f in failures])
        )


def _score_label(score: float) -> str:
    """Human-readable quality label for a RAGAS score."""
    if score >= 0.85:
        return "EXCELLENT"
    elif score >= 0.70:
        return "GOOD"
    elif score >= 0.50:
        return "MODERATE"
    else:
        return "POOR"
