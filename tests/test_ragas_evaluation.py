"""
tests/test_ragas_evaluation.py

Pytest test for RAGAS metric evaluation with Allure reporting. Evaluates every
test case in the `rag_responses` fixture (see conftest.py) against the metrics
configured in config.yaml's ragas.metrics list, and attaches full parameter +
result details to the Allure report.

Run without opening the Streamlit UI, e.g.:
    pytest tests/test_ragas_evaluation.py --test-data-file test_data/my_tests.json
    RAGAS_TEST_DATA_FILE=test_data/my_tests.json pytest tests/test_ragas_evaluation.py
"""

import json

import allure
import pytest

from ragas_metrics import configured_metric_keys, evaluate_single_row, METRIC_LABELS


def _score_label(score: float) -> str:
    if score >= 0.85:
        return "EXCELLENT"
    elif score >= 0.70:
        return "GOOD"
    elif score >= 0.50:
        return "MODERATE"
    else:
        return "POOR"


@allure.epic("RAG Evaluation")
@allure.feature("RAGAS Metrics")
@allure.story("Metrics per test case (config-driven)")
def test_evaluate_all_metrics(rag_responses, app_config, llm, embeddings):
    """Evaluates config.yaml's configured RAGAS metrics for every test case.

    Each case gets its own Allure step with inputs + scores attached.
    """
    assert rag_responses, "No RAG responses to evaluate. Check your test data and API config."

    metric_keys = configured_metric_keys(app_config)
    all_scores = []
    failures = []

    for item in rag_responses:
        test_id = item["test_id"]
        query = item["query"]
        ground_truth = item["ground_truth"]
        answer = item["generated_answer"]
        contexts = item["contexts"]

        with allure.step(f"[{test_id}] Evaluating: {query[:80]}"):
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

            scores, duration_s = evaluate_single_row(
                llm, embeddings, metric_keys, query, answer, contexts, ground_truth
            )
            missing = [METRIC_LABELS[k] for k, v in scores.items() if v != v]  # NaN check

            metric_report = {
                "test_id": test_id,
                "duration_s": round(duration_s, 2),
                "metrics": {k: round(v, 4) for k, v in scores.items() if v == v},
                "interpretation": {k: _score_label(v) for k, v in scores.items() if v == v},
            }
            if missing:
                metric_report["could_not_compute"] = missing
                failures.append({"test_id": test_id, "missing_metrics": missing})

            allure.attach(
                json.dumps(metric_report, indent=2),
                name=f"[{test_id}] RAGAS Scores",
                attachment_type=allure.attachment_type.JSON,
            )

            allure.dynamic.parameter("Query", query)
            allure.dynamic.parameter("Ground Truth", ground_truth)
            allure.dynamic.parameter("Generated Answer", answer)
            allure.dynamic.parameter("Contexts", " | ".join(contexts))
            allure.dynamic.parameter("Duration (s)", f"{duration_s:.2f}")
            for k, v in scores.items():
                label = METRIC_LABELS[k]
                value_str = "—" if v != v else f"{v:.4f} ({_score_label(v)})"
                allure.dynamic.parameter(label, value_str)

            all_scores.append(metric_report)

    if all_scores:
        averages = {}
        for k in metric_keys:
            values = [s["metrics"][k] for s in all_scores if k in s["metrics"]]
            if values:
                averages[k] = round(sum(values) / len(values), 4)

        summary = {
            "total_test_cases": len(all_scores),
            "failed_cases": len(failures),
            "metrics_evaluated": metric_keys,
            "averages": averages,
            "per_case": all_scores,
        }
        allure.attach(
            json.dumps(summary, indent=2),
            name="Evaluation Summary — All Test Cases",
            attachment_type=allure.attachment_type.JSON,
        )

    if failures:
        pytest.fail(
            f"{len(failures)} of {len(all_scores)} test case(s) have metrics that could not be "
            f"computed: {json.dumps(failures)}"
        )
