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

from ragas_metrics import (
    configured_metric_keys,
    evaluate_single_row,
    diagnose_row,
    TooManyContextsError,
    METRIC_LABELS,
)


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

            try:
                scores, duration_s, errors, token_usage, metric_reasons = evaluate_single_row(
                    llm, embeddings, metric_keys, query, answer, contexts, ground_truth
                )
            except TooManyContextsError as exc:
                allure.attach(
                    str(exc),
                    name=f"[{test_id}] Context Count Error",
                    attachment_type=allure.attachment_type.TEXT,
                )
                allure.dynamic.parameter("Error", str(exc))
                failures.append({"test_id": test_id, "error": str(exc)})
                continue
            missing = [METRIC_LABELS[k] for k, v in scores.items() if v != v]  # NaN check
            diagnosis = diagnose_row(scores)

            metric_report = {
                "test_id": test_id,
                "duration_s": round(duration_s, 2),
                "tokens": token_usage,
                "metrics": {k: round(v, 4) for k, v in scores.items() if v == v},
                "interpretation": {k: _score_label(v) for k, v in scores.items() if v == v},
                "diagnosis": diagnosis,
            }
            if missing:
                metric_report["could_not_compute"] = missing
                metric_report["errors"] = {METRIC_LABELS[k]: msg for k, msg in errors.items()}
                failures.append({
                    "test_id": test_id,
                    "missing_metrics": missing,
                    "errors": metric_report["errors"],
                })

            allure.attach(
                json.dumps(metric_report, indent=2),
                name=f"[{test_id}] RAGAS Scores",
                attachment_type=allure.attachment_type.JSON,
            )
            if metric_reasons:
                # The judge LLM's own per-chunk/per-statement reasoning for why each
                # metric scored the way it did -- kept as its own attachment since it
                # can be long (up to MAX_CONTEXTS chunks' worth for context_precision).
                allure.attach(
                    json.dumps(metric_reasons, indent=2, ensure_ascii=False),
                    name=f"[{test_id}] RAGAS Metric Reasons",
                    attachment_type=allure.attachment_type.JSON,
                )

            allure.dynamic.parameter("Query", query)
            allure.dynamic.parameter("Ground Truth", ground_truth)
            allure.dynamic.parameter("Generated Answer", answer)
            allure.dynamic.parameter("Contexts", " | ".join(contexts))
            allure.dynamic.parameter("Duration (s)", f"{duration_s:.2f}")
            allure.dynamic.parameter(
                "Tokens Used",
                f"{token_usage['total_tokens']} (in: {token_usage['input_tokens']}, "
                f"out: {token_usage['output_tokens']})",
            )
            allure.dynamic.parameter(
                "Suggested Focus",
                " + ".join(diagnosis["focus_areas"]) if diagnosis["focus_areas"] else "Healthy",
            )
            for k, v in scores.items():
                label = METRIC_LABELS[k]
                if v == v:
                    value_str = f"{v:.4f} ({_score_label(v)})"
                elif k in errors:
                    value_str = f"— ({errors[k]})"
                else:
                    value_str = "—"
                allure.dynamic.parameter(label, value_str)

            all_scores.append(metric_report)

    if all_scores:
        averages = {}
        for k in metric_keys:
            values = [s["metrics"][k] for s in all_scores if k in s["metrics"]]
            if values:
                averages[k] = round(sum(values) / len(values), 4)

        focus_area_counts = {}
        for s in all_scores:
            for area in s["diagnosis"]["focus_areas"]:
                focus_area_counts[area] = focus_area_counts.get(area, 0) + 1

        summary = {
            "total_test_cases": len(all_scores),
            "failed_cases": len(failures),
            "metrics_evaluated": metric_keys,
            "averages": averages,
            "total_tokens": {
                "input_tokens": sum(s["tokens"]["input_tokens"] for s in all_scores),
                "output_tokens": sum(s["tokens"]["output_tokens"] for s in all_scores),
                "total_tokens": sum(s["tokens"]["total_tokens"] for s in all_scores),
            },
            "suggested_focus_areas": focus_area_counts,
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
