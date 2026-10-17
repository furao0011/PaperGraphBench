"""Present the four evaluation dimensions without changing source metrics."""

from __future__ import annotations


# The four dimensions retain the paper's distinct denominators and units.
DIMENSIONS = (
    ("knowledge_coverage", "知识覆盖", "one_turn_kc_coverage", "one_turn_kc_coverage"),
    ("knowledge_coverage", "知识覆盖", "one_turn_macro_complete_rate", "one_turn_macro_complete_rate"),
    ("challenge_induced_hallucination", "挑战诱发幻觉", "challenge_induced_hallucination", "challenge_failure_rate"),
    ("challenge_induced_hallucination", "挑战诱发幻觉", "text_challenge_induced_hallucination", "text_challenge_failure_rate"),
    ("challenge_induced_hallucination", "挑战诱发幻觉", "multimodal_challenge_induced_hallucination", "multimodal_challenge_failure_rate"),
    ("hallucination_behavior", "幻觉行为", "mean_hallucination_events", "mean_hallucination_events"),
    ("hallucination_behavior", "幻觉行为", "hallucination_turn_rate", "hallucination_turn_rate"),
    ("error_repair", "错误修复", "detail_repair_success_rate", "detail_repair_success_rate"),
    ("error_repair", "错误修复", "detail_repair_avg_rounds", "detail_repair_avg_rounds"),
    ("error_repair", "错误修复", "hallucination_repair_success_rate", "hallucination_repair_success_rate"),
    ("error_repair", "错误修复", "hallucination_repair_avg_rounds", "hallucination_repair_avg_rounds"),
)

DIMENSION_COLUMNS = [
    "model", "status", "dimension_id", "dimension", "metric", "source_metric",
    "value", "aggregation", "paper_std", "valid_papers", "valid_runs", "micro", "numerator_sum", "denominator_sum",
]


def dimension_rows(models: list[dict]) -> list[dict]:
    """Expose eleven metrics under four dimensions, retaining all denominators."""
    from statistics_metrics import METRICS

    result = []
    for model in models:
        for dimension_id, label, metric, source in DIMENSIONS:
            row = {"model": model["model"], "status": model["status"], "dimension_id": dimension_id,
                   "dimension": label, "metric": metric, "source_metric": source,
                   "value": model[source], "aggregation": METRICS[source], "paper_std": model[source + "_std"]}
            for field in ("valid_papers", "valid_runs", "micro", "numerator_sum", "denominator_sum"):
                row[field] = model[f"{source}_{field}"]
            result.append(row)
    return result
