"""Present the four evaluation dimensions without changing source metrics."""

from __future__ import annotations


# Challenge failure already denotes hallucination in the challenge judge.
# This is a turn-based ratio, not the reporter's similarly named event bucket.
DIMENSIONS = (
    ("knowledge_coverage", "知识覆盖", "global_kc_coverage", "global_kc_coverage"),
    ("challenge_induced_hallucination", "挑战诱发幻觉", "challenge_induced_hallucination", "challenge_failure"),
    ("hallucination_behavior", "幻觉行为", "hallucination_turn_rate", "hallucination_turn_rate"),
    ("hallucination_behavior", "幻觉行为", "hallucination_event_density", "hallucination_event_density"),
    ("error_repair", "错误修复", "hallucination_event_closure", "hallucination_event_closure"),
    ("error_repair", "错误修复", "detail_completion", "detail_completion"),
)

DIMENSION_COLUMNS = [
    "model", "status", "dimension_id", "dimension", "metric", "source_metric",
    "mean", "std", "valid_papers", "valid_runs", "micro", "numerator_sum", "denominator_sum",
]


def dimension_rows(models: list[dict]) -> list[dict]:
    """Expose six metrics under four dimensions, retaining all denominators."""
    result = []
    for model in models:
        for dimension_id, label, metric, source in DIMENSIONS:
            row = {"model": model["model"], "status": model["status"], "dimension_id": dimension_id,
                   "dimension": label, "metric": metric, "source_metric": source}
            for field in ("mean", "std", "valid_papers", "valid_runs", "micro", "numerator_sum", "denominator_sum"):
                row[field] = model[f"{source}_{field}"]
            result.append(row)
    return result
