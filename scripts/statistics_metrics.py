"""Paper-aligned trajectory metrics with explicit paper-mean/pooled aggregation."""

from __future__ import annotations

import math
from collections import defaultdict
from statistics import mean, stdev

from statistics_repairs import extract_repair_tasks
from statistics_trajectory import trajectory_counts


# Primary values follow the paper; the alternative averaging is also exported.
METRICS = {
    "one_turn_kc_coverage": "paper_mean",
    "one_turn_macro_complete_rate": "paper_mean",
    "challenge_failure_rate": "pooled",
    "text_challenge_failure_rate": "pooled",
    "multimodal_challenge_failure_rate": "pooled",
    "mean_hallucination_events": "paper_mean",
    "hallucination_turn_rate": "pooled",
    "hallucination_event_density": "pooled",
    "detail_repair_success_rate": "pooled",
    "hallucination_repair_success_rate": "pooled",
    "detail_repair_avg_rounds": "pooled",
    "hallucination_repair_avg_rounds": "pooled",
    "mean_total_turns": "paper_mean",
    "mean_answer_chars_per_turn": "pooled",
}


def count(value, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer, got {value!r}")
    return value


def extract_metrics(report: dict, trajectory: dict) -> dict:
    """Recompute from recorded turns, never final-coverage or event-closure aliases."""
    counts = trajectory_counts(trajectory)
    reported = count(report["summary"]["total_turns"], "summary.total_turns")
    if reported != counts["total_turns"]:
        raise ValueError("Report total_turns disagrees with the trajectory")
    tasks = extract_repair_tasks(trajectory)
    detail = [task for task in tasks if task["repair_type"] == "detail"]
    hallucination = [task for task in tasks if task["repair_type"] == "hallucination"]
    pairs = {
        "one_turn_kc_coverage": (counts["macro_covered_kc_mentions"], counts["macro_target_kc_mentions"]),
        "one_turn_macro_complete_rate": (counts["macro_complete_turns"], counts["macro_turns"]),
        "challenge_failure_rate": (counts["challenge_failed_turns"], counts["challenge_turns"]),
        "text_challenge_failure_rate": (counts["text_challenge_failed_turns"], counts["text_challenge_turns"]),
        "multimodal_challenge_failure_rate": (counts["multimodal_challenge_failed_turns"], counts["multimodal_challenge_turns"]),
        "mean_hallucination_events": (counts["hallucination_events"], 1),
        "hallucination_turn_rate": (counts["hallucination_turns"], counts["total_turns"]),
        "hallucination_event_density": (counts["hallucination_events"], counts["total_turns"]),
        "detail_repair_success_rate": (sum(task["success"] for task in detail), len(detail)),
        "hallucination_repair_success_rate": (sum(task["success"] for task in hallucination), len(hallucination)),
        "detail_repair_avg_rounds": (sum(task["attempts"] for task in detail), len(detail)),
        "hallucination_repair_avg_rounds": (sum(task["attempts"] for task in hallucination), len(hallucination)),
        "mean_total_turns": (counts["total_turns"], 1),
        "mean_answer_chars_per_turn": (counts["answer_chars"], counts["total_turns"]),
    }
    values = {"total_turns": counts["total_turns"]}
    for name, (numerator, denominator) in pairs.items():
        values.update({name: numerator / denominator if denominator else None,
                       name + "_numerator": numerator, name + "_denominator": denominator})
    return values


def _validated_pair(row: dict, name: str) -> tuple[int, int]:
    numerator = count(row[name + "_numerator"], name + "_numerator")
    denominator = count(row[name + "_denominator"], name + "_denominator")
    value = row[name]
    if not denominator:
        if value is not None or numerator:
            raise ValueError(f"{name} with zero denominator must have zero numerator and NA value")
    elif (isinstance(value, bool) or not isinstance(value, (float, int))
          or not math.isfinite(value) or not math.isclose(value, numerator / denominator, rel_tol=1e-12, abs_tol=1e-12)):
        raise ValueError(f"{name} disagrees with its count numerator/denominator")
    return numerator, denominator


def aggregate(rows: list[dict], expected: list[dict] | None) -> list[dict]:
    """Paper means for coverage/counts; pooled count ratios for challenges/repair."""
    groups = defaultdict(list)
    for row in rows:
        groups[row["model"]].append(row)
    result = []
    for model, model_rows in sorted(groups.items()):
        completed = [row for row in model_rows if row["status"] == "completed"]
        planned = sum(task["model"] == model for task in expected) if expected is not None else None
        summary = {"model": model, "planned_tasks": planned,
                   "discovered_tasks": sum(row["status"] != "missing" for row in model_rows),
                   "completed_tasks": len(completed), "incomplete_tasks": len(model_rows) - len(completed),
                   "completed_papers": len({row["paper_id"] for row in completed}),
                   "status": "incomplete" if len(completed) != len(model_rows) else "completed"}
        summary["total_turns_sum"] = sum(count(row["total_turns"], "total_turns") for row in completed)
        summary["hallucination_events_sum"] = sum(count(row["mean_hallucination_events_numerator"], "hallucination_events") for row in completed) if completed else None
        summary["hallucination_events_valid_runs"] = len(completed)
        for name, method in METRICS.items():
            pairs = [_validated_pair(row, name) for row in completed]
            valid = [row for row in completed if row[name] is not None]
            by_paper = defaultdict(list)
            for row in valid:
                by_paper[row["paper_id"]].append(row[name])
            paper_means = [mean(rates) for rates in by_paper.values()]
            paper_mean = mean(paper_means) if paper_means else None
            numerator, denominator = sum(pair[0] for pair in pairs), sum(pair[1] for pair in pairs)
            pooled = numerator / denominator if denominator else None
            summary.update({name: paper_mean if method == "paper_mean" else pooled,
                            name + "_paper_mean": paper_mean, name + "_micro": pooled,
                            name + "_std": stdev(paper_means) if len(paper_means) > 1 else None,
                            name + "_valid_papers": len(paper_means), name + "_valid_runs": len(valid),
                            name + "_numerator_sum": numerator, name + "_denominator_sum": denominator})
        result.append(summary)
    return result
