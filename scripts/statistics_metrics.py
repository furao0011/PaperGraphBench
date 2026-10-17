"""Count-based metrics and equal-paper aggregation for evaluation reports."""

from __future__ import annotations

import math
from collections import defaultdict
from statistics import mean, stdev


METRICS = {
    "global_kc_coverage": ("coverage_metrics", "global_kc_lit_count", "global_kc_total"),
    "target_kc_coverage": ("coverage_metrics", "target_kc_lit_count", "target_kc_total"),
    "critical_kc_coverage": ("coverage_metrics", "critical_kc_lit_count", "critical_kc_total"),
    "macro_completion": ("macro_metrics", None, "macro_total"),
    "challenge_failure": ("challenge_metrics", "challenge_failure_count", "challenge_total"),
    "hallucination_turn_rate": ("hallucination_metrics", "hallucination_turn_count", None),
    "hallucination_event_density": ("hallucination_metrics", "hallucination_event_count", None),
    "hallucination_event_closure": ("hallucination_metrics", None, "hallucination_event_count"),
    "detail_completion": ("detail_completion_metrics", "coverage_gap_resolved_count", "coverage_gap_count"),
    "bridge_reasoning_success": ("thread_metrics", "bridge_reasoning_success_count", "bridge_reasoning_tested_count"),
}


def count(value, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer, got {value!r}")
    return value


def extract_metrics(report: dict) -> dict:
    """Recompute ratios; zero denominators are NA, never the reporter's 0.0."""
    values = {"total_turns": count(report["summary"]["total_turns"], "total_turns")}
    for name, (section, numerator, denominator) in METRICS.items():
        source = report.get(section, {})
        if not isinstance(source, dict):
            raise ValueError(f"{section} must be an object")
        if not source:
            values.update({name: None, name + "_numerator": None, name + "_denominator": None})
            continue
        total = values["total_turns"] if denominator is None else count(source.get(denominator), denominator)
        if name == "macro_completion":
            rate = source.get("macro_completion_rate")
            if isinstance(rate, bool) or not isinstance(rate, (float, int)) or not math.isfinite(rate) or not 0 <= rate <= 1:
                raise ValueError("macro_completion_rate must be a finite fraction within [0, 1]")
            completed = round(rate * total)
            if (not total and rate) or (total and abs(round(completed / total, 4) - rate) > 0.000001):
                raise ValueError("macro_completion_rate disagrees with macro_total")
            if total and any(0 <= other <= total and abs(round(other / total, 4) - rate) < 0.000001
                             for other in (completed - 1, completed + 1)):
                raise ValueError("Rounded macro_completion_rate does not uniquely determine its numerator")
        elif name == "hallucination_event_closure":
            statuses = source.get("hallucination_by_status")
            if not isinstance(statuses, dict):
                raise ValueError("hallucination_by_status must contain final event statuses")
            if sum(count(value, f"hallucination_by_status.{key}") for key, value in statuses.items()) != total:
                raise ValueError("Final hallucination status counts disagree with event count")
            completed = count(statuses.get("resolved", 0), "resolved hallucination events")
        else:
            completed = count(source.get(numerator), numerator)
        if name != "hallucination_event_density" and completed > total:
            raise ValueError(f"{name} numerator exceeds denominator")
        values.update({name: completed / total if total else None,
                       name + "_numerator": completed, name + "_denominator": total})
    return values


def aggregate(rows: list[dict], expected: list[dict] | None) -> list[dict]:
    """Across repetitions average each paper first, then weight papers equally."""
    groups = defaultdict(list)
    for row in rows:
        groups[row["model"]].append(row)
    result = []
    for model, model_rows in sorted(groups.items()):
        completed = [row for row in model_rows if row["status"] == "completed"]
        planned = sum(task["model"] == model for task in expected) if expected is not None else None
        summary = {"model": model, "planned_tasks": planned, "discovered_tasks": sum(row["status"] != "missing" for row in model_rows),
                   "completed_tasks": len(completed), "incomplete_tasks": len(model_rows) - len(completed),
                   "completed_papers": len({row["paper_id"] for row in completed}),
                   "status": "incomplete" if len(completed) != len(model_rows) else "completed"}
        summary["total_turns_sum"] = sum(row["total_turns"] for row in completed)
        events = [row["hallucination_event_density_numerator"] for row in completed if row["hallucination_event_density_numerator"] is not None]
        summary["hallucination_events_sum"] = sum(events) if events else None
        summary["hallucination_events_valid_runs"] = len(events)
        for name in METRICS:
            by_paper = defaultdict(list)
            valid = [row for row in completed if row[name] is not None]
            for row in valid:
                by_paper[row["paper_id"]].append(row[name])
            paper_means = [mean(rates) for rates in by_paper.values()]
            numerator = sum(row[name + "_numerator"] for row in valid)
            denominator = sum(row[name + "_denominator"] for row in valid)
            summary.update({name + "_mean": mean(paper_means) if paper_means else None,
                            name + "_std": stdev(paper_means) if len(paper_means) > 1 else None,
                            name + "_valid_papers": len(paper_means), name + "_valid_runs": len(valid),
                            name + "_micro": numerator / denominator if denominator else None,
                            name + "_numerator_sum": numerator, name + "_denominator_sum": denominator})
        result.append(summary)
    return result
