"""Compare offline statistics with the frozen paper results, without model APIs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "tests" / "fixtures" / "paper_reference_results.json"


def _json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _number(value, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{field} must be a finite number, got {value!r}")
    return value


def _count(value, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a nonnegative integer")
    return value


def _index(rows: list[dict], source: str) -> dict[str, dict]:
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{source} requires model rows")
    indexed = {}
    for row in rows:
        model = row.get("model") if isinstance(row, dict) else None
        if not isinstance(model, str) or not model.strip() or model in indexed:
            raise ValueError(f"{source} contains missing or duplicate model IDs")
        indexed[model] = row
    return indexed


def _csv_categories(path: Path, categories: list[str]) -> dict[str, dict]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for field in [*categories, "event_count", "categorized_event_count", "unmapped_event_count",
                      "task_total", "completed_tasks", "incomplete_tasks"]:
            raw = row.get(field)
            if not isinstance(raw, str) or not raw.isascii() or not raw.isdecimal():
                raise ValueError(f"Invalid category CSV count {field}: {path}")
            row[field] = int(raw)
    return _index(rows, str(path))


def _known_costs(reference: dict, models: dict) -> dict[tuple[str, str], dict]:
    """Only an explicit frozen model/field/actual/reference tuple is permitted."""
    indexed = {}
    rows = reference.get("known_cost_discrepancies", [])
    if not isinstance(rows, list):
        raise ValueError("Frozen known_cost_discrepancies must be a list")
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Frozen known cost discrepancy must be an object")
        model, field = row.get("model"), row.get("field")
        if model not in models or field != "mean_answer_chars_per_turn" or (model, field) in indexed:
            raise ValueError("Invalid or duplicate frozen known cost discrepancy")
        _number(row.get("actual"), "known cost actual")
        if _number(row.get("reference"), "known cost reference") != models[model]["chars_per_turn"]:
            raise ValueError("Known cost reference disagrees with the frozen paper cost")
        indexed[model, field] = row
    return indexed


def verify_paper_results(statistics_dir: Path, reference_path: Path = REFERENCE) -> dict:
    """Verify the original fixed experiment, not arbitrary new model evaluations."""
    directory = Path(statistics_dir).expanduser().resolve()
    reference = _json(Path(reference_path))
    summary = _json(directory / "summary.json")
    expected_models = _index(reference["models"], "paper reference")
    known_costs = _known_costs(reference, expected_models)
    actual_models = _index(summary["models"], "summary.json")
    categories = reference["category_metrics"]
    category_rows = _csv_categories(directory / "failure_category_by_model.csv", categories)
    wanted = reference["expected"]
    core_issues, cost_issues = [], []
    checks = {name: {"checked": 0, "matched": 0} for name in ("main_metrics", "repair_metrics", "failure_categories", "cost")}

    def same(field, actual, expected, model=None, *, cost=False, tolerance=0):
        matching = (math.isclose(_number(actual, field), expected, rel_tol=0, abs_tol=tolerance)
                    if isinstance(expected, (int, float)) and not isinstance(expected, bool) else actual == expected)
        if not matching:
            issue = {"field": field, "actual": actual, "reference": expected}
            if model is not None:
                issue["model"] = model
            if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
                issue["difference"] = actual - expected
            (cost_issues if cost else core_issues).append(issue)
        return matching

    same("summary.status", summary.get("status"), "completed")
    for source, indexed in (("summary.models", actual_models), ("category.models", category_rows)):
        same(source, sorted(indexed), sorted(expected_models))
        same(source + ".count", len(indexed), wanted["models"])
    counts = summary["task_counts"]
    for field, expected in (("completed", wanted["tasks"]), ("discovered", wanted["tasks"]), ("incomplete", 0)):
        same("task_counts." + field, _count(counts.get(field), field), expected)
    if counts.get("planned") is not None:
        same("task_counts.planned", _count(counts["planned"], "planned"), wanted["tasks"])
    classification = summary["classification"]
    for field, expected in (("event_count", wanted["hallucination_events"]),
                            ("categorized_event_count", wanted["hallucination_events"]),
                            ("unmapped_event_count", wanted["unmapped_events"])):
        same("classification." + field, _count(classification.get(field), field), expected)
    for field, expected in (("total_turns_sum", wanted["total_turns"]),
                            ("hallucination_events_sum", wanted["hallucination_events"])):
        same("models." + field, sum(_count(row.get(field), field) for row in actual_models.values()), expected)

    for model, reference_row in expected_models.items():
        if model not in actual_models or model not in category_rows:
            continue
        actual, category = actual_models[model], category_rows[model]
        same("model.status", actual.get("status"), "completed", model)
        same("category.status", category.get("status"), "completed", model)
        for field in ("completed_tasks", "completed_papers"):
            same(field, _count(actual.get(field), field), wanted["tasks_per_model"], model)
        same("incomplete_tasks", _count(actual.get("incomplete_tasks"), "incomplete_tasks"), 0, model)
        for field, expected in (("task_total", wanted["tasks_per_model"]),
                                ("completed_tasks", wanted["tasks_per_model"]), ("incomplete_tasks", 0)):
            same("category." + field, category[field], expected, model)
        for group, reference_key in (("main_metrics", "main"), ("repair_metrics", "repair")):
            keys, values = reference[group], reference_row[reference_key]
            if len(keys) != len(values):
                raise ValueError(f"Invalid frozen reference dimensions: {model}/{group}")
            for field, expected in zip(keys, values):
                checks[group]["checked"] += 1
                checks[group]["matched"] += same(field, actual.get(field), expected, model,
                    tolerance=reference["tolerance"]["core_absolute"])
        if len(categories) != len(reference_row["categories"]):
            raise ValueError(f"Invalid frozen category dimensions: {model}")
        for field, expected in zip(categories, reference_row["categories"]):
            checks["failure_categories"]["checked"] += 1
            checks["failure_categories"]["matched"] += same(field, category[field], expected, model)
        same("category.categorized_event_count", category["categorized_event_count"], sum(category[field] for field in categories), model)
        same("category.event_count", category["event_count"], category["categorized_event_count"] + category["unmapped_event_count"], model)
        same("category.event_count_vs_summary", category["event_count"], actual["hallucination_events_sum"], model)
        same("category.unmapped_event_count", category["unmapped_event_count"], wanted["unmapped_events"], model)
        checks["cost"]["checked"] += 1
        checks["cost"]["matched"] += same("mean_answer_chars_per_turn", actual.get("mean_answer_chars_per_turn"),
            reference_row["chars_per_turn"], model, cost=True,
            tolerance=reference["tolerance"]["displayed_chars_absolute"])
    # A known archived discrepancy is a frozen pair, not a model-wide waiver.
    # Even moving that model's result onto the printed paper value is a change.
    for (model, field), known in known_costs.items():
        if model not in actual_models:
            continue
        actual = _number(actual_models[model].get(field), field)
        if math.isclose(actual, known["actual"], rel_tol=0, abs_tol=reference["tolerance"]["core_absolute"]):
            continue
        issue = next((issue for issue in cost_issues if issue["model"] == model and issue["field"] == field), None)
        if issue is None:
            issue = {"model": model, "field": field, "actual": actual, "reference": known["reference"],
                     "difference": actual - known["reference"]}
            cost_issues.append(issue)
        issue["expected_archived_actual"] = known["actual"]
    known_observed, unexpected_costs = [], []
    for issue in cost_issues:
        known = known_costs.get((issue["model"], issue["field"]))
        exact_known = known is not None and all(math.isclose(issue[key], known[key], rel_tol=0,
            abs_tol=reference["tolerance"]["core_absolute"]) for key in ("actual", "reference"))
        (known_observed if exact_known else unexpected_costs).append(issue)
    status = ("core_failed" if core_issues else "failed_cost_regression" if unexpected_costs
              else "core_pass_with_cost_discrepancy" if cost_issues else "passed")
    return {"status": status, "statistics_dir": str(directory), "reference": str(Path(reference_path).resolve()),
            "checks": checks, "core_discrepancies": core_issues, "cost_discrepancies": cost_issues,
            "known_cost_discrepancies": known_observed, "unexpected_cost_discrepancies": unexpected_costs,
            "inventory": {"models": len(actual_models), "completed_tasks": counts["completed"],
                          "events": classification["event_count"], "unmapped_events": classification["unmapped_event_count"]},
            "interpretation": "Core checks compare the fixed paper experiment, not a guarantee for new runs. Only exact frozen known cost discrepancies are permitted; new cost regressions fail without altering results.",
            "paper_cost_note": reference["cost_note"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--statistics-dir", required=True, type=Path, help="Existing summary.json and failure_category_by_model.csv directory.")
    parser.add_argument("--output", type=Path, help="Optionally save the verification JSON; stdout is always printed.")
    args = parser.parse_args(argv)
    try:
        result = verify_paper_results(args.statistics_dir)
        rendered = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered, encoding="utf-8")
        print(rendered, end="")
        return 2 if result["status"] in {"core_failed", "failed_cost_regression"} else 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"Paper verification error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
