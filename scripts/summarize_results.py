"""Summarize existing reports into per-paper CSV and equal-paper model results."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "papergraph_demo"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from statistics_inputs import collect_rows, planned_tasks
from statistics_metrics import METRICS, aggregate
from statistics_classification import EVENT_COLUMNS, collect_classification
from statistics_dimensions import DIMENSION_COLUMNS, dimension_rows


def _csv_value(value):
    """Keep untrusted text literal in spreadsheet apps; preserve numbers and NA."""
    if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")):
        return "'" + value
    return value


def _csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="raise")
        writer.writeheader()
        writer.writerows({key: _csv_value(value) for key, value in row.items()} for row in rows)


def _classification_csv(classification: dict) -> tuple[list[dict], list[dict]]:
    """Keep original labels in JSON; encode structured CSV cells explicitly."""
    events = [{key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
               for key, value in event.items()} for event in classification["events"]]
    models = []
    for model in classification["summary"]["models"]:
        counts = model["task_counts"]
        models.append({"model": model["model"], "status": model["status"],
                       "task_total": counts["total"], "completed_tasks": counts["completed"],
                       "incomplete_tasks": counts["incomplete"], "event_count": model["event_count"],
                       "three_fields_present_count": model["three_fields_present_count"],
                       "three_fields_present_rate": model["three_fields_present_rate"],
                       "non_repeating_path_count": model["non_repeating_path_count"],
                       "warning_counts": json.dumps(model["warning_counts"], ensure_ascii=False, sort_keys=True)})
    return events, models


def summarize(result_root: Path, output_dir: Path | None = None, expected_tasks: list[dict] | None = None) -> dict:
    """Validate first; no API calls, configuration loading or dataset mutations."""
    root = Path(result_root).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Result directory does not exist: {root}")
    expected = planned_tasks(root, expected_tasks)
    rows = collect_rows(root, expected)
    completed = sum(row["status"] == "completed" for row in rows)
    models = aggregate(rows, expected)
    classification = collect_classification(root, rows)
    events, classification_models = _classification_csv(classification)
    dimensions = dimension_rows(models)
    result = {"status": "completed" if completed == len(rows) else "incomplete", "result_root": str(root),
              "task_counts": {"planned": len(expected) if expected is not None else None,
                              "discovered": sum(row["status"] != "missing" for row in rows),
                              "completed": completed, "incomplete": len(rows) - completed},
              "aggregation": {"macro": "Average each paper's valid repeated runs first, then weight papers equally",
                              "std": "Sample standard deviation across paper means; NA with fewer than two papers",
                              "micro": "Ratio of count sums across valid completed runs; repeat-weighted",
                              "na": "Missing metric or zero denominator; excluded from means, never replaced by zero",
                              "plan": "Known from manifest/explicit tasks" if expected is not None else "Unknown: only discovered reports were checked"},
              "models": models, "four_dimensions": dimensions, "classification": classification["summary"]}
    destination = Path(output_dir).expanduser().resolve() if output_dir is not None else root / "statistics"
    destination.mkdir(parents=True, exist_ok=True)
    per_paper_columns = ["round", "paper_id", "model", "status", "reason", "report_path", "total_turns"]
    for metric in METRICS:
        per_paper_columns += [metric, metric + "_numerator", metric + "_denominator"]
    summary_columns = list(models[0])
    _csv(destination / "per_paper.csv", rows, per_paper_columns)
    _csv(destination / "summary.csv", models, summary_columns)
    _csv(destination / "four_dimensions.csv", dimensions, DIMENSION_COLUMNS)
    _csv(destination / "classification_events.csv", events, EVENT_COLUMNS)
    _csv(destination / "classification_summary.csv", classification_models, [
        "model", "status", "task_total", "completed_tasks", "incomplete_tasks", "event_count",
        "three_fields_present_count", "three_fields_present_rate", "non_repeating_path_count", "warning_counts",
    ])
    paths = [{"model": model["model"], "family": path["family"], "type": path["type"],
              "subtype": path["subtype"], "event_count": path["count"]}
             for model in classification["summary"]["models"] for path in model["classification_counts"]]
    _csv(destination / "classification_paths.csv", paths, ["model", "family", "type", "subtype", "event_count"])
    for name, value in (("classification_events", classification["events"]),
                        ("classification_summary", classification["summary"])):
        (destination / f"{name}.json").write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (destination / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    result["output_dir"] = str(destination)
    return result


def print_summary(result: dict) -> None:
    counts = result["task_counts"]
    print(f"[statistics] status={result['status']} completed={counts['completed']} planned={counts['planned'] if counts['planned'] is not None else 'unknown'} incomplete={counts['incomplete']}")
    for model in result["models"]:
        def formatted(name: str) -> str:
            value = model[name + "_mean"]
            return "NA" if value is None else f"{value:.2%} (N={model[name + '_valid_papers']})"
        density = model["hallucination_event_density_mean"]
        print(f"  {model['model']}:")
        print(f"    1. Knowledge coverage: {formatted('global_kc_coverage')}")
        print(f"    2. Challenge-induced hallucination: {formatted('challenge_failure')}")
        print(f"    3. Hallucination behavior: turns={formatted('hallucination_turn_rate')}; event density={density if density is not None else 'NA'} events/turn")
        print(f"    4. Error repair: event closure={formatted('hallucination_event_closure')}; omission completion={formatted('detail_completion')}")
    classification = result["classification"]
    print(f"[classification] events={classification['event_count']} three-fields-present={classification['three_fields_present_count']} non-repeating-candidates={classification['non_repeating_path_count']} (not automatic semantic-level validation)")
    print(f"[statistics] CSV/JSON: {result['output_dir']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, required=True, help="An evaluation directory or one complete testing/measurement session.")
    parser.add_argument("--output-dir", type=Path, help="Default: <result-root>/statistics")
    args = parser.parse_args(argv)
    try:
        result = summarize(args.result_root, args.output_dir)
        print_summary(result)
        return 0 if result["status"] == "completed" else 2
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"Statistics error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
