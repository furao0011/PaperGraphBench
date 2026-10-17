"""Discover and strictly validate report/trajectory pairs without loading APIs."""

from __future__ import annotations

import json
import re
from pathlib import Path

from src.artifact_layout import safe_path_component
from src.pipeline_artifacts import completed_evaluation
from statistics_metrics import extract_metrics


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def planned_tasks(root: Path, requested: list[dict] | None) -> list[dict] | None:
    if requested is None:
        manifest = next((root / name for name in ("testing.json", "measurement.json") if (root / name).is_file()), None)
        if manifest:
            data = read_json(manifest)
            if "expected_tasks" in data:
                requested = data["expected_tasks"]
            else:
                papers, models, repeats = data.get("paper_ids"), data.get("target_models"), data.get("repeats")
                if not isinstance(papers, list) or not papers or not isinstance(models, list) or not models or not isinstance(repeats, int) or repeats < 1:
                    raise ValueError(f"Manifest does not specify a complete task plan: {manifest}")
                requested = [{"round": f"round-{index:02d}", "paper_id": paper, "model": model}
                             for index in range(1, repeats + 1) for paper in papers for model in models]
    if requested is None:
        return None
    if not isinstance(requested, list) or not requested:
        raise ValueError("expected_tasks must be a non-empty list")
    result, seen, destinations = [], set(), set()
    for task in requested:
        if not isinstance(task, dict) or any(not isinstance(task.get(key), str) or not task[key].strip() for key in ("round", "paper_id", "model")):
            raise ValueError("Each expected task requires round, paper_id and model strings")
        if task["round"] != "single" and not re.fullmatch(r"round-\d+", task["round"]):
            raise ValueError(f"Invalid task round: {task['round']}")
        key = tuple(task[name] for name in ("round", "paper_id", "model"))
        destination = report_path(root, task)
        if key in seen or destination in destinations:
            raise ValueError(f"Duplicate or colliding expected task: {key}")
        seen.add(key)
        destinations.add(destination)
        result.append({name: task[name] for name in ("round", "paper_id", "model")})
    return result


def report_path(root: Path, task: dict) -> Path:
    parent = root if task["round"] == "single" else root / task["round"] / "evaluation"
    return parent / safe_path_component(task["model"], "model") / safe_path_component(task["paper_id"], "paper_id") / "evaluation_report.json"


def _read_row(root: Path, task: dict, path: Path) -> dict:
    row = {**task, "status": "missing", "report_path": path.relative_to(root).as_posix(), "reason": "Report not found"}
    if not path.is_file():
        return row
    report = read_json(path)
    if report.get("paper_id") != task["paper_id"] or report.get("target_model") != task["model"]:
        raise ValueError(f"Report identity disagrees with task: {path}")
    summary = report.get("summary")
    if not isinstance(summary, dict):
        raise ValueError(f"Report has no summary object: {path}")
    if summary.get("evaluation_status") == "completed" and summary.get("failed") is False:
        if not completed_evaluation(path, task["paper_id"], task["model"]):
            row.update(status="incomplete", reason="Completed report is missing its trajectory")
            return row
        row.update(extract_metrics(report), status="completed", reason="")
    else:
        if summary.get("evaluation_status") == "completed":
            raise ValueError(f"Completed report has inconsistent failure status: {path}")
        row.update(status="incomplete", reason=str(summary.get("failure_reason") or summary.get("evaluation_status") or "Unknown evaluation status"))
    return row


def collect_rows(root: Path, expected: list[dict] | None) -> list[dict]:
    discovered = sorted(root.rglob("evaluation_report.json"))
    if expected is not None:
        planned_paths = {report_path(root, task) for task in expected}
        unexpected = set(discovered) - planned_paths
        if unexpected:
            raise ValueError("Unexpected report outside the task plan: " + str(sorted(unexpected)[0]))
        return [_read_row(root, task, report_path(root, task)) for task in expected]
    if not discovered:
        raise ValueError(f"No evaluation reports found in {root}")
    rows, seen = [], set()
    for path in discovered:
        report = read_json(path)
        task = {"round": "single", "paper_id": report.get("paper_id"), "model": report.get("target_model")}
        rounds = [part for part in path.relative_to(root).parts[:-3] if re.fullmatch(r"round-\d+", part)]
        if rounds:
            task["round"] = rounds[-1]
        if any(not isinstance(value, str) or not value for value in task.values()):
            raise ValueError(f"Invalid report identity: {path}")
        key = tuple(task[name] for name in ("round", "paper_id", "model"))
        if key in seen:
            raise ValueError(f"Duplicate report task: {key}")
        seen.add(key)
        if path.parent.name != safe_path_component(task["paper_id"], "paper_id") or path.parent.parent.name != safe_path_component(task["model"], "model"):
            raise ValueError(f"Report identity disagrees with directory names: {path}")
        rows.append(_read_row(root, task, path))
    return rows
