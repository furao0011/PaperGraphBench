"""Validate final event stores and summarize recorded classification labels.

Nonempty or distinct strings are observable candidates, not proof of a semantic
three-level ontology. Missing labels remain missing and are never reconstructed.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from statistics_inputs import read_json
from statistics_metrics import count


_LABELS = ("hallucination_family", "hallucination_type", "subtype")
EVENT_COLUMNS = (
    "round", "model", "paper_id", "event_id", "created_at_turn", "source_question_id",
    "family", "type", "subtype", "three_fields_present", "non_repeating_path", "issues",
    "related_kc_ids", "claim", "status", "followup_turns", "resolved_at_turn", "source", "state_path",
)
_STATE_LABELS = {
    "hallucination", "misled", "global_overclaim", "refuse_to_correct", "incomplete",
    "challenge_resisted", "main_progress", "resolved", "unresolved", "exhausted",
    "regressed", "pending", "completed", "corrected", "lit", "not_tested",
}


def _identity(value: dict, row: dict, path: Path) -> None:
    if value.get("paper_id") != row["paper_id"] or value.get("target_model") != row["model"]:
        raise ValueError(f"Classification identity mismatch: {path}")


def _label(value, field: str, path: Path) -> str | None:
    if value is not None and not isinstance(value, str):
        raise ValueError(f"{field} must be a string or null: {path}")
    return value


def _strings(value, field: str, path: Path) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item.strip() for item in value):
        raise ValueError(f"{field} must be a list of nonempty strings: {path}")
    return value


def _distribution(value, expected: Counter, field: str, path: Path) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"Missing classification distribution {field}: {path}")
    actual = {key: count(number, f"{field}.{key}") for key, number in value.items()}
    if actual != dict(expected):
        raise ValueError(f"Classification {field} disagrees with final events: {path}")


def _event_row(event: dict, event_id: str, row: dict, turns: dict, state_path: Path, root: Path) -> dict:
    if event.get("event_id") != event_id:
        raise ValueError(f"Classification event ID/key mismatch: {state_path}")
    created = event.get("created_at_turn")
    if not isinstance(created, str) or created not in turns:
        raise ValueError(f"Classification source turn not found: {event_id} in {state_path}")
    source_question = event.get("source_question_id")
    if source_question is not None and source_question != turns[created].get("question_id"):
        raise ValueError(f"Classification source question mismatch: {event_id} in {state_path}")
    related = _strings(event.get("related_kc_ids"), "related_kc_ids", state_path)
    labels = [_label(event.get(name), name, state_path) for name in _LABELS]
    normalized = [label.strip().casefold() if label is not None else "" for label in labels]
    issues = [f"missing_{name}" for name, value in zip(("family", "type", "subtype"), normalized) if not value]
    present = [value for value in normalized if value]
    if len(present) != len(set(present)):
        issues.append("duplicate_labels")
    for name, value in zip(("family", "type", "subtype"), normalized):
        if value.replace("-", "_").replace(" ", "_") in _STATE_LABELS:
            issues.append("state_only_" + name)
    if not related:
        issues.append("empty_related_kc_ids")
    if source_question is None:
        issues.append("missing_source_question_id")
    followups = event.get("followup_turns")
    if followups is None:
        issues.append("missing_followup_turns")
    elif any(turn not in turns for turn in _strings(followups, "followup_turns", state_path)):
        raise ValueError(f"Classification follow-up turn not found: {event_id} in {state_path}")
    resolved = event.get("resolved_at_turn")
    if resolved is not None and (not isinstance(resolved, str) or resolved not in turns):
        raise ValueError(f"Classification resolved turn not found: {event_id} in {state_path}")
    return {**{key: row[key] for key in ("round", "model", "paper_id")},
            "event_id": event_id, "created_at_turn": created, "source_question_id": source_question,
            "family": labels[0], "type": labels[1], "subtype": labels[2],
            "three_fields_present": len(present) == 3,
            "non_repeating_path": len(present) == 3 and len(set(present)) == 3,
            "issues": issues, "related_kc_ids": related, "claim": event.get("claim"),
            "status": _label(event.get("status"), "status", state_path),
            "followup_turns": followups, "resolved_at_turn": resolved, "source": event.get("source"),
            "state_path": state_path.relative_to(root).as_posix()}


def _task_events(root: Path, row: dict) -> list[dict]:
    report_path = (root / row["report_path"]).resolve()
    if not report_path.is_relative_to(root):
        raise ValueError("Classification report path escapes its result directory")
    state_path = report_path.parent / "cache" / "eval_state_graph.json"
    if not state_path.is_file():
        raise ValueError(f"Missing final classification state: {state_path}")
    report, state = read_json(report_path), read_json(state_path)
    trajectory_path = report_path.parent / "dialogue_trajectory.json"
    trajectory = read_json(trajectory_path)
    for value, path in ((report, report_path), (state, state_path), (trajectory, trajectory_path)):
        _identity(value, row, path)
    raw_turns = trajectory.get("turns")
    if not isinstance(raw_turns, list) or not raw_turns:
        raise ValueError(f"Classification trajectory has no turns: {trajectory_path}")
    turns = {}
    for turn in raw_turns:
        turn_id = turn.get("turn_id") if isinstance(turn, dict) else None
        if not isinstance(turn_id, str) or not turn_id.strip() or turn_id in turns:
            raise ValueError(f"Classification trajectory has invalid/duplicate turn IDs: {trajectory_path}")
        turns[turn_id] = turn
    events = state.get("hallucination_events")
    if not isinstance(events, dict):
        raise ValueError(f"Final hallucination_events must be an event dictionary: {state_path}")
    metrics = report.get("hallucination_metrics")
    if not isinstance(metrics, dict):
        raise ValueError(f"Missing hallucination_metrics: {report_path}")
    if count(metrics.get("hallucination_event_count"), "hallucination_event_count") != len(events):
        raise ValueError(f"Classification event count mismatch: {state_path}")
    records = []
    for event_id, event in sorted(events.items()):
        if not isinstance(event_id, str) or not event_id.strip() or not isinstance(event, dict):
            raise ValueError(f"Invalid classification event entry: {state_path}")
        records.append(_event_row(event, event_id, row, turns, state_path, root))
    _distribution(metrics.get("hallucination_by_type"), Counter(event["type"] or "unknown" for event in records), "hallucination_by_type", report_path)
    _distribution(metrics.get("hallucination_by_status"), Counter(event["status"] or "unknown" for event in records), "hallucination_by_status", report_path)
    return records


def _summary(events: list[dict], rows: list[dict]) -> dict:
    completed = sum(row["status"] == "completed" for row in rows)
    present = sum(event["three_fields_present"] for event in events)
    paths = Counter((event["family"], event["type"], event["subtype"]) for event in events)
    ordered_paths = sorted(paths, key=lambda path: tuple((0, "") if value is None else (1, value) for value in path))
    return {"status": "completed" if completed == len(rows) else "incomplete",
            "task_counts": {"total": len(rows), "completed": completed, "incomplete": len(rows) - completed},
            "event_count": len(events), "three_fields_present_count": present,
            "three_fields_present_rate": present / len(events) if events else None,
            "non_repeating_path_count": sum(event["non_repeating_path"] for event in events),
            "warning_counts": dict(sorted(Counter(issue for event in events for issue in event["issues"]).items())),
            "classification_counts": [{"family": path[0], "type": path[1], "subtype": path[2], "count": paths[path]} for path in ordered_paths]}


def collect_classification(root: Path, rows: list[dict]) -> dict:
    """Validate completed tasks, preserve scoped events, and flag label limitations."""
    root = Path(root).expanduser().resolve()
    if not rows:
        raise ValueError("Classification statistics require task rows.")
    events, seen = [], set()
    ordered_rows = sorted(rows, key=lambda row: (row["round"], row["model"], row["paper_id"]))
    for row in ordered_rows:
        key = tuple(row[name] for name in ("round", "model", "paper_id"))
        if key in seen:
            raise ValueError(f"Duplicate classification task: {key}")
        seen.add(key)
        if row["status"] == "completed":
            events.extend(_task_events(root, row))
    summary = _summary(events, ordered_rows)
    summary["models"] = [dict(model=model,
        **_summary([event for event in events if event["model"] == model],
                   [row for row in ordered_rows if row["model"] == model]))
        for model in sorted({row["model"] for row in ordered_rows})]
    summary["interpretation"] = "Nonempty/distinct label strings are candidate paths, not proof of three semantic levels; missing labels are not filled."
    return {"events": events, "summary": summary}
