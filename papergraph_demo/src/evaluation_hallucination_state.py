from __future__ import annotations

import os


EVENT_STATUS_UNRESOLVED = "unresolved"
EVENT_STATUS_RESOLVED = "resolved"
EVENT_STATUS_EXHAUSTED = "exhausted"
EVENT_STATUS_REGRESSED = "regressed"

GAP_STATUS_PENDING = "pending"
GAP_STATUS_RESOLVED = "resolved"
GAP_STATUS_EXHAUSTED = "exhausted"


def ensure_hallucination_state(eval_state: dict) -> None:
    eval_state.setdefault("hallucination_events", {})
    eval_state.setdefault("coverage_gaps", {})
    eval_state.setdefault("global_state", {})
    eval_state["global_state"].setdefault("hallucination_event_count", 0)
    eval_state["global_state"].setdefault("hallucination_event_resolved_count", 0)
    eval_state["global_state"].setdefault("hallucination_event_exhausted_count", 0)
    eval_state["global_state"].setdefault("coverage_gap_count", 0)
    eval_state["global_state"].setdefault("coverage_gap_resolved_count", 0)
    eval_state["global_state"].setdefault("coverage_gap_exhausted_count", 0)


def record_structured_judge_state(
    eval_state: dict,
    turn_id: str,
    judge_result: dict,
    macro_id: str | None,
    question_type: str | None,
) -> dict:
    ensure_hallucination_state(eval_state)
    events = record_hallucination_events(eval_state, judge_result)
    gap = record_coverage_gap(eval_state, turn_id, judge_result, macro_id, question_type)
    return {
        "hallucination_event_ids": [event["event_id"] for event in events],
        "coverage_gap_id": gap.get("gap_id") if gap else None,
    }


def record_hallucination_events(eval_state: dict, judge_result: dict) -> list[dict]:
    ensure_hallucination_state(eval_state)
    added: list[dict] = []
    for event in judge_result.get("hallucination_events", []) or []:
        event_id = event.get("event_id")
        if not event_id:
            raise ValueError("Hallucination event requires event_id.")
        if event_id in eval_state["hallucination_events"]:
            continue
        stored = dict(event)
        stored.setdefault("status", EVENT_STATUS_UNRESOLVED)
        stored.setdefault("followup_turns", [])
        stored.setdefault("resolved_at_turn", None)
        stored.setdefault("exhausted_at_turn", None)
        eval_state["hallucination_events"][event_id] = stored
        eval_state["global_state"]["hallucination_event_count"] = (
            eval_state["global_state"].get("hallucination_event_count", 0) + 1
        )
        added.append(stored)
    return added


def record_coverage_gap(
    eval_state: dict,
    turn_id: str,
    judge_result: dict,
    macro_id: str | None,
    question_type: str | None,
) -> dict | None:
    ensure_hallucination_state(eval_state)
    missing_kc_ids = list(judge_result.get("coverage", {}).get("missing_kc_ids") or judge_result.get("missing_kc_ids", []))
    if not missing_kc_ids:
        return None
    gap_id = f"G_{turn_id}_1"
    if gap_id in eval_state["coverage_gaps"]:
        return eval_state["coverage_gaps"][gap_id]
    gap = {
        "gap_id": gap_id,
        "created_at_turn": turn_id,
        "macro_id": macro_id,
        "source_question_type": question_type,
        "missing_kc_ids": list(dict.fromkeys(missing_kc_ids)),
        "status": GAP_STATUS_PENDING,
        "followup_turns": [],
        "max_followups": _env_int("MAX_DETAIL_FOLLOWUPS_PER_TASK", 3),
        "resolved_at_turn": None,
        "exhausted_at_turn": None,
    }
    eval_state["coverage_gaps"][gap_id] = gap
    eval_state["global_state"]["coverage_gap_count"] = eval_state["global_state"].get("coverage_gap_count", 0) + 1
    return gap


def mark_hallucination_events_followed_up(eval_state: dict, event_ids: list[str], turn_id: str) -> None:
    ensure_hallucination_state(eval_state)
    for event_id in event_ids:
        event = _event(eval_state, event_id)
        if turn_id not in event.setdefault("followup_turns", []):
            event["followup_turns"].append(turn_id)


def mark_hallucination_events_resolved(eval_state: dict, event_ids: list[str], turn_id: str) -> None:
    ensure_hallucination_state(eval_state)
    for event_id in event_ids:
        event = _event(eval_state, event_id)
        previous = event.get("status")
        event["status"] = EVENT_STATUS_RESOLVED
        event["resolved_at_turn"] = turn_id
        if previous != EVENT_STATUS_RESOLVED:
            eval_state["global_state"]["hallucination_event_resolved_count"] = (
                eval_state["global_state"].get("hallucination_event_resolved_count", 0) + 1
            )


def mark_hallucination_events_exhausted(eval_state: dict, event_ids: list[str], turn_id: str | None) -> None:
    ensure_hallucination_state(eval_state)
    for event_id in event_ids:
        event = _event(eval_state, event_id)
        previous = event.get("status")
        event["status"] = EVENT_STATUS_EXHAUSTED
        event["exhausted_at_turn"] = turn_id
        if previous != EVENT_STATUS_EXHAUSTED:
            eval_state["global_state"]["hallucination_event_exhausted_count"] = (
                eval_state["global_state"].get("hallucination_event_exhausted_count", 0) + 1
            )


def mark_coverage_gap_followed_up(eval_state: dict, gap_id: str | None, turn_id: str) -> None:
    if not gap_id:
        return
    ensure_hallucination_state(eval_state)
    gap = _gap(eval_state, gap_id)
    if turn_id not in gap.setdefault("followup_turns", []):
        gap["followup_turns"].append(turn_id)


def mark_coverage_gap_resolved(eval_state: dict, gap_id: str | None, turn_id: str) -> None:
    if not gap_id:
        return
    ensure_hallucination_state(eval_state)
    gap = _gap(eval_state, gap_id)
    previous = gap.get("status")
    gap["status"] = GAP_STATUS_RESOLVED
    gap["resolved_at_turn"] = turn_id
    if previous != GAP_STATUS_RESOLVED:
        eval_state["global_state"]["coverage_gap_resolved_count"] = (
            eval_state["global_state"].get("coverage_gap_resolved_count", 0) + 1
        )


def mark_coverage_gap_exhausted(eval_state: dict, gap_id: str | None, turn_id: str | None) -> None:
    if not gap_id:
        return
    ensure_hallucination_state(eval_state)
    gap = _gap(eval_state, gap_id)
    previous = gap.get("status")
    gap["status"] = GAP_STATUS_EXHAUSTED
    gap["exhausted_at_turn"] = turn_id
    if previous != GAP_STATUS_EXHAUSTED:
        eval_state["global_state"]["coverage_gap_exhausted_count"] = (
            eval_state["global_state"].get("coverage_gap_exhausted_count", 0) + 1
        )


def no_gameover_on_unresolved_hallucination() -> bool:
    raw = os.getenv("NO_GAMEOVER_ON_UNRESOLVED_HALLUCINATION", "true")
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _event(eval_state: dict, event_id: str) -> dict:
    if event_id not in eval_state.get("hallucination_events", {}):
        raise KeyError(f"Hallucination event not found: {event_id}")
    return eval_state["hallucination_events"][event_id]


def _gap(eval_state: dict, gap_id: str) -> dict:
    if gap_id not in eval_state.get("coverage_gaps", {}):
        raise KeyError(f"Coverage gap not found: {gap_id}")
    return eval_state["coverage_gaps"][gap_id]


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    value = int(raw)
    if value < 1:
        raise ValueError(f"{name} must be >= 1.")
    return value
