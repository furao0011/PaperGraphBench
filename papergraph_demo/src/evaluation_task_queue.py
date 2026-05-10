from __future__ import annotations

import os
import re
from typing import Iterable


TASK_TYPE_HALLUCINATION_REPAIR = "HALLUCINATION_REPAIR"
TASK_TYPE_DETAIL_COMPLETION = "DETAIL_COMPLETION"
TASK_TYPE_THREAD_REASONING = "THREAD_REASONING"
TASK_TYPE_CHALLENGE_EVALUATION = "CHALLENGE_EVALUATION"
TASK_TYPE_REVIEW = "REVIEW"

TASK_STATUS_PENDING = "pending"
TASK_STATUS_RUNNING = "running"
TASK_STATUS_COMPLETED = "completed"
TASK_STATUS_EXHAUSTED = "exhausted"
TASK_STATUS_CANCELLED = "cancelled"

TASK_PRIORITIES = {
    TASK_TYPE_HALLUCINATION_REPAIR: 1,
    TASK_TYPE_DETAIL_COMPLETION: 2,
    TASK_TYPE_THREAD_REASONING: 3,
    TASK_TYPE_CHALLENGE_EVALUATION: 4,
    TASK_TYPE_REVIEW: 5,
}


def ensure_task_state(eval_state: dict, graph: dict | None = None) -> None:
    eval_state.setdefault("stage_tasks", {})
    eval_state.setdefault("macro_stage_status", {})
    eval_state.setdefault("global_state", {})
    eval_state["global_state"].setdefault("stage_task_count", 0)
    eval_state["global_state"].setdefault("stage_task_completed_count", 0)
    eval_state["global_state"].setdefault("stage_task_exhausted_count", 0)

    for macro in (graph or {}).get("macro_nodes", []):
        macro_id = macro.get("macro_id")
        if not macro_id:
            continue
        ensure_macro_stage_status(eval_state, macro_id)


def ensure_macro_stage_status(eval_state: dict, macro_id: str) -> dict:
    status = eval_state.setdefault("macro_stage_status", {}).setdefault(macro_id, {})
    status.setdefault("status", "not_started")
    status.setdefault("main_done", False)
    status.setdefault("repair_done", False)
    status.setdefault("thread_done", False)
    status.setdefault("challenge_done", False)
    status.setdefault("pending_task_ids", [])
    status.setdefault("running_task_ids", [])
    status.setdefault("completed_task_ids", [])
    status.setdefault("exhausted_task_ids", [])
    status.setdefault("cancelled_task_ids", [])
    return status


def build_repair_tasks_from_judge_result(
    judge_result: dict,
    turn: dict,
    next_action: str | None = None,
) -> list[dict]:
    if not turn.get("turn_id"):
        raise ValueError("Stage task construction requires turn.turn_id.")
    if not turn.get("question_id"):
        raise ValueError("Stage task construction requires turn.question_id.")

    action = next_action or judge_result.get("policy_next_action") or judge_result.get("next_action")
    tasks: list[dict] = []
    repair_context = turn.get("repair_context") or {}
    in_hallucination_repair = repair_context.get("repair_type") == "hallucination"
    has_hallucination = (
        bool(judge_result.get("hallucination_events"))
        or (not in_hallucination_repair and (action == "hallucination_followup" or _is_hallucination_state(judge_result.get("state"))))
    )
    if has_hallucination:
        tasks.append(
            _make_task(
                task_type=TASK_TYPE_HALLUCINATION_REPAIR,
                turn=turn,
                target_kc_ids=_target_kc_ids_for_hallucination(judge_result, turn),
                hallucination_event_ids=_hallucination_event_ids(judge_result),
                repair_context=_hallucination_repair_context(judge_result, turn),
                max_turns=_env_int("MAX_HALLUCINATION_FOLLOWUPS_PER_EVENT", 3),
                ordinal=len(tasks) + 1,
                source_action=action,
            )
        )
    if judge_result.get("missing_kc_ids"):
        tasks.append(
            _make_task(
                task_type=TASK_TYPE_DETAIL_COMPLETION,
                turn=turn,
                target_kc_ids=list(judge_result.get("missing_kc_ids", [])),
                hallucination_event_ids=[],
                repair_context=_detail_repair_context(judge_result, turn),
                max_turns=_env_int("MAX_DETAIL_FOLLOWUPS_PER_TASK", 3),
                ordinal=len(tasks) + 1,
                source_action=action,
            )
        )
    return sort_stage_tasks(tasks)


def attach_recommended_stage_tasks(judge_result: dict, turn: dict, next_action: str | None = None) -> list[dict]:
    tasks = build_repair_tasks_from_judge_result(judge_result, turn, next_action)
    judge_result["recommended_stage_tasks"] = tasks
    judge_result["recommended_tasks"] = tasks
    return tasks


def enqueue_stage_tasks(eval_state: dict, tasks: Iterable[dict]) -> list[str]:
    ensure_task_state(eval_state)
    added: list[str] = []
    for task in sort_stage_tasks(list(tasks)):
        task_id = task["task_id"]
        if task_id in eval_state["stage_tasks"]:
            continue
        eval_state["stage_tasks"][task_id] = dict(task)
        added.append(task_id)
        eval_state["global_state"]["stage_task_count"] = eval_state["global_state"].get("stage_task_count", 0) + 1
        macro_id = task.get("macro_id")
        if macro_id:
            macro_status = ensure_macro_stage_status(eval_state, macro_id)
            _move_task_between_lists(macro_status, task_id, task["status"])
            macro_status["status"] = "in_progress"
    return added


def next_pending_stage_task(
    eval_state: dict,
    macro_id: str | None = None,
    task_types: set[str] | None = None,
) -> dict | None:
    ensure_task_state(eval_state)
    tasks = [
        task
        for task in eval_state.get("stage_tasks", {}).values()
        if task.get("status") == TASK_STATUS_PENDING
        and (macro_id is None or task.get("macro_id") == macro_id)
        and (task_types is None or task.get("task_type") in task_types)
    ]
    ordered = sort_stage_tasks(tasks)
    return ordered[0] if ordered else None


def mark_stage_task_running(eval_state: dict, task_id: str) -> dict:
    return update_stage_task_status(eval_state, task_id, TASK_STATUS_RUNNING)


def mark_stage_task_completed(eval_state: dict, task_id: str) -> dict:
    previous = eval_state.get("stage_tasks", {}).get(task_id, {}).get("status")
    task = update_stage_task_status(eval_state, task_id, TASK_STATUS_COMPLETED)
    if previous != TASK_STATUS_COMPLETED:
        eval_state["global_state"]["stage_task_completed_count"] = (
            eval_state["global_state"].get("stage_task_completed_count", 0) + 1
        )
    return task


def mark_stage_task_exhausted(eval_state: dict, task_id: str) -> dict:
    previous = eval_state.get("stage_tasks", {}).get(task_id, {}).get("status")
    task = update_stage_task_status(eval_state, task_id, TASK_STATUS_EXHAUSTED)
    if previous != TASK_STATUS_EXHAUSTED:
        eval_state["global_state"]["stage_task_exhausted_count"] = (
            eval_state["global_state"].get("stage_task_exhausted_count", 0) + 1
        )
    return task


def update_stage_task_status(eval_state: dict, task_id: str, status: str) -> dict:
    ensure_task_state(eval_state)
    if status not in {
        TASK_STATUS_PENDING,
        TASK_STATUS_RUNNING,
        TASK_STATUS_COMPLETED,
        TASK_STATUS_EXHAUSTED,
        TASK_STATUS_CANCELLED,
    }:
        raise ValueError(f"Unknown stage task status: {status}")
    task = eval_state.get("stage_tasks", {}).get(task_id)
    if not task:
        raise KeyError(f"Stage task not found: {task_id}")
    task["status"] = status
    macro_id = task.get("macro_id")
    if macro_id:
        macro_status = ensure_macro_stage_status(eval_state, macro_id)
        _move_task_between_lists(macro_status, task_id, status)
    return task


def record_stage_task_turn(eval_state: dict, task_id: str, turn_id: str) -> dict:
    ensure_task_state(eval_state)
    task = eval_state.get("stage_tasks", {}).get(task_id)
    if not task:
        raise KeyError(f"Stage task not found: {task_id}")
    if not turn_id:
        raise ValueError("record_stage_task_turn requires a non-empty turn_id.")
    if turn_id not in task.setdefault("followup_turns", []):
        task["followup_turns"].append(turn_id)
        task["current_turns"] = len(task["followup_turns"])
    return task


def stage_task_has_budget(task: dict) -> bool:
    return int(task.get("current_turns", 0)) < int(task.get("max_turns", 0))


def sort_stage_tasks(tasks: Iterable[dict]) -> list[dict]:
    return sorted(
        tasks,
        key=lambda task: (
            int(task.get("priority", 99)),
            int(task.get("created_at_turn_no", 0)),
            str(task.get("task_id", "")),
        ),
    )


def _make_task(
    task_type: str,
    turn: dict,
    target_kc_ids: list[str],
    hallucination_event_ids: list[str],
    repair_context: dict,
    max_turns: int,
    ordinal: int,
    source_action: str | None,
) -> dict:
    turn_id = str(turn["turn_id"])
    macro_id = turn.get("macro_id")
    task_id = "TASK_{macro}_{turn}_{kind}_{ordinal}".format(
        macro=_safe_id(macro_id or "GLOBAL"),
        turn=_safe_id(turn_id),
        kind=_safe_id(task_type),
        ordinal=ordinal,
    )
    return {
        "task_id": task_id,
        "macro_id": macro_id,
        "task_type": task_type,
        "source_turn_id": turn_id,
        "source_question_id": turn["question_id"],
        "source_question_type": turn.get("question_type"),
        "source_action": source_action,
        "target_kc_ids": target_kc_ids,
        "hallucination_event_ids": hallucination_event_ids,
        "repair_context": repair_context,
        "priority": TASK_PRIORITIES[task_type],
        "max_turns": max_turns,
        "current_turns": 0,
        "status": TASK_STATUS_PENDING,
        "followup_turns": [],
        "created_at_turn_no": _turn_no(turn_id),
    }


def _target_kc_ids_for_hallucination(judge_result: dict, turn: dict) -> list[str]:
    ids = list(judge_result.get("covered_kc_ids", [])) + list(judge_result.get("missing_kc_ids", []))
    if not ids:
        ids = list(turn.get("target_kc_ids", []))
    return list(dict.fromkeys(ids))


def _hallucination_repair_context(judge_result: dict, turn: dict) -> dict:
    events = [dict(event) for event in judge_result.get("hallucination_events", []) or []]
    return {
        "repair_type": "hallucination",
        "root_turn_id": turn.get("turn_id"),
        "root_question_id": turn.get("question_id"),
        "root_question_type": turn.get("question_type"),
        "active_hallucinations": [
            {
                "event_id": event.get("event_id"),
                "hallucination_type": event.get("hallucination_type"),
                "subtype": event.get("subtype"),
                "claim": event.get("claim"),
                "related_kc_ids": event.get("related_kc_ids", []),
                "matched_forbidden_claims": event.get("matched_forbidden_claims", []),
            }
            for event in events
        ],
        "attempted_turn_ids": [],
    }


def _detail_repair_context(judge_result: dict, turn: dict) -> dict:
    missing = list(dict.fromkeys(judge_result.get("missing_kc_ids", []) or []))
    return {
        "repair_type": "detail",
        "root_turn_id": turn.get("turn_id"),
        "root_question_id": turn.get("question_id"),
        "root_question_type": turn.get("question_type"),
        "remaining_kc_ids": missing,
        "covered_during_repair": [],
    }


def _hallucination_event_ids(judge_result: dict) -> list[str]:
    ids = list(judge_result.get("hallucination_event_ids", []) or [])
    ids.extend(
        event.get("event_id")
        for event in judge_result.get("hallucination_events", []) or []
        if event.get("event_id")
    )
    return list(dict.fromkeys(ids))


def _is_hallucination_state(state: str | None) -> bool:
    return state in {"HALLUCINATION", "MISLED", "GLOBAL_OVERCLAIM", "REFUSE_TO_CORRECT"}


def _move_task_between_lists(macro_status: dict, task_id: str, status: str) -> None:
    for key in (
        "pending_task_ids",
        "running_task_ids",
        "completed_task_ids",
        "exhausted_task_ids",
        "cancelled_task_ids",
    ):
        if task_id in macro_status.setdefault(key, []):
            macro_status[key].remove(task_id)
    key = f"{status}_task_ids"
    if key in macro_status and task_id not in macro_status[key]:
        macro_status[key].append(task_id)


def _safe_id(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_]+", "_", value.strip())
    return safe.strip("_") or "NA"


def _turn_no(turn_id: str) -> int:
    match = re.search(r"(\d+)$", turn_id)
    return int(match.group(1)) if match else 0


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    value = int(raw)
    if value < 1:
        raise ValueError(f"{name} must be >= 1.")
    return value
