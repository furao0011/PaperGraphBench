from __future__ import annotations

import os
import re

from src.evaluation_task_queue import (
    TASK_PRIORITIES,
    TASK_STATUS_PENDING,
    TASK_TYPE_CHALLENGE_EVALUATION,
    TASK_TYPE_DETAIL_COMPLETION,
    TASK_TYPE_HALLUCINATION_REPAIR,
    TASK_TYPE_REVIEW,
    TASK_TYPE_THREAD_REASONING,
    enqueue_stage_tasks,
)


def enqueue_anchor_task(
    eval_state: dict,
    task_type: str,
    task_key: str | None,
    macro_id: str | None,
    question_id: str,
    target_kc_ids: list[str],
    created_at_turn_no: int,
) -> str:
    task_id = f"TASK_{_safe_task_id(macro_id or 'GLOBAL')}_{_safe_task_id(task_type)}_{_safe_task_id(task_key or question_id)}"
    enqueue_stage_tasks(
        eval_state,
        [
            {
                "task_id": task_id,
                "macro_id": macro_id,
                "task_type": task_type,
                "source_turn_id": None,
                "source_question_id": question_id,
                "source_question_type": question_type_for_anchor(task_type),
                "source_action": None,
                "target_kc_ids": list(target_kc_ids),
                "hallucination_event_ids": [],
                "priority": TASK_PRIORITIES[task_type],
                "max_turns": 1,
                "current_turns": 0,
                "status": TASK_STATUS_PENDING,
                "followup_turns": [],
                "created_at_turn_no": created_at_turn_no,
            }
        ],
    )
    return task_id


def attach_coverage_gap_ids(tasks: list[dict], turn: dict) -> None:
    gap_id = turn.get("state_update", {}).get("structured_update", {}).get("coverage_gap_id")
    if not gap_id:
        return
    for task in tasks:
        if task.get("task_type") == TASK_TYPE_DETAIL_COMPLETION:
            task["coverage_gap_id"] = gap_id


def action_for_repair_task(task: dict) -> str:
    task_type = task.get("task_type")
    if task_type == TASK_TYPE_HALLUCINATION_REPAIR:
        return "hallucination_followup"
    if task_type == TASK_TYPE_DETAIL_COMPLETION:
        return "detail_followup"
    raise ValueError(f"Unsupported repair task type: {task_type}")


def repair_task_resolved(task: dict, judge_result: dict) -> bool:
    task_type = task.get("task_type")
    if task_type == TASK_TYPE_DETAIL_COMPLETION:
        missing = set(judge_result.get("missing_kc_ids", []))
        return not any(kc_id in missing for kc_id in task.get("target_kc_ids", []))
    if task_type == TASK_TYPE_HALLUCINATION_REPAIR:
        if judge_result.get("hallucination_events"):
            return True
        next_action = judge_result.get("policy_next_action") or judge_result.get("next_action")
        if next_action == "hallucination_followup":
            return False
        return judge_result.get("state") not in {"HALLUCINATION", "MISLED", "GLOBAL_OVERCLAIM", "REFUSE_TO_CORRECT"}
    return True


def review_allows_hallucination_followup() -> bool:
    return env_bool("REVIEW_ALLOW_HALLUCINATION_FOLLOWUP", True)


def env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def question_type_for_anchor(task_type: str) -> str:
    if task_type == TASK_TYPE_THREAD_REASONING:
        return "thread_question"
    if task_type == TASK_TYPE_CHALLENGE_EVALUATION:
        return "challenge_question"
    if task_type == TASK_TYPE_REVIEW:
        return "review_followup"
    return "unknown"


def _safe_task_id(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_]+", "_", str(value).strip())
    return safe.strip("_") or "NA"
