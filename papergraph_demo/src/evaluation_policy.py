from __future__ import annotations

import os

from src.evaluation_task_queue import attach_recommended_stage_tasks
from src.policy_controller import choose_next_action


def is_immediate_followup(action: str | None) -> bool:
    return action in {
        "detail_followup",
        "hallucination_followup",
    }


def bounded_env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    raw = os.getenv(name)
    try:
        value = int(raw) if raw is not None and raw.strip() else default
    except ValueError:
        value = default
    return max(minimum, min(maximum, value))


def review_target_at_end() -> int:
    return bounded_env_int("EVAL_REVIEW_AT_END", 2, 2, 3)


def apply_effective_next_action(eval_state: dict, judge_result: dict, turn: dict) -> str:
    next_action = choose_next_action_for_turn(eval_state, judge_result, turn)
    judge_result["next_action"] = next_action
    judge_result["policy_next_action"] = next_action
    attach_recommended_stage_tasks(judge_result, turn, next_action)
    return next_action


def choose_next_action_for_turn(eval_state: dict, judge_result: dict, turn: dict) -> str:
    base_action = choose_next_action(eval_state, judge_result)
    if base_action in {"hallucination_followup", "detail_followup", "end_failed"}:
        return base_action
    return base_action
