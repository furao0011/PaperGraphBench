from __future__ import annotations


THREAD_QUESTION_TYPES = {
    "thread_premise_question",
    "thread_evidence_question",
    "thread_bridge_question",
    "thread_review_question",
    "thread_question",
}


def ensure_thread_states(eval_state: dict, threads: list[dict]) -> None:
    states = eval_state.setdefault("thread_states", {})
    for thread in threads:
        thread_id = thread.get("thread_id")
        if not thread_id:
            continue
        state = states.setdefault(
            thread_id,
            {
                "status": "not_started",
                "completed_steps": [],
                "current_step": None,
                "success": None,
                "failure_reason": None,
                "related_turns": [],
                "bridge_success": None,
                "review_consistency": None,
            },
        )
        state.setdefault("status", "not_started")
        state.setdefault("completed_steps", [])
        state.setdefault("current_step", None)
        state.setdefault("success", None)
        state.setdefault("failure_reason", None)
        state.setdefault("related_turns", [])
        state.setdefault("bridge_success", None)
        state.setdefault("review_consistency", None)


def completed_thread_step_ids(eval_state: dict) -> set[str]:
    completed: set[str] = set()
    for state in eval_state.get("thread_states", {}).values():
        completed.update(step for step in state.get("completed_steps", []) if step)
    return completed


def get_ready_thread_turn(eval_state: dict, threads: list[dict], review_stage: bool = False) -> dict | None:
    ensure_thread_states(eval_state, threads)
    for thread in threads:
        thread_id = thread.get("thread_id")
        state = eval_state.get("thread_states", {}).get(thread_id, {})
        completed = set(state.get("completed_steps", []))
        for step in thread.get("planned_turns", []):
            step_id = step.get("thread_turn_id")
            if not step_id or step_id in completed:
                continue
            role = step.get("role")
            if role == "review_consistency":
                if review_stage and _review_step_ready(state):
                    return _make_thread_seed(thread, step)
                break
            if review_stage:
                continue
            if _step_ready(eval_state, step):
                return _make_thread_seed(thread, step)
            break
    return None


def record_thread_step_result(
    eval_state: dict,
    thread_id: str | None,
    step_id: str | None,
    turn_id: str,
    question_type: str,
    judge_result: dict,
) -> dict:
    if not thread_id or not step_id:
        return {}
    state = eval_state.setdefault("thread_states", {}).setdefault(
        thread_id,
        {
            "status": "not_started",
            "completed_steps": [],
            "current_step": None,
            "success": None,
            "failure_reason": None,
            "related_turns": [],
            "bridge_success": None,
            "review_consistency": None,
        },
    )
    completed = state.setdefault("completed_steps", [])
    if step_id not in completed:
        completed.append(step_id)
    related = state.setdefault("related_turns", [])
    if turn_id not in related:
        related.append(turn_id)

    state["current_step"] = step_id
    if state.get("status") == "not_started":
        state["status"] = "in_progress"

    result = judge_result.get("thread_step_result")
    failed = judge_result.get("state") in {"HALLUCINATION", "MISLED", "THREAD_FAIL"}
    missing = bool(judge_result.get("missing_kc_ids"))

    if question_type == "thread_bridge_question":
        success = result == "bridge_success" or (
            judge_result.get("used_previous_premise") is True and not failed and not missing
        )
        state["bridge_success"] = success
        state["status"] = "completed_success" if success else "completed_partial"
    elif question_type == "thread_review_question":
        consistent = result == "review_consistent" or (
            not judge_result.get("contradicted_previous_turns") and not failed
        )
        state["review_consistency"] = consistent
        state["status"] = "reviewed_consistent" if consistent else "reviewed_inconsistent"
        state["success"] = bool(state.get("bridge_success")) and consistent
    elif failed:
        state["status"] = "completed_fail"
        state["failure_reason"] = judge_result.get("judge_explanation") or judge_result.get("state")
    elif missing:
        state["status"] = "in_progress"
    else:
        state["status"] = "bridge_ready" if _thread_bridge_ready(eval_state, thread_id) else "in_progress"

    return {
        "thread_id": thread_id,
        "thread_step_id": step_id,
        "thread_status": state.get("status"),
        "bridge_success": state.get("bridge_success"),
        "review_consistency": state.get("review_consistency"),
    }


def _step_ready(eval_state: dict, step: dict) -> bool:
    condition = step.get("trigger_condition", {}) or {}
    macro_id = condition.get("macro_reached")
    if macro_id and not _macro_reached(eval_state, macro_id):
        return False
    required = condition.get("required_lit_kc_ids") or condition.get("required_lit_kc") or []
    if required and not _all_kcs_lit(eval_state, required):
        return False
    return True


def _macro_reached(eval_state: dict, macro_id: str) -> bool:
    state = eval_state.get("macro_states", {}).get(macro_id, {})
    return bool(state.get("main_question_asked") or state.get("related_turns"))


def _all_kcs_lit(eval_state: dict, kc_ids: list[str]) -> bool:
    states = eval_state.get("kc_states", {})
    return all(states.get(kc_id, {}).get("status") in {"lit", "corrected"} for kc_id in kc_ids)


def _review_step_ready(thread_state: dict) -> bool:
    completed = thread_state.get("completed_steps", [])
    return bool(completed) and thread_state.get("review_consistency") is None


def _thread_bridge_ready(eval_state: dict, thread_id: str) -> bool:
    state = eval_state.get("thread_states", {}).get(thread_id, {})
    return state.get("bridge_success") is None and len(state.get("completed_steps", [])) >= 2


def _make_thread_seed(thread: dict, step: dict) -> dict:
    role = step.get("role")
    qtype = {
        "establish_premise": "thread_premise_question",
        "establish_evidence": "thread_evidence_question",
        "bridge_reasoning": "thread_bridge_question",
        "review_consistency": "thread_review_question",
    }.get(role, "thread_question")
    return {
        "question_id": f"Q_{step.get('thread_turn_id')}",
        "question_type": qtype,
        "thread_id": thread.get("thread_id"),
        "thread_turn_id": step.get("thread_turn_id"),
        "thread_role": role,
        "thread_type": thread.get("thread_type"),
        "preferred_macro_id": step.get("preferred_macro_id"),
        "macro_id": step.get("preferred_macro_id"),
        "target_kc_ids": step.get("target_kc_ids", []),
        "question_goal": step.get("question_goal", ""),
        "trigger_condition": step.get("trigger_condition", {}),
        "success_criteria": thread.get("success_criteria", []),
        "requires_runtime_generation": True,
    }
