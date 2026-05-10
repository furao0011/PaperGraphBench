from __future__ import annotations

from src.evaluation_hallucination_state import ensure_hallucination_state
from src.evaluation_task_queue import ensure_task_state
from src.thread_scheduler import ensure_thread_states


def ensure_eval_state_defaults(eval_state: dict, graph: dict) -> None:
    eval_state.setdefault("macro_states", {})
    default_active_ids = set()
    for macro in graph.get("macro_nodes", []):
        kc_ids = list(macro.get("kc_ids", []))
        active_count = int(macro.get("active_kc_count") or min(3, len(kc_ids)) or 0)
        default_active_ids.update(kc_ids[:active_count])
    for kc_id, state in eval_state.get("kc_states", {}).items():
        state.setdefault("globally_supported_by_turns", [])
        state.setdefault("is_active_target", kc_id in default_active_ids)
    for macro in graph.get("macro_nodes", []):
        macro_id = macro.get("macro_id")
        if not macro_id:
            continue
        eval_state["macro_states"].setdefault(macro_id, {})
        eval_state["macro_states"][macro_id].setdefault("status", "not_started")
        eval_state["macro_states"][macro_id].setdefault("main_question_asked", False)
        eval_state["macro_states"][macro_id].setdefault("covered_kc_ids", [])
        eval_state["macro_states"][macro_id].setdefault("missing_kc_ids", [])
        eval_state["macro_states"][macro_id].setdefault(
            "target_kc_ids",
            list(macro.get("kc_ids", []))[: int(macro.get("active_kc_count") or min(3, len(macro.get("kc_ids", []))) or 0)],
        )
        eval_state["macro_states"][macro_id].setdefault("bank_kc_ids", list(macro.get("kc_ids", [])))
        eval_state["macro_states"][macro_id].setdefault("related_turns", [])
        eval_state["macro_states"][macro_id].setdefault(
            "bank_kc_count",
            macro.get("bank_kc_count", len(macro.get("kc_ids", []))),
        )
        eval_state["macro_states"][macro_id].setdefault(
            "active_kc_count",
            macro.get("active_kc_count", len(eval_state["macro_states"][macro_id].get("target_kc_ids", []))),
        )
    ensure_thread_states(eval_state, graph.get("reasoning_threads", []))
    ensure_task_state(eval_state, graph)
    ensure_hallucination_state(eval_state)
    eval_state.setdefault("claim_verification_states", {})
    eval_state.setdefault("global_state", {})
    eval_state["global_state"].setdefault("hallucination_event_count", 0)
    eval_state["global_state"].setdefault("hallucination_event_resolved_count", 0)
    eval_state["global_state"].setdefault("hallucination_event_exhausted_count", 0)
    eval_state["global_state"].setdefault("coverage_gap_count", 0)
    eval_state["global_state"].setdefault("coverage_gap_resolved_count", 0)
    eval_state["global_state"].setdefault("coverage_gap_exhausted_count", 0)
    eval_state["global_state"].setdefault("review_question_count", 0)
    eval_state["global_state"].setdefault("global_overclaim_count", 0)
    eval_state["global_state"].setdefault("global_contradicted_claim_count", 0)
    eval_state["global_state"].setdefault("not_enough_info_claim_count", 0)
    eval_state["global_state"].setdefault("thread_bridge_tested_count", 0)
    eval_state["global_state"].setdefault("thread_bridge_success_count", 0)
    eval_state["global_state"].setdefault("challenge_question_count", 0)
    eval_state["global_state"].setdefault("challenge_failure_count", 0)
    eval_state["global_state"].setdefault("challenge_resisted_count", 0)
    eval_state["global_state"].setdefault("evaluation_status", "not_started")
    eval_state["global_state"].setdefault("completion_reason", None)
    eval_state["global_state"].setdefault("completed_at_turn", None)


def rebuild_eval_turn_counts(eval_state: dict, trajectory: dict) -> None:
    review_total = 0
    for turn in trajectory.get("turns", []):
        if turn.get("question_type") == "review_followup":
            review_total += 1
    current_reviews = eval_state["global_state"].get("review_question_count", 0)
    eval_state["global_state"]["review_question_count"] = max(current_reviews, review_total)


def mark_evaluation_running(eval_state: dict) -> None:
    global_state = eval_state.setdefault("global_state", {})
    if global_state.get("evaluation_status") not in {"completed", "failed"}:
        global_state["evaluation_status"] = "running"
        global_state["completion_reason"] = None
        global_state["completed_at_turn"] = None


def mark_evaluation_finished(eval_state: dict, status: str, reason: str, turn_no: int) -> None:
    global_state = eval_state.setdefault("global_state", {})
    global_state["evaluation_status"] = status
    global_state["completion_reason"] = reason
    global_state["completed_at_turn"] = turn_no
    if status == "failed":
        global_state["failed"] = True
        global_state["failure_reason"] = global_state.get("failure_reason") or reason


def final_evaluation_status(eval_state: dict, turn_no: int, max_turns: int) -> tuple[str, str]:
    global_state = eval_state.get("global_state", {})
    if global_state.get("failed"):
        return "failed", global_state.get("failure_reason") or "failed"
    if max_turns and turn_no >= max_turns:
        return "stopped_by_max_turns", f"EVAL_MAX_TURNS reached: {max_turns}"
    return "completed", "all scheduled macro, follow-up, thread, and review turns finished"
