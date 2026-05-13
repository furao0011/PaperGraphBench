from __future__ import annotations

from src.challenge_scheduler import record_challenge_result
from src.evaluation_hallucination_state import (
    ensure_hallucination_state,
    no_gameover_on_unresolved_hallucination,
    record_structured_judge_state,
)
from src.thread_scheduler import record_thread_step_result


def initialize_eval_state(master_graph: dict, target_model: str) -> dict:
    default_active_by_macro = _default_active_kc_ids_by_macro(master_graph)
    kc_states = {}
    for kc in master_graph.get("kc_nodes", []):
        macro_id = kc.get("macro_id")
        kc_states[kc["kc_id"]] = {
            "status": "unlit",
            "covered_by_turns": [],
            "globally_supported_by_turns": [],
            "missed_by_turns": [],
            "confidence": 0.0,
            "hallucination_history": [],
            "correction_status": "not_needed",
            "same_hallucination_followed_up_times": 0,
            "importance": kc.get("importance", "normal"),
            "macro_id": macro_id,
            "is_active_target": kc["kc_id"] in default_active_by_macro.get(macro_id, []),
        }
    path_states = {}
    for path in master_graph.get("reasoning_paths", []):
        path_states[path["path_id"]] = {
            "status": "not_tested",
            "tested_by_turn": None,
            "result": None,
            "kc_sequence": path.get("kc_sequence", []),
            "required_lit_kc": path.get("trigger_condition", {}).get("required_lit_kc", path.get("kc_sequence", [])[:2]),
        }
    return {
        "paper_id": master_graph.get("paper_id"),
        "target_model": target_model,
        "kc_states": kc_states,
        "path_states": path_states,
        "macro_states": {
            macro.get("macro_id"): {
                "status": "not_started",
                "main_question_asked": False,
                "covered_kc_ids": [],
                "missing_kc_ids": [],
                "target_kc_ids": list(default_active_by_macro.get(macro.get("macro_id"), [])),
                "bank_kc_ids": list(macro.get("kc_ids", [])),
                "related_turns": [],
                "bank_kc_count": macro.get("bank_kc_count", len(macro.get("kc_ids", []))),
                "active_kc_count": macro.get(
                    "active_kc_count",
                    len(default_active_by_macro.get(macro.get("macro_id"), [])),
                ),
            }
            for macro in master_graph.get("macro_nodes", [])
            if macro.get("macro_id")
        },
        "thread_states": {
            thread.get("thread_id"): {
                "status": "not_started",
                "completed_steps": [],
                "current_step": None,
                "success": None,
                "failure_reason": None,
                "related_turns": [],
                "bridge_success": None,
                "review_consistency": None,
                "thread_challenge_count": 0,
                "thread_challenge_question_ids": [],
                "thread_challenge_results": [],
            }
            for thread in master_graph.get("reasoning_threads", [])
            if thread.get("thread_id")
        },
        "claim_verification_states": {},
        "hallucination_events": {},
        "coverage_gaps": {},
        "stage_tasks": {},
        "macro_stage_status": {
            macro.get("macro_id"): {
                "status": "not_started",
                "main_done": False,
                "repair_done": False,
                "thread_done": False,
                "challenge_done": False,
                "pending_task_ids": [],
                "running_task_ids": [],
                "completed_task_ids": [],
                "exhausted_task_ids": [],
                "cancelled_task_ids": [],
            }
            for macro in master_graph.get("macro_nodes", [])
            if macro.get("macro_id")
        },
        "global_state": {
            "turn_count": 0,
            "hallucination_count": 0,
            "hallucination_event_count": 0,
            "hallucination_event_resolved_count": 0,
            "hallucination_event_exhausted_count": 0,
            "coverage_gap_count": 0,
            "coverage_gap_resolved_count": 0,
            "coverage_gap_exhausted_count": 0,
            "review_question_count": 0,
            "global_overclaim_count": 0,
            "global_contradicted_claim_count": 0,
            "not_enough_info_claim_count": 0,
            "thread_bridge_tested_count": 0,
            "thread_bridge_success_count": 0,
            "challenge_question_count": 0,
            "challenge_failure_count": 0,
            "challenge_resisted_count": 0,
            "stage_task_count": 0,
            "stage_task_completed_count": 0,
            "stage_task_exhausted_count": 0,
            "failed": False,
            "failure_reason": None,
            "evaluation_status": "not_started",
            "completion_reason": None,
            "completed_at_turn": None,
        },
    }


def apply_judge_result(
    eval_state: dict,
    turn_id: str,
    judge_result: dict,
    path_id: str | None = None,
    macro_id: str | None = None,
    question_type: str | None = None,
    thread_id: str | None = None,
    thread_step_id: str | None = None,
    question: dict | None = None,
) -> dict:
    ensure_hallucination_state(eval_state)
    lit = []
    missing = []
    for kc_id in judge_result.get("covered_kc_ids", []):
        if kc_id not in eval_state["kc_states"]:
            continue
        s = eval_state["kc_states"][kc_id]
        s["status"] = "lit"
        s["covered_by_turns"].append(turn_id)
        s["confidence"] = judge_result.get("confidence", 0.7)
        lit.append(kc_id)
    for kc_id in judge_result.get("missing_kc_ids", []):
        if kc_id not in eval_state["kc_states"]:
            continue
        s = eval_state["kc_states"][kc_id]
        if s["status"] == "unlit":
            s["status"] = "missing"
        s["missed_by_turns"].append(turn_id)
        missing.append(kc_id)
    if judge_result.get("state") in {"HALLUCINATION", "MISLED", "GLOBAL_OVERCLAIM", "REFUSE_TO_CORRECT"}:
        eval_state["global_state"]["hallucination_count"] += 1
        for kc_id in judge_result.get("covered_kc_ids", []) + judge_result.get("missing_kc_ids", []):
            if kc_id not in eval_state["kc_states"]:
                continue
            s = eval_state["kc_states"][kc_id]
            s["hallucination_history"].append(turn_id)
            s["status"] = "hallucinated"
            s["same_hallucination_followed_up_times"] += 1
            s["correction_status"] = "uncorrected"
            if not no_gameover_on_unresolved_hallucination() and s["same_hallucination_followed_up_times"] >= 2:
                s["status"] = "failed"
                eval_state["global_state"]["failed"] = True
                eval_state["global_state"]["failure_reason"] = f"REFUSE_TO_CORRECT:{kc_id}"
    else:
        for kc_id in judge_result.get("covered_kc_ids", []):
            if kc_id not in eval_state["kc_states"]:
                continue
            s = eval_state["kc_states"][kc_id]
            if s["correction_status"] == "uncorrected":
                s["correction_status"] = "corrected"
                s["status"] = "corrected"
            s["same_hallucination_followed_up_times"] = 0
    if path_id:
        if judge_result.get("state") in {"HALLUCINATION", "MISLED", "GLOBAL_OVERCLAIM", "REFUSE_TO_CORRECT"}:
            status = "fail"
        elif missing:
            status = "partial"
        else:
            status = "success"
        eval_state["path_states"][path_id].update({"status": status, "tested_by_turn": turn_id, "result": status})
    macro_update = _apply_macro_update(
        eval_state,
        turn_id,
        macro_id,
        question_type,
        lit,
        missing,
        question.get("target_kc_ids", []) if question else [],
    )
    thread_update = record_thread_step_result(
        eval_state,
        thread_id=thread_id,
        step_id=thread_step_id,
        turn_id=turn_id,
        question_type=question_type or "",
        judge_result=judge_result,
    )
    if question_type == "thread_bridge_question":
        eval_state["global_state"]["thread_bridge_tested_count"] += 1
        if thread_update.get("bridge_success") is True:
            eval_state["global_state"]["thread_bridge_success_count"] += 1
    challenge_update = record_challenge_result(
        eval_state,
        question=question or {"question_type": question_type, "macro_id": macro_id},
        turn_id=turn_id,
        judge_result=judge_result,
    )
    structured_update = record_structured_judge_state(
        eval_state,
        turn_id=turn_id,
        judge_result=judge_result,
        macro_id=macro_id,
        question_type=question_type,
    )
    eval_state["global_state"]["turn_count"] += 1
    return {
        "lit_kc": lit,
        "missing_kc": missing,
        "macro_update": macro_update,
        "thread_update": thread_update,
        "challenge_update": challenge_update,
        "thread_challenge_update": challenge_update if question_type == "thread_challenge_question" else {},
        "structured_update": structured_update,
        "failed": eval_state["global_state"]["failed"],
    }


def apply_claim_verification_results(eval_state: dict, turn_id: str, results: list[dict]) -> dict:
    labels: dict[str, int] = {}
    for item in results:
        label = str(item.get("label", "NOT_ENOUGH_INFO"))
        labels[label] = labels.get(label, 0) + 1
    supported_kc_ids = _apply_supported_claim_lighting(eval_state, turn_id, results)
    summary = {
        "verified_claim_count": len(results),
        "supported": labels.get("SUPPORTED", 0) + labels.get("NOT_IN_KC_BUT_SUPPORTED_BY_EVIDENCE", 0),
        "contradicted": labels.get("CONTRADICTED", 0),
        "overclaim": labels.get("OVERCLAIM", 0),
        "not_enough_info": labels.get("NOT_ENOUGH_INFO", 0),
        "supported_kc_ids": supported_kc_ids,
        "labels": labels,
    }
    eval_state.setdefault("claim_verification_states", {})[turn_id] = summary
    global_state = eval_state.setdefault("global_state", {})
    global_state["global_overclaim_count"] = global_state.get("global_overclaim_count", 0) + summary["overclaim"]
    global_state["global_contradicted_claim_count"] = (
        global_state.get("global_contradicted_claim_count", 0) + summary["contradicted"]
    )
    global_state["not_enough_info_claim_count"] = (
        global_state.get("not_enough_info_claim_count", 0) + summary["not_enough_info"]
    )
    if summary["contradicted"] or summary["overclaim"]:
        global_state["hallucination_count"] = global_state.get("hallucination_count", 0) + summary["contradicted"] + summary["overclaim"]
    return summary


def _apply_supported_claim_lighting(eval_state: dict, turn_id: str, results: list[dict]) -> list[str]:
    supported_labels = {"SUPPORTED", "NOT_IN_KC_BUT_SUPPORTED_BY_EVIDENCE"}
    lit: list[str] = []
    for item in results:
        if item.get("label") not in supported_labels:
            continue
        for kc_id in item.get("supporting_kc_ids") or item.get("matched_kc_ids") or item.get("retrieved_kc_ids") or []:
            if kc_id not in eval_state.get("kc_states", {}):
                continue
            state = eval_state["kc_states"][kc_id]
            if turn_id not in state.setdefault("covered_by_turns", []):
                state["covered_by_turns"].append(turn_id)
            if turn_id not in state.setdefault("globally_supported_by_turns", []):
                state["globally_supported_by_turns"].append(turn_id)
            state["confidence"] = max(float(state.get("confidence", 0.0) or 0.0), float(item.get("confidence", 0.0) or 0.0))
            if state.get("status") != "failed":
                state["status"] = "lit"
                if state.get("correction_status") == "uncorrected":
                    state["correction_status"] = "corrected"
            macro_id = state.get("macro_id")
            if macro_id:
                macro_state = eval_state.setdefault("macro_states", {}).setdefault(
                    macro_id,
                    {
                        "status": "not_started",
                        "main_question_asked": False,
                        "covered_kc_ids": [],
                        "missing_kc_ids": [],
                        "target_kc_ids": [],
                        "bank_kc_ids": [],
                        "related_turns": [],
                        "bank_kc_count": 0,
                        "active_kc_count": 0,
                    },
                )
                if kc_id not in macro_state.setdefault("covered_kc_ids", []):
                    macro_state["covered_kc_ids"].append(kc_id)
                if kc_id in macro_state.setdefault("missing_kc_ids", []):
                    macro_state["missing_kc_ids"].remove(kc_id)
                if turn_id not in macro_state.setdefault("related_turns", []):
                    macro_state["related_turns"].append(turn_id)
            if kc_id not in lit:
                lit.append(kc_id)
    return lit


def _apply_macro_update(
    eval_state: dict,
    turn_id: str,
    macro_id: str | None,
    question_type: str | None,
    lit: list[str],
    missing: list[str],
    target_kc_ids: list[str] | None,
) -> dict:
    if not macro_id:
        return {}
    state = eval_state.setdefault("macro_states", {}).setdefault(
        macro_id,
        {
            "status": "not_started",
            "main_question_asked": False,
            "covered_kc_ids": [],
            "missing_kc_ids": [],
            "target_kc_ids": [],
            "bank_kc_ids": [],
            "related_turns": [],
            "bank_kc_count": 0,
            "active_kc_count": 0,
        },
    )
    state["status"] = "in_progress"
    if question_type in {"main", "macro_main_question"}:
        state["main_question_asked"] = True
        targets = list(dict.fromkeys(kc_id for kc_id in (target_kc_ids or []) if kc_id))
        if targets:
            state["target_kc_ids"] = targets
            state["active_kc_count"] = len(targets)
    if turn_id not in state.setdefault("related_turns", []):
        state["related_turns"].append(turn_id)
    for kc_id in lit:
        if kc_id not in state.setdefault("covered_kc_ids", []):
            state["covered_kc_ids"].append(kc_id)
        if kc_id in state.setdefault("missing_kc_ids", []):
            state["missing_kc_ids"].remove(kc_id)
    for kc_id in missing:
        if kc_id not in state.setdefault("missing_kc_ids", []):
            state["missing_kc_ids"].append(kc_id)
    target_ids = set(state.get("target_kc_ids", []) or [])
    if target_ids:
        if target_ids <= set(state.get("covered_kc_ids", [])):
            state["status"] = "completed"
    elif state.get("active_kc_count") and len(state.get("covered_kc_ids", [])) >= state["active_kc_count"]:
        state["status"] = "completed"
    return {
        "macro_id": macro_id,
        "status": state.get("status"),
        "covered_kc_ids": state.get("covered_kc_ids", []),
        "missing_kc_ids": state.get("missing_kc_ids", []),
        "target_kc_ids": state.get("target_kc_ids", []),
    }


def _default_active_kc_ids_by_macro(master_graph: dict) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    active_id_set = set(master_graph.get("active_kc_ids", []))
    by_kc = {
        kc.get("kc_id"): kc
        for kc in master_graph.get("kc_nodes", [])
        if kc.get("kc_id")
    }
    for macro in master_graph.get("macro_nodes", []):
        macro_id = macro.get("macro_id")
        if not macro_id:
            continue
        macro_active_ids = [
            kc_id
            for kc_id in macro.get("active_kc_ids", [])
            if kc_id in by_kc
        ]
        if not macro_active_ids:
            macro_active_ids = [
                kc_id
                for kc_id in macro.get("kc_ids", [])
                if kc_id in active_id_set
                or by_kc.get(kc_id, {}).get("flags", {}).get("active_for_question_generation")
            ]
        out[macro_id] = macro_active_ids
    return out
