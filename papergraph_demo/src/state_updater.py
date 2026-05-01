from __future__ import annotations


def initialize_eval_state(master_graph: dict, target_model: str) -> dict:
    kc_states = {}
    for kc in master_graph.get("kc_nodes", []):
        kc_states[kc["kc_id"]] = {
            "status": "unlit",
            "covered_by_turns": [],
            "missed_by_turns": [],
            "confidence": 0.0,
            "hallucination_history": [],
            "correction_status": "not_needed",
            "same_hallucination_followed_up_times": 0,
            "importance": kc.get("importance", "normal"),
            "macro_id": kc.get("macro_id"),
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
                "misleading_question_count": 0,
            }
            for macro in master_graph.get("macro_nodes", [])
            if macro.get("macro_id")
        },
        "global_state": {
            "turn_count": 0,
            "hallucination_count": 0,
            "misleading_question_count": 0,
            "review_question_count": 0,
            "failed": False,
            "failure_reason": None,
        },
    }


def apply_judge_result(eval_state: dict, turn_id: str, judge_result: dict, path_id: str | None = None) -> dict:
    lit = []
    missing = []
    for kc_id in judge_result.get("covered_kc_ids", []):
        s = eval_state["kc_states"][kc_id]
        s["status"] = "lit"
        s["covered_by_turns"].append(turn_id)
        s["confidence"] = judge_result.get("confidence", 0.7)
        lit.append(kc_id)
    for kc_id in judge_result.get("missing_kc_ids", []):
        s = eval_state["kc_states"][kc_id]
        if s["status"] == "unlit":
            s["status"] = "missing"
        s["missed_by_turns"].append(turn_id)
        missing.append(kc_id)
    if judge_result.get("state") in {"HALLUCINATION", "MISLED"}:
        eval_state["global_state"]["hallucination_count"] += 1
        for kc_id in judge_result.get("covered_kc_ids", []) + judge_result.get("missing_kc_ids", []):
            s = eval_state["kc_states"][kc_id]
            s["hallucination_history"].append(turn_id)
            s["status"] = "hallucinated"
            s["same_hallucination_followed_up_times"] += 1
            s["correction_status"] = "uncorrected"
            if s["same_hallucination_followed_up_times"] >= 2:
                s["status"] = "failed"
                eval_state["global_state"]["failed"] = True
                eval_state["global_state"]["failure_reason"] = f"REFUSE_TO_CORRECT:{kc_id}"
    else:
        for kc_id in judge_result.get("covered_kc_ids", []):
            s = eval_state["kc_states"][kc_id]
            if s["correction_status"] == "uncorrected":
                s["correction_status"] = "corrected"
                s["status"] = "corrected"
            s["same_hallucination_followed_up_times"] = 0
    if path_id:
        if judge_result.get("state") in {"HALLUCINATION", "MISLED"}:
            status = "fail"
        elif missing:
            status = "partial"
        else:
            status = "success"
        eval_state["path_states"][path_id].update({"status": status, "tested_by_turn": turn_id, "result": status})
    eval_state["global_state"]["turn_count"] += 1
    return {"lit_kc": lit, "missing_kc": missing, "failed": eval_state["global_state"]["failed"]}
