from __future__ import annotations


def choose_next_action(eval_state: dict, judge_result: dict) -> str:
    if judge_result["state"] == "HALLUCINATION":
        return "hallucination_followup"
    if judge_result["state"] == "REFUSE_TO_CORRECT":
        return "end_failed"
    if judge_result.get("missing_kc_ids"):
        return "detail_followup"
    untested_paths = [p for p in eval_state.get("path_states", {}).values() if p["status"] == "not_tested"]
    if untested_paths:
        return "multi_hop_question"
    if eval_state["global_state"]["misleading_question_count"] < 1:
        return "misleading_followup"
    if eval_state["global_state"]["review_question_count"] < 1:
        return "review_followup"
    return "next_main_question"
