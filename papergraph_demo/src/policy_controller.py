from __future__ import annotations


def choose_next_action(eval_state: dict, judge_result: dict) -> str:
    if judge_result["state"] in {"HALLUCINATION", "MISLED"}:
        return "hallucination_followup"
    if judge_result["state"] == "REFUSE_TO_CORRECT":
        return "end_failed"
    if judge_result.get("missing_kc_ids"):
        return "detail_followup"
    if _available_reasoning_path_exists(eval_state):
        return "multi_hop_question"
    return "next_main_question"


def _available_reasoning_path_exists(eval_state: dict) -> bool:
    kc_states = eval_state.get("kc_states", {})
    for path in eval_state.get("path_states", {}).values():
        if path.get("status") != "not_tested":
            continue
        sequence = path.get("kc_sequence", [])
        required = path.get("required_lit_kc") or sequence[:2]
        lit_count = sum(
            1
            for kc_id in set(required or sequence)
            if kc_states.get(kc_id, {}).get("status") in {"lit", "corrected"}
        )
        if len(sequence) >= 3 and lit_count >= 2:
            return True
    return False
