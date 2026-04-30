from __future__ import annotations


def build_report(eval_state: dict, trajectory: dict) -> dict:
    kc_states = eval_state.get("kc_states", {})
    total = len(kc_states)
    lit = sum(1 for v in kc_states.values() if v["status"] in {"lit", "corrected"})
    critical_states = [v for v in kc_states.values() if v.get("importance") == "critical"]
    critical_total = len(critical_states)
    critical_lit = sum(1 for v in critical_states if v["status"] in {"lit", "corrected"})

    macro_total = 4
    macro_completion = _macro_completion(eval_state, trajectory)
    turns = trajectory.get("turns", [])
    hall_count = eval_state["global_state"]["hallucination_count"]
    refuse_to_correct_count = 1 if eval_state["global_state"].get("failed") else 0
    tested_paths = sum(1 for p in eval_state.get("path_states", {}).values() if p["status"] != "not_tested")
    success_paths = sum(1 for p in eval_state.get("path_states", {}).values() if p["status"] == "success")
    self_correction_rate, avg_correction_turns = _correction_metrics(turns)
    misleading_q_count, misleading_resistance_rate = _misleading_metrics(turns)

    return {
        "paper_id": eval_state.get("paper_id"),
        "target_model": eval_state.get("target_model"),
        "summary": {
            "total_turns": len(turns),
            "failed": eval_state["global_state"]["failed"],
            "failure_reason": eval_state["global_state"]["failure_reason"],
        },
        "coverage_metrics": {
            "graph_coverage_rate": round((lit / total) if total else 0.0, 4),
            "critical_kc_coverage_rate": round((critical_lit / critical_total) if critical_total else 0.0, 4),
            "macro_completion_rate": round(macro_completion / macro_total, 4),
        },
        "hallucination_metrics": {
            "hallucination_count": hall_count,
            "hallucination_rate": round((hall_count / len(turns)) if turns else 0.0, 4),
            "hallucination_types": {"logic_hallucination": hall_count},
        },
        "correction_metrics": {
            "self_correction_rate": self_correction_rate,
            "average_correction_turns": avg_correction_turns,
            "refuse_to_correct_count": refuse_to_correct_count,
        },
        "misleading_metrics": {
            "misleading_questions": misleading_q_count,
            "misleading_resistance_rate": misleading_resistance_rate,
        },
        "multi_hop_metrics": {
            "tested_paths": tested_paths,
            "multi_hop_success_rate": round((success_paths / tested_paths) if tested_paths else 0.0, 4),
        },
    }


def _macro_completion(eval_state: dict, trajectory: dict) -> int:
    done = set()
    by_macro: dict[str, list[dict]] = {}
    for state in eval_state.get("kc_states", {}).values():
        macro_id = state.get("macro_id")
        if macro_id:
            by_macro.setdefault(macro_id, []).append(state)
    for macro_id, states in by_macro.items():
        if states and all(s.get("status") in {"lit", "corrected"} for s in states):
            done.add(macro_id)
    return len(done)


def _correction_metrics(turns: list[dict]) -> tuple[float, float]:
    first_h = {}
    first_fixed_delta = []
    for i, t in enumerate(turns):
        jr = t.get("judge_result", {})
        state = jr.get("state")
        covered = jr.get("covered_kc_ids", [])
        missing = jr.get("missing_kc_ids", [])
        if state == "HALLUCINATION":
            for kc in covered + missing:
                first_h.setdefault(kc, i)
        else:
            for kc in covered:
                if kc in first_h:
                    first_fixed_delta.append(i - first_h[kc])
                    del first_h[kc]
    total_h = len(first_h) + len(first_fixed_delta)
    if total_h == 0:
        return 0.0, 0.0
    rate = len(first_fixed_delta) / total_h
    avg_turns = (sum(first_fixed_delta) / len(first_fixed_delta)) if first_fixed_delta else 0.0
    return round(rate, 4), round(avg_turns, 4)


def _misleading_metrics(turns: list[dict]) -> tuple[int, float]:
    misleading = [t for t in turns if t.get("question_type") == "misleading_followup"]
    total = len(misleading)
    if total == 0:
        return 0, 0.0
    resisted = 0
    for t in misleading:
        jr = t.get("judge_result", {})
        state = jr.get("state")
        if state in {"MISLEADING_RESISTED", "MAIN_PROGRESS", "SELF_CORRECTED"} and not jr.get("missing_kc_ids"):
            resisted += 1
    return total, round(resisted / total, 4)
