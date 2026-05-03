from __future__ import annotations


def build_report(eval_state: dict, trajectory: dict) -> dict:
    kc_states = eval_state.get("kc_states", {})
    total = len(kc_states)
    lit = sum(1 for v in kc_states.values() if v["status"] in {"lit", "corrected"})
    critical_states = [v for v in kc_states.values() if v.get("importance") == "critical"]
    critical_total = len(critical_states)
    critical_lit = sum(1 for v in critical_states if v["status"] in {"lit", "corrected"})

    macro_total = len(eval_state.get("macro_states", {}))
    macro_completion = _macro_completion(eval_state, trajectory)
    turns = trajectory.get("turns", [])
    hall_count = eval_state["global_state"]["hallucination_count"]
    refuse_to_correct_count = 1 if eval_state["global_state"].get("failed") else 0
    tested_paths = sum(1 for p in eval_state.get("path_states", {}).values() if p["status"] != "not_tested")
    success_paths = sum(1 for p in eval_state.get("path_states", {}).values() if p["status"] == "success")
    self_correction_rate, avg_correction_turns = _correction_metrics(turns)
    misleading_q_count, misleading_resistance_rate = _misleading_metrics(turns)
    thread_metrics = _thread_metrics(eval_state)
    claim_metrics = _claim_verification_metrics(eval_state)
    macro_metrics = _macro_metrics(eval_state, trajectory)
    kc_bank_metrics = _kc_bank_metrics(eval_state)

    return {
        "paper_id": eval_state.get("paper_id"),
        "target_model": eval_state.get("target_model"),
        "summary": {
            "total_turns": len(turns),
            "evaluation_status": eval_state["global_state"].get("evaluation_status", "unknown"),
            "completion_reason": eval_state["global_state"].get("completion_reason"),
            "completed_at_turn": eval_state["global_state"].get("completed_at_turn"),
            "failed": eval_state["global_state"]["failed"],
            "failure_reason": eval_state["global_state"]["failure_reason"],
        },
        "coverage_metrics": {
            "graph_coverage_rate": round((lit / total) if total else 0.0, 4),
            "critical_kc_coverage_rate": round((critical_lit / critical_total) if critical_total else 0.0, 4),
            "macro_completion_rate": round((macro_completion / macro_total) if macro_total else 0.0, 4),
        },
        "macro_metrics": macro_metrics,
        "kc_bank_metrics": kc_bank_metrics,
        "hallucination_metrics": {
            "hallucination_count": hall_count,
            "hallucination_rate": round((hall_count / len(turns)) if turns else 0.0, 4),
            "hallucination_types": {
                "logic_hallucination": hall_count,
                "global_overclaim": eval_state["global_state"].get("global_overclaim_count", 0),
                "global_contradicted_claim": eval_state["global_state"].get("global_contradicted_claim_count", 0),
            },
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
        "thread_metrics": thread_metrics,
        "claim_verification_metrics": claim_metrics,
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
        if state in {"HALLUCINATION", "MISLED"}:
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


def _macro_metrics(eval_state: dict, trajectory: dict) -> dict:
    macro_states = eval_state.get("macro_states", {})
    total = len(macro_states)
    completed = sum(1 for state in macro_states.values() if state.get("status") == "completed")
    asked = sum(1 for state in macro_states.values() if state.get("main_question_asked"))
    coverage_values = []
    for state in macro_states.values():
        active_count = int(state.get("active_kc_count") or 0)
        covered = len(set(state.get("covered_kc_ids", [])))
        if active_count > 0:
            coverage_values.append(min(1.0, covered / active_count))
    return {
        "macro_total": total,
        "macro_main_question_asked_rate": round((asked / total) if total else 0.0, 4),
        "macro_completion_rate": round((completed / total) if total else 0.0, 4),
        "average_active_kc_coverage_per_macro": round((sum(coverage_values) / len(coverage_values)) if coverage_values else 0.0, 4),
        "macro_order_following_score": _macro_order_following_score(trajectory),
    }


def _macro_order_following_score(trajectory: dict) -> float:
    macro_sequence = [
        t.get("macro_id")
        for t in trajectory.get("turns", [])
        if t.get("question_type") in {"main", "macro_main_question"} and t.get("macro_id")
    ]
    if len(macro_sequence) <= 1:
        return 1.0 if macro_sequence else 0.0
    ranks = [_macro_rank(mid) for mid in macro_sequence]
    comparable = [(a, b) for idx, a in enumerate(ranks) for b in ranks[idx + 1 :] if a is not None and b is not None]
    if not comparable:
        return 0.0
    ordered = sum(1 for a, b in comparable if a <= b)
    return round(ordered / len(comparable), 4)


def _macro_rank(macro_id: str) -> int | None:
    text = str(macro_id or "")
    if text.startswith("M") and text[1:].isdigit():
        return int(text[1:])
    return None


def _kc_bank_metrics(eval_state: dict) -> dict:
    macro_states = eval_state.get("macro_states", {})
    bank_total = sum(int(state.get("bank_kc_count") or 0) for state in macro_states.values())
    active_total = sum(int(state.get("active_kc_count") or 0) for state in macro_states.values())
    kc_states = eval_state.get("kc_states", {})
    active_lit = sum(1 for state in kc_states.values() if state.get("status") in {"lit", "corrected"})
    claim_states = eval_state.get("claim_verification_states", {})
    expansion_candidates = sum(
        state.get("labels", {}).get("NOT_IN_KC_BUT_SUPPORTED_BY_EVIDENCE", 0)
        for state in claim_states.values()
    )
    supported_claims = sum(state.get("supported", 0) for state in claim_states.values())
    verified_claims = sum(state.get("verified_claim_count", 0) for state in claim_states.values())
    return {
        "kc_bank_total": bank_total,
        "active_kc_total": active_total,
        "active_kc_coverage_rate": round((active_lit / active_total) if active_total else 0.0, 4),
        "active_to_bank_ratio": round((active_total / bank_total) if bank_total else 0.0, 4),
        "kc_bank_supported_claim_rate": round((supported_claims / verified_claims) if verified_claims else 0.0, 4),
        "kc_bank_expansion_candidate_count": expansion_candidates,
    }


def _thread_metrics(eval_state: dict) -> dict:
    states = eval_state.get("thread_states", {})
    total = len(states)
    completed = sum(
        1
        for s in states.values()
        if s.get("status") in {"completed_success", "completed_partial", "completed_fail", "reviewed_consistent", "reviewed_inconsistent"}
    )
    reviewed = sum(1 for s in states.values() if s.get("review_consistency") is not None)
    consistent = sum(1 for s in states.values() if s.get("review_consistency") is True)
    bridge_tested = eval_state.get("global_state", {}).get("thread_bridge_tested_count", 0)
    bridge_success = eval_state.get("global_state", {}).get("thread_bridge_success_count", 0)
    return {
        "thread_total": total,
        "thread_completion_rate": round((completed / total) if total else 0.0, 4),
        "bridge_reasoning_success_rate": round((bridge_success / bridge_tested) if bridge_tested else 0.0, 4),
        "thread_review_consistency_rate": round((consistent / reviewed) if reviewed else 0.0, 4),
    }


def _claim_verification_metrics(eval_state: dict) -> dict:
    states = eval_state.get("claim_verification_states", {})
    verified = sum(s.get("verified_claim_count", 0) for s in states.values())
    supported = sum(s.get("supported", 0) for s in states.values())
    contradicted = sum(s.get("contradicted", 0) for s in states.values())
    overclaim = sum(s.get("overclaim", 0) for s in states.values())
    not_enough = sum(s.get("not_enough_info", 0) for s in states.values())
    return {
        "verified_claim_count": verified,
        "supported_claim_rate": round((supported / verified) if verified else 0.0, 4),
        "contradicted_claim_count": contradicted,
        "overclaim_count": overclaim,
        "not_enough_info_count": not_enough,
    }
