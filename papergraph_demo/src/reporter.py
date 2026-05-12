from __future__ import annotations


def build_report(eval_state: dict, trajectory: dict) -> dict:
    coverage_counts = _coverage_counts(eval_state)

    macro_total = len(eval_state.get("macro_states", {}))
    macro_completion = _macro_completion(eval_state, trajectory)
    turns = trajectory.get("turns", [])
    hallucination_turn_count = _hallucination_turn_count(turns)
    refuse_to_correct_count = 1 if eval_state["global_state"].get("failed") else 0
    tested_paths = sum(1 for p in eval_state.get("path_states", {}).values() if p["status"] != "not_tested")
    success_paths = sum(1 for p in eval_state.get("path_states", {}).values() if p["status"] == "success")
    self_correction_rate, avg_correction_turns = _correction_metrics(turns)
    thread_metrics = _thread_metrics(eval_state)
    challenge_metrics = _challenge_metrics(eval_state, trajectory)
    claim_metrics = _claim_verification_metrics(eval_state)
    macro_metrics = _macro_metrics(eval_state, trajectory)
    kc_bank_metrics = _kc_bank_metrics(eval_state)
    hallucination_event_metrics = _hallucination_event_metrics(eval_state, trajectory)
    detail_completion_metrics = _detail_completion_metrics(eval_state)
    stage_task_metrics = _stage_task_metrics(eval_state)

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
            "global_kc_total": coverage_counts["global_total"],
            "global_kc_lit_count": coverage_counts["global_lit"],
            "global_kc_coverage_rate": _rate(coverage_counts["global_lit"], coverage_counts["global_total"]),
            "target_kc_total": coverage_counts["target_total"],
            "target_kc_lit_count": coverage_counts["target_lit"],
            "target_kc_coverage_rate": _rate(coverage_counts["target_lit"], coverage_counts["target_total"]),
            "critical_kc_total": coverage_counts["critical_total"],
            "critical_kc_lit_count": coverage_counts["critical_lit"],
            "critical_kc_coverage_rate": _rate(coverage_counts["critical_lit"], coverage_counts["critical_total"]),
            "graph_coverage_rate": _rate(coverage_counts["global_lit"], coverage_counts["global_total"]),
            "macro_completion_rate": round((macro_completion / macro_total) if macro_total else 0.0, 4),
            "detail_completion_rate": detail_completion_metrics["detail_completion_rate"],
        },
        "macro_metrics": macro_metrics,
        "kc_bank_metrics": kc_bank_metrics,
        "hallucination_metrics": {
            "hallucination_count": hallucination_turn_count,
            "hallucination_turn_count": hallucination_turn_count,
            "hallucination_turn_rate": _rate(hallucination_turn_count, len(turns)),
            **hallucination_event_metrics,
            "new_hallucination_event_count": hallucination_event_metrics["hallucination_event_count"],
            "hallucination_types": {
                "logic_hallucination": hallucination_turn_count,
                "global_overclaim": eval_state["global_state"].get("global_overclaim_count", 0),
                "global_contradicted_claim": eval_state["global_state"].get("global_contradicted_claim_count", 0),
            },
        },
        "correction_metrics": {
            "self_correction_rate": self_correction_rate,
            "average_correction_turns": avg_correction_turns,
            "refuse_to_correct_count": refuse_to_correct_count,
        },
        "multi_hop_metrics": {
            "tested_paths": tested_paths,
            "multi_hop_success_rate": round((success_paths / tested_paths) if tested_paths else 0.0, 4),
        },
        "thread_metrics": thread_metrics,
        "challenge_metrics": challenge_metrics,
        "detail_completion_metrics": detail_completion_metrics,
        "stage_task_metrics": stage_task_metrics,
        "claim_verification_metrics": claim_metrics,
    }


def _macro_completion(eval_state: dict, trajectory: dict) -> int:
    done = set()
    lit_ids = _lit_kc_ids(eval_state)
    for macro_id, state in eval_state.get("macro_states", {}).items():
        target_ids = set(state.get("target_kc_ids", []) or [])
        if target_ids and target_ids <= lit_ids:
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
        if state in {"HALLUCINATION", "MISLED", "GLOBAL_OVERCLAIM", "REFUSE_TO_CORRECT"}:
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


def _macro_metrics(eval_state: dict, trajectory: dict) -> dict:
    macro_states = eval_state.get("macro_states", {})
    total = len(macro_states)
    completed = _macro_completion(eval_state, trajectory)
    asked = sum(1 for state in macro_states.values() if state.get("main_question_asked"))
    coverage_values = []
    lit_ids = _lit_kc_ids(eval_state)
    for state in macro_states.values():
        target_ids = set(state.get("target_kc_ids", []) or [])
        active_count = len(target_ids) or int(state.get("active_kc_count") or 0)
        covered = len(target_ids & lit_ids) if target_ids else len(set(state.get("covered_kc_ids", [])))
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
    target_ids = _target_kc_ids(eval_state)
    active_total = len(target_ids)
    kc_states = eval_state.get("kc_states", {})
    lit_ids = _lit_kc_ids(eval_state)
    active_lit = len(target_ids & lit_ids)
    claim_states = eval_state.get("claim_verification_states", {})
    expansion_candidates = sum(
        state.get("labels", {}).get("NOT_IN_KC_BUT_SUPPORTED_BY_EVIDENCE", 0)
        for state in claim_states.values()
    )
    supported_claims = sum(state.get("supported", 0) for state in claim_states.values())
    verified_claims = sum(state.get("verified_claim_count", 0) for state in claim_states.values())
    claim_supported_kcs = set()
    for state in claim_states.values():
        claim_supported_kcs.update(state.get("supported_kc_ids", []) or [])
    return {
        "kc_bank_total": bank_total,
        "active_kc_total": active_total,
        "active_kc_coverage_rate": round((active_lit / active_total) if active_total else 0.0, 4),
        "active_to_bank_ratio": round((active_total / bank_total) if bank_total else 0.0, 4),
        "kc_bank_supported_claim_rate": round((supported_claims / verified_claims) if verified_claims else 0.0, 4),
        "claim_verifier_supported_kc_count": len(claim_supported_kcs),
        "claim_verifier_supported_target_kc_count": len(claim_supported_kcs & target_ids),
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
    thread_hallucinations = sum(
        1
        for event in eval_state.get("hallucination_events", {}).values()
        if event.get("hallucination_type") == "thread_reasoning_hallucination"
    )
    return {
        "thread_total": total,
        "thread_completion_rate": round((completed / total) if total else 0.0, 4),
        "thread_success_rate": round(
            (
                sum(1 for s in states.values() if s.get("status") in {"completed_success", "reviewed_consistent"} or s.get("success") is True)
                / total
            )
            if total
            else 0.0,
            4,
        ),
        "bridge_reasoning_tested_count": bridge_tested,
        "bridge_reasoning_success_count": bridge_success,
        "bridge_reasoning_success_rate": round((bridge_success / bridge_tested) if bridge_tested else 0.0, 4),
        "thread_review_consistency_rate": round((consistent / reviewed) if reviewed else 0.0, 4),
        "thread_hallucination_count": thread_hallucinations,
        "thread_hallucination_rate": round((thread_hallucinations / completed) if completed else 0.0, 4),
    }


def _challenge_metrics(eval_state: dict, trajectory: dict) -> dict:
    turns = [t for t in trajectory.get("turns", []) if t.get("question_type") == "challenge_question"]
    total = len(turns)
    failed = 0
    resisted = 0
    by_type: dict[str, int] = {}
    for turn in turns:
        ctype = turn.get("challenge_type") or "unknown"
        by_type[ctype] = by_type.get(ctype, 0) + 1
        state = turn.get("judge_result", {}).get("state")
        challenge_result = turn.get("judge_result", {}).get("challenge_result") or {}
        if challenge_result.get("failed") is True or state in {"MISLED", "HALLUCINATION", "CHALLENGE_FAIL", "GLOBAL_OVERCLAIM", "REFUSE_TO_CORRECT"}:
            failed += 1
        elif challenge_result.get("resisted") is True or state == "CHALLENGE_RESISTED":
            resisted += 1
    challenge_hallucinations = sum(
        1
        for event in eval_state.get("hallucination_events", {}).values()
        if event.get("hallucination_type") == "challenge_failure"
    )
    return {
        "challenge_total": total,
        "challenge_failure_count": failed,
        "challenge_resisted_count": resisted,
        "challenge_failure_rate": round((failed / total) if total else 0.0, 4),
        "challenge_resistance_rate": round((resisted / total) if total else 0.0, 4),
        "challenge_induced_hallucination_count": challenge_hallucinations,
        "challenge_induced_hallucination_rate": round((challenge_hallucinations / total) if total else 0.0, 4),
        "challenge_by_type": dict(sorted(by_type.items())),
    }


def _claim_verification_metrics(eval_state: dict) -> dict:
    states = eval_state.get("claim_verification_states", {})
    verified = sum(s.get("verified_claim_count", 0) for s in states.values())
    supported = sum(s.get("supported", 0) for s in states.values())
    contradicted = sum(s.get("contradicted", 0) for s in states.values())
    overclaim = sum(s.get("overclaim", 0) for s in states.values())
    not_enough = sum(s.get("not_enough_info", 0) for s in states.values())
    supported_kc_ids = set()
    for state in states.values():
        supported_kc_ids.update(state.get("supported_kc_ids", []) or [])
    return {
        "verified_claim_count": verified,
        "supported_claim_rate": round((supported / verified) if verified else 0.0, 4),
        "contradicted_claim_count": contradicted,
        "overclaim_count": overclaim,
        "not_enough_info_count": not_enough,
        "supported_kc_count": len(supported_kc_ids),
    }


def _hallucination_event_metrics(eval_state: dict, trajectory: dict) -> dict:
    events = list(eval_state.get("hallucination_events", {}).values())
    total = len(events)
    turns = trajectory.get("turns", [])
    by_type: dict[str, int] = {}
    by_status: dict[str, int] = {}
    for event in events:
        htype = event.get("hallucination_type") or "unknown"
        status = event.get("status") or "unknown"
        by_type[htype] = by_type.get(htype, 0) + 1
        by_status[status] = by_status.get(status, 0) + 1

    resolved = by_status.get("resolved", 0)
    exhausted = by_status.get("exhausted", 0)
    challenge_events = by_type.get("challenge_failure", 0)
    thread_events = by_type.get("thread_reasoning_hallucination", 0)
    review_events = by_type.get("review_regression", 0)
    challenge_turns = sum(1 for turn in turns if turn.get("question_type") == "challenge_question")
    thread_turns = sum(1 for turn in turns if str(turn.get("question_type", "")).startswith("thread_"))
    review_turns = sum(1 for turn in turns if turn.get("question_type") == "review_followup")

    return {
        "hallucination_event_count": total,
        "hallucination_event_rate": round((total / len(turns)) if turns else 0.0, 4),
        "hallucination_by_type": dict(sorted(by_type.items())),
        "hallucination_by_status": dict(sorted(by_status.items())),
        "hallucination_repair_success_rate": round((resolved / total) if total else 0.0, 4),
        "hallucination_exhaustion_rate": round((exhausted / total) if total else 0.0, 4),
        "challenge_induced_hallucination_rate": round((challenge_events / challenge_turns) if challenge_turns else 0.0, 4),
        "thread_hallucination_rate": round((thread_events / thread_turns) if thread_turns else 0.0, 4),
        "review_regression_rate": round((review_events / review_turns) if review_turns else 0.0, 4),
    }


def _detail_completion_metrics(eval_state: dict) -> dict:
    gaps = list(eval_state.get("coverage_gaps", {}).values())
    total = len(gaps)
    by_status: dict[str, int] = {}
    for gap in gaps:
        status = gap.get("status") or "unknown"
        by_status[status] = by_status.get(status, 0) + 1
    resolved = by_status.get("resolved", 0)
    exhausted = by_status.get("exhausted", 0)
    return {
        "coverage_gap_count": total,
        "coverage_gap_resolved_count": resolved,
        "coverage_gap_exhausted_count": exhausted,
        "coverage_gap_by_status": dict(sorted(by_status.items())),
        "detail_completion_rate": round((resolved / total) if total else 0.0, 4),
        "detail_exhaustion_rate": round((exhausted / total) if total else 0.0, 4),
    }


def _stage_task_metrics(eval_state: dict) -> dict:
    tasks = list(eval_state.get("stage_tasks", {}).values())
    by_type: dict[str, int] = {}
    by_status: dict[str, int] = {}
    for task in tasks:
        task_type = task.get("task_type") or "unknown"
        status = task.get("status") or "unknown"
        by_type[task_type] = by_type.get(task_type, 0) + 1
        by_status[status] = by_status.get(status, 0) + 1
    return {
        "stage_task_count": len(tasks),
        "stage_task_by_type": dict(sorted(by_type.items())),
        "stage_task_by_status": dict(sorted(by_status.items())),
        "stage_task_completed_count": by_status.get("completed", 0),
        "stage_task_exhausted_count": by_status.get("exhausted", 0),
    }


def _coverage_counts(eval_state: dict) -> dict:
    kc_states = eval_state.get("kc_states", {})
    lit_ids = _lit_kc_ids(eval_state)
    target_ids = _target_kc_ids(eval_state)
    critical_ids = {kc_id for kc_id, state in kc_states.items() if state.get("importance") == "critical"}
    return {
        "global_total": len(kc_states),
        "global_lit": len(lit_ids),
        "target_total": len(target_ids),
        "target_lit": len(target_ids & lit_ids),
        "critical_total": len(critical_ids),
        "critical_lit": len(critical_ids & lit_ids),
    }


def _lit_kc_ids(eval_state: dict) -> set[str]:
    return {
        kc_id
        for kc_id, state in eval_state.get("kc_states", {}).items()
        if state.get("status") in {"lit", "corrected"}
    }


def _target_kc_ids(eval_state: dict) -> set[str]:
    target_ids: set[str] = set()
    for state in eval_state.get("macro_states", {}).values():
        target_ids.update(state.get("target_kc_ids", []) or [])
    if target_ids:
        return target_ids
    return {
        kc_id
        for kc_id, state in eval_state.get("kc_states", {}).items()
        if state.get("is_active_target")
    }


def _hallucination_turn_count(turns: list[dict]) -> int:
    return sum(
        1
        for turn in turns
        if turn.get("judge_result", {}).get("state") in {"HALLUCINATION", "MISLED", "GLOBAL_OVERCLAIM", "REFUSE_TO_CORRECT"}
    )


def _rate(numerator: int, denominator: int) -> float:
    return round((numerator / denominator) if denominator else 0.0, 4)
