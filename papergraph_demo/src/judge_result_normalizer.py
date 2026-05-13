from __future__ import annotations

import re


HALLUCINATION_STATES = {"HALLUCINATION", "MISLED", "GLOBAL_OVERCLAIM", "REFUSE_TO_CORRECT"}
CHALLENGE_STATES = {"CHALLENGE_RESISTED", "MISLED", "HALLUCINATION", "INCOMPLETE"}
THREAD_QUESTION_TYPES = {
    "thread_premise_question",
    "thread_evidence_question",
    "thread_bridge_question",
    "thread_review_question",
    "thread_question",
}
THREAD_CHALLENGE_QUESTION_TYPE = "thread_challenge_question"


def normalize_judge_result(judge_result: dict, turn_context: dict) -> dict:
    if not turn_context.get("turn_id"):
        raise ValueError("JudgeResult normalization requires turn_context.turn_id.")
    if not turn_context.get("question_id"):
        raise ValueError("JudgeResult normalization requires turn_context.question_id.")
    if not turn_context.get("question_type"):
        raise ValueError("JudgeResult normalization requires turn_context.question_type.")

    fixed = _repair_legacy_incomplete_without_missing(judge_result)
    normalized = dict(fixed)
    if isinstance(normalized.get("coverage"), dict):
        coverage = normalized["coverage"]
        normalized.setdefault("covered_kc_ids", coverage.get("covered_kc_ids", []))
        normalized.setdefault("missing_kc_ids", coverage.get("missing_kc_ids", []))
    normalized.setdefault("covered_kc_ids", [])
    normalized.setdefault("missing_kc_ids", [])
    if isinstance(normalized.get("hallucination_events"), list):
        normalized.setdefault(
            "hallucinated_claims",
            [event.get("claim") for event in normalized["hallucination_events"] if event.get("claim")],
        )
    else:
        normalized.setdefault("hallucinated_claims", [])
    normalized.setdefault("matched_forbidden_claims", [])
    normalized.setdefault("mentioned_unexplained_kc_ids", [])
    normalized.setdefault("reasoning_path_result", None)

    normalized["coverage"] = _normalize_coverage(normalized, turn_context)
    normalized["hallucination_events"] = _normalize_hallucination_events(normalized, turn_context)
    normalized["challenge_result"] = _normalize_challenge_result(normalized, turn_context)
    normalized["thread_result"] = _normalize_thread_result(normalized, turn_context)
    normalized.setdefault("recommended_tasks", normalized.get("recommended_stage_tasks", []))
    normalized["normalization_version"] = "v2.step3"
    return normalized


def normalize_after_global_claim_verification(judge_result: dict, turn_context: dict, claim_results: list[dict]) -> dict:
    normalized = dict(judge_result)
    normalized["global_claim_verification"] = claim_results
    risky = [item for item in claim_results if item.get("label") in {"CONTRADICTED", "OVERCLAIM"}]
    if risky:
        normalized["state"] = "GLOBAL_OVERCLAIM"
        normalized["next_action"] = "hallucination_followup"
        normalized["policy_next_action"] = "hallucination_followup"
    return normalize_judge_result(normalized, turn_context)


def _repair_legacy_incomplete_without_missing(judge_result: dict) -> dict:
    missing = judge_result.get("missing_kc_ids")
    if missing is None and isinstance(judge_result.get("coverage"), dict):
        missing = judge_result["coverage"].get("missing_kc_ids")
    if judge_result.get("state") != "INCOMPLETE" or missing:
        return judge_result
    fixed = dict(judge_result)
    fixed["state"] = "MAIN_PROGRESS"
    fixed["next_action"] = "next_main_question"
    explanation = fixed.get("judge_explanation", "")
    fixed["judge_explanation"] = (
        explanation
        + " Normalized: all target KCs were covered; incompleteness only concerned off-target material."
    ).strip()
    return fixed


def _normalize_coverage(judge_result: dict, turn_context: dict) -> dict:
    if isinstance(judge_result.get("coverage"), dict):
        coverage = dict(judge_result["coverage"])
        coverage.setdefault("target_kc_ids", list(turn_context.get("target_kc_ids", [])))
        coverage["covered_kc_ids"] = _dedupe(coverage.get("covered_kc_ids", []))
        coverage["missing_kc_ids"] = _dedupe(coverage.get("missing_kc_ids", []))
        coverage.setdefault("coverage_complete", not coverage["missing_kc_ids"])
        coverage.setdefault("confidence", float(judge_result.get("confidence", 0.0) or 0.0))
        return coverage
    target_kc_ids = list(turn_context.get("target_kc_ids", []))
    covered_kc_ids = _dedupe(judge_result.get("covered_kc_ids", []))
    missing_kc_ids = _dedupe(judge_result.get("missing_kc_ids", []))
    return {
        "target_kc_ids": target_kc_ids,
        "covered_kc_ids": covered_kc_ids,
        "missing_kc_ids": missing_kc_ids,
        "coverage_complete": not missing_kc_ids,
        "confidence": float(judge_result.get("confidence", 0.0) or 0.0),
    }


def _normalize_hallucination_events(judge_result: dict, turn_context: dict) -> list[dict]:
    events: list[dict] = []
    state = judge_result.get("state")
    question_type = turn_context.get("question_type")

    for idx, raw_event in enumerate(judge_result.get("hallucination_events", []) or [], start=1):
        if not isinstance(raw_event, dict):
            continue
        event = dict(raw_event)
        event.setdefault("event_id", f"H_{_safe_id(str(turn_context['turn_id']))}_{idx}")
        event.setdefault("created_at_turn", str(turn_context["turn_id"]))
        event.setdefault("macro_id", turn_context.get("macro_id"))
        event.setdefault("source_question_id", turn_context.get("question_id"))
        event.setdefault("source_question_type", turn_context.get("question_type"))
        raw_hallucination_type = event.get("hallucination_type") or judge_result.get("hallucination_type")
        event["hallucination_family"] = _hallucination_family_for_question(question_type, state)
        event["hallucination_type"] = _hallucination_type_for_context(turn_context, state)
        event.setdefault("subtype", raw_hallucination_type or _state_subtype(state))
        if raw_hallucination_type and raw_hallucination_type != event["hallucination_type"]:
            event["judge_hallucination_type"] = raw_hallucination_type
        _attach_challenge_event_context(event, turn_context)
        event.setdefault("claim", "")
        event.setdefault("source", "judge_result.hallucination_events")
        event["related_kc_ids"] = _dedupe(event.get("related_kc_ids", []) or _related_kc_ids(judge_result, turn_context))
        event.setdefault("matched_forbidden_claims", judge_result.get("matched_forbidden_claims", []))
        event.setdefault("status", "unresolved")
        event.setdefault("followup_turns", [])
        event.setdefault("max_followups", int(turn_context.get("max_followups", 3) or 3))
        event.setdefault("resolved_at_turn", None)
        events.append(event)

    for idx, claim in enumerate([] if events else (judge_result.get("hallucinated_claims", []) or []), start=1):
        events.append(
            _make_event(
                turn_context,
                idx,
                hallucination_type=_hallucination_type_for_context(turn_context, state),
                hallucination_family=_hallucination_family_for_question(question_type, state),
                subtype=judge_result.get("hallucination_type"),
                claim=str(claim),
                source="judge_result.hallucinated_claims",
                related_kc_ids=_related_kc_ids(judge_result, turn_context),
                matched_forbidden_claims=judge_result.get("matched_forbidden_claims", []),
            )
        )

    for claim_result in judge_result.get("global_claim_verification", []) or []:
        label = claim_result.get("label")
        if label not in {"CONTRADICTED", "OVERCLAIM"}:
            continue
        events.append(
            _make_event(
                turn_context,
                len(events) + 1,
                hallucination_type="contradicted_kc_claim" if label == "CONTRADICTED" else "fabricated_claim",
                hallucination_family="contradicted_kc_claim" if label == "CONTRADICTED" else "fabricated_claim",
                subtype=label.lower(),
                claim=str(claim_result.get("claim", "")),
                source="global_claim_verification",
                related_kc_ids=_dedupe(
                    list(claim_result.get("matched_kc_ids", []) or [])
                    + list(claim_result.get("contradicted_kc_ids", []) or [])
                    + list(turn_context.get("target_kc_ids", []) or [])
                ),
                matched_forbidden_claims=claim_result.get("matched_forbidden_claims", []),
            )
        )

    repair_context = turn_context.get("repair_context") or {}
    active_repair_without_new_event = (
        repair_context.get("repair_type") == "hallucination"
        and bool(repair_context.get("active_hallucinations"))
    )
    if state in HALLUCINATION_STATES and not events and not active_repair_without_new_event:
        events.append(
            _make_event(
                turn_context,
                1,
                hallucination_type=_hallucination_type_for_context(turn_context, state),
                hallucination_family=_hallucination_family_for_question(question_type, state),
                subtype=judge_result.get("hallucination_type") or _state_subtype(state),
                claim=str(judge_result.get("judge_explanation", "") or state),
                source="judge_state",
                related_kc_ids=_related_kc_ids(judge_result, turn_context),
                matched_forbidden_claims=judge_result.get("matched_forbidden_claims", []),
            )
        )
    return events


def _normalize_challenge_result(judge_result: dict, turn_context: dict) -> dict | None:
    if turn_context.get("question_type") not in {"challenge_question", THREAD_CHALLENGE_QUESTION_TYPE}:
        return None
    state = judge_result.get("state")
    return {
        "state": state,
        "challenge_scope": turn_context.get("challenge_scope", "macro"),
        "challenge_type": turn_context.get("challenge_type"),
        "challenge_trigger": turn_context.get("challenge_trigger"),
        "target_failure_mode": turn_context.get("target_failure_mode"),
        "expected_behavior": turn_context.get("expected_behavior"),
        "resisted": state == "CHALLENGE_RESISTED",
        "failed": state in HALLUCINATION_STATES,
        "incomplete": state == "INCOMPLETE",
        "confidence": float(judge_result.get("confidence", 0.0) or 0.0),
    }


def _normalize_thread_result(judge_result: dict, turn_context: dict) -> dict | None:
    question_type = turn_context.get("question_type")
    if question_type not in THREAD_QUESTION_TYPES and question_type != THREAD_CHALLENGE_QUESTION_TYPE:
        return None
    state = judge_result.get("state")
    missing = bool(judge_result.get("missing_kc_ids"))
    failed = state in HALLUCINATION_STATES
    return {
        "state": state,
        "thread_id": turn_context.get("thread_id"),
        "thread_turn_id": turn_context.get("thread_turn_id"),
        "thread_role": turn_context.get("thread_role"),
        "reasoning_path_result": judge_result.get("reasoning_path_result"),
        "success": not failed and not missing,
        "partial": not failed and missing,
        "failed": failed,
        "confidence": float(judge_result.get("confidence", 0.0) or 0.0),
    }


def _make_event(
    turn_context: dict,
    ordinal: int,
    hallucination_type: str,
    hallucination_family: str,
    subtype: str | None,
    claim: str,
    source: str,
    related_kc_ids: list[str],
    matched_forbidden_claims: list[str],
) -> dict:
    turn_id = str(turn_context["turn_id"])
    event_id = f"H_{_safe_id(turn_id)}_{ordinal}"
    event = {
        "event_id": event_id,
        "created_at_turn": turn_id,
        "macro_id": turn_context.get("macro_id"),
        "source_question_id": turn_context.get("question_id"),
        "source_question_type": turn_context.get("question_type"),
        "hallucination_family": hallucination_family,
        "hallucination_type": hallucination_type,
        "subtype": subtype,
        "claim": claim,
        "source": source,
        "related_kc_ids": related_kc_ids,
        "matched_forbidden_claims": matched_forbidden_claims or [],
        "status": "unresolved",
        "followup_turns": [],
        "max_followups": int(turn_context.get("max_followups", 3) or 3),
        "resolved_at_turn": None,
    }
    _attach_challenge_event_context(event, turn_context)
    return event


def _hallucination_family_for_question(question_type: str | None, state: str | None) -> str:
    if question_type == THREAD_CHALLENGE_QUESTION_TYPE:
        return "thread_reasoning_hallucination"
    if question_type == "challenge_question" or state == "MISLED":
        return "challenge_failure"
    if question_type in THREAD_QUESTION_TYPES:
        return "thread_reasoning_hallucination"
    if question_type == "review_followup":
        return "review_regression"
    if state == "GLOBAL_OVERCLAIM":
        return "fabricated_claim"
    return "contradicted_kc_claim"


def _hallucination_type_for_context(turn_context: dict, state: str | None) -> str:
    question_type = turn_context.get("question_type")
    if question_type == THREAD_CHALLENGE_QUESTION_TYPE:
        return _thread_challenge_hallucination_type(turn_context)
    if question_type == "challenge_question" or state == "MISLED":
        return _challenge_hallucination_type(turn_context)
    return _hallucination_family_for_question(question_type, state)


def _challenge_hallucination_type(turn_context: dict) -> str:
    failure_mode = str(turn_context.get("target_failure_mode") or "").strip()
    challenge_type = str(turn_context.get("challenge_type") or "").strip()
    if failure_mode == "false_premise" or challenge_type == "false_premise_challenge":
        return "challenge_false_premise_acceptance"
    if failure_mode == "overclaim" or challenge_type == "overclaim_challenge":
        return "challenge_overclaim_acceptance"
    if failure_mode == "wrong_relation" or challenge_type == "wrong_relation_challenge":
        return "challenge_wrong_relation_acceptance"
    return "challenge_failure"


def _thread_challenge_hallucination_type(turn_context: dict) -> str:
    failure_mode = str(turn_context.get("target_failure_mode") or "").strip()
    challenge_type = str(turn_context.get("challenge_type") or "").strip()
    if failure_mode == "thread_wrong_bridge" or challenge_type == "thread_wrong_bridge_challenge":
        return "thread_wrong_bridge_acceptance"
    if failure_mode == "thread_overclaim" or challenge_type == "thread_overclaim_challenge":
        return "thread_overclaim_acceptance"
    if failure_mode == "thread_premise_mutation" or challenge_type == "thread_premise_mutation_challenge":
        return "thread_premise_mutation_acceptance"
    return "thread_reasoning_hallucination"


def _attach_challenge_event_context(event: dict, turn_context: dict) -> None:
    if event.get("hallucination_family") not in {"challenge_failure", "thread_reasoning_hallucination"}:
        return
    if turn_context.get("challenge_scope"):
        event["challenge_scope"] = turn_context.get("challenge_scope")
    if turn_context.get("thread_id"):
        event["thread_id"] = turn_context.get("thread_id")
    if turn_context.get("thread_turn_id"):
        event["thread_turn_id"] = turn_context.get("thread_turn_id")
    if turn_context.get("challenge_type"):
        event["challenge_type"] = turn_context.get("challenge_type")
    if turn_context.get("target_failure_mode"):
        event["target_failure_mode"] = turn_context.get("target_failure_mode")
    if turn_context.get("challenge_trigger"):
        event["challenge_trigger"] = turn_context.get("challenge_trigger")
    if turn_context.get("expected_behavior"):
        event["expected_behavior"] = turn_context.get("expected_behavior")
    event["requires_multimodal_input"] = bool(turn_context.get("requires_multimodal_input"))
    if turn_context.get("modality_pool"):
        event["modality_pool"] = turn_context.get("modality_pool")
    asset_refs = turn_context.get("asset_references") or []
    if asset_refs:
        event["asset_ids"] = _dedupe([ref.get("asset_id") for ref in asset_refs if isinstance(ref, dict)])
        event["asset_types"] = _dedupe([ref.get("asset_type") for ref in asset_refs if isinstance(ref, dict)])


def _state_subtype(state: str | None) -> str | None:
    if not state:
        return None
    return str(state).lower()


def _related_kc_ids(judge_result: dict, turn_context: dict) -> list[str]:
    return _dedupe(
        list(judge_result.get("covered_kc_ids", []) or [])
        + list(judge_result.get("missing_kc_ids", []) or [])
        + list(turn_context.get("target_kc_ids", []) or [])
    )


def _dedupe(values: list) -> list:
    return list(dict.fromkeys(str(v) for v in values if v is not None and str(v)))


def _safe_id(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_]+", "_", value.strip())
    return safe.strip("_") or "NA"
