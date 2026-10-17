"""Validated source counts for the paper's first-answer and dialogue metrics."""

from __future__ import annotations


def validated_turns(trajectory: dict) -> list[dict]:
    turns = trajectory.get("turns")
    if not isinstance(turns, list) or not turns:
        raise ValueError("Trajectory must contain a nonempty turns list")
    seen = set()
    for turn in turns:
        if not isinstance(turn, dict):
            raise ValueError("Every trajectory turn must be an object")
        turn_id = turn.get("turn_id")
        if not isinstance(turn_id, str) or not turn_id.strip() or turn_id in seen:
            raise ValueError("Trajectory turn IDs must be nonempty and unique")
        seen.add(turn_id)
        for field in ("question_id", "question_type", "model_answer"):
            if not isinstance(turn.get(field), str):
                raise ValueError(f"Turn {turn_id} requires string {field}")
        if not turn["question_id"].strip() or not turn["question_type"].strip():
            raise ValueError(f"Turn {turn_id} requires nonempty question identity/type")
        if not isinstance(turn.get("judge_result"), dict):
            raise ValueError(f"Turn {turn_id} requires judge_result object")
        events(turn)
    return turns


def events(turn: dict) -> list[dict]:
    value = turn["judge_result"].get("hallucination_events")
    if not isinstance(value, list) or any(not isinstance(event, dict) for event in value):
        raise ValueError(f"Turn {turn.get('turn_id')} requires hallucination_events list of objects")
    return value


def _id_list(value, field: str, turn: dict) -> set[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item.strip() for item in value):
        raise ValueError(f"Turn {turn.get('turn_id')} requires {field} list of nonempty strings")
    return set(value)


def _coverage(turn: dict) -> dict:
    judge = turn["judge_result"]
    if "coverage" not in judge:
        return {}  # Documented legacy schema stores covered/missing KCs at judge level.
    value = judge["coverage"]
    if not isinstance(value, dict):
        raise ValueError(f"Turn {turn.get('turn_id')} coverage must be an object")
    return value


def kc_ids(turn: dict, kind: str) -> set[str]:
    """Support the original normalized schema and explicit legacy KC aliases."""
    field = kind + "_kc_ids"
    coverage = _coverage(turn)
    legacy = turn if kind == "target" else turn["judge_result"]
    provided = []
    for source in (coverage, legacy):
        if field in source:
            provided.append(_id_list(source[field], field, turn))
    if not provided:
        raise ValueError(f"Turn {turn.get('turn_id')} is missing {field} in coverage/legacy schema")
    return next((value for value in provided if value), provided[0])


def coverage_complete(turn: dict) -> bool:
    coverage = _coverage(turn)
    if "coverage_complete" in coverage:
        value = coverage["coverage_complete"]
        if not isinstance(value, bool):
            raise ValueError(f"Turn {turn.get('turn_id')} coverage_complete must be boolean")
        return value
    targets = kc_ids(turn, "target")
    return bool(targets) and targets <= kc_ids(turn, "covered")


def _challenge_failed(turn: dict) -> bool:
    result = turn["judge_result"].get("challenge_result")
    if result is not None and not isinstance(result, dict):
        raise ValueError(f"Turn {turn.get('turn_id')} challenge_result must be object or null")
    if result is not None and "failed" in result:
        if not isinstance(result["failed"], bool):
            raise ValueError(f"Turn {turn.get('turn_id')} challenge_result.failed must be boolean")
        return result["failed"]
    # The original legacy contract predates challenge_result.failed.
    return any(str(event.get("hallucination_type", "")).startswith("challenge_") for event in events(turn))


def _multimodal(turn: dict) -> bool:
    required = turn.get("requires_multimodal_input", False)
    refs = turn.get("asset_references", [])
    if not isinstance(required, bool) or not isinstance(refs, list):
        raise ValueError(f"Turn {turn.get('turn_id')} has invalid multimodal metadata")
    return required or bool(refs)


def trajectory_counts(trajectory: dict) -> dict[str, int]:
    turns = validated_turns(trajectory)
    macro = [turn for turn in turns if turn["question_type"] in {"main", "macro_main_question"}]
    # Thread challenges contribute events, but not the paper's Chal. Fail columns.
    challenges = [turn for turn in turns if turn["question_type"] == "challenge_question"]
    text = [turn for turn in challenges if not _multimodal(turn)]
    multimodal = [turn for turn in challenges if _multimodal(turn)]
    return {"total_turns": len(turns), "answer_chars": sum(len(turn["model_answer"]) for turn in turns),
            "macro_turns": len(macro),
            "macro_target_kc_mentions": sum(len(kc_ids(turn, "target")) for turn in macro),
            "macro_covered_kc_mentions": sum(len(kc_ids(turn, "target") & kc_ids(turn, "covered")) for turn in macro),
            "macro_complete_turns": sum(coverage_complete(turn) for turn in macro),
            "challenge_turns": len(challenges), "challenge_failed_turns": sum(_challenge_failed(turn) for turn in challenges),
            "text_challenge_turns": len(text), "text_challenge_failed_turns": sum(_challenge_failed(turn) for turn in text),
            "multimodal_challenge_turns": len(multimodal), "multimodal_challenge_failed_turns": sum(_challenge_failed(turn) for turn in multimodal),
            "hallucination_events": sum(len(events(turn)) for turn in turns),
            "hallucination_turns": sum(bool(events(turn)) for turn in turns)}
