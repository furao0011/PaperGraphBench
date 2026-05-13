from __future__ import annotations

import os


CHALLENGE_QUESTION_TYPE = "challenge_question"
THREAD_CHALLENGE_QUESTION_TYPE = "thread_challenge_question"
CHALLENGE_QUESTION_TYPES = {CHALLENGE_QUESTION_TYPE, THREAD_CHALLENGE_QUESTION_TYPE}


def ensure_challenge_states(eval_state: dict, challenge_questions: list[dict]) -> None:
    states = eval_state.setdefault("challenge_states", {})
    for question in challenge_questions:
        question_id = question.get("question_id")
        if not question_id:
            continue
        state = states.setdefault(
            question_id,
            {
                "status": "not_asked",
                "turn_id": None,
                "result": None,
                "challenge_type": question.get("challenge_type"),
                "macro_ids": question.get("target_macro_ids", []),
                "thread_id": question.get("target_thread_id"),
                "challenge_scope": question.get("challenge_scope", "macro"),
            },
        )
        state.setdefault("status", "not_asked")
        state.setdefault("turn_id", None)
        state.setdefault("result", None)
        state.setdefault("challenge_type", question.get("challenge_type"))
        state.setdefault("macro_ids", question.get("target_macro_ids", []))
        state.setdefault("thread_id", question.get("target_thread_id"))
        state.setdefault("challenge_scope", question.get("challenge_scope", "macro"))


def get_macro_challenge(eval_state: dict, challenge_questions: list[dict], macro_id: str | None) -> dict | None:
    limit = _env_limit("EVAL_CHALLENGE_PER_MACRO", default=1, maximum=len(challenge_questions))
    if not macro_id or _macro_challenge_count(eval_state, macro_id) >= limit:
        return None
    ensure_challenge_states(eval_state, challenge_questions)
    for question in challenge_questions:
        if _already_asked(eval_state, question):
            continue
        if macro_id in (question.get("target_macro_ids") or []):
            return _make_challenge_turn(question, macro_id=macro_id, trigger="macro")
    return None


def get_thread_challenge(eval_state: dict, challenge_questions: list[dict], thread_id: str | None) -> dict | None:
    limit = _env_limit("EVAL_CHALLENGE_PER_THREAD", default=1, maximum=len(challenge_questions))
    if not thread_id or _thread_challenge_count(eval_state, thread_id) >= limit:
        return None
    ensure_challenge_states(eval_state, challenge_questions)
    for question in challenge_questions:
        if _already_asked(eval_state, question):
            continue
        if question.get("target_thread_id") == thread_id:
            return _make_challenge_turn(
                question,
                macro_id=_first_macro(question),
                trigger="thread",
            )
    return None


def get_ready_thread_challenge(
    eval_state: dict,
    thread_challenge_questions: list[dict],
    thread_id: str | None,
    completed_step_id: str | None,
    bridge_success: bool | None,
) -> dict | None:
    if not _env_bool("EVAL_THREAD_CHALLENGE_ENABLED", True):
        return None
    if not thread_id or not completed_step_id:
        return None
    if _env_bool("EVAL_THREAD_CHALLENGE_REQUIRE_BRIDGE_SUCCESS", False) and bridge_success is not True:
        return None
    limit = _env_limit("EVAL_THREAD_CHALLENGE_PER_THREAD", default=1, maximum=len(thread_challenge_questions))
    if _thread_challenge_question_count(eval_state, thread_id) >= limit:
        return None
    ensure_challenge_states(eval_state, thread_challenge_questions)
    for question in thread_challenge_questions:
        if _already_asked(eval_state, question):
            continue
        if question.get("target_thread_id") != thread_id and question.get("thread_id") != thread_id:
            continue
        insert_after = question.get("insert_after_step") or question.get("target_thread_turn_id")
        if insert_after and insert_after != completed_step_id:
            continue
        if not _thread_challenge_preconditions_met(eval_state, question):
            continue
        return _make_thread_challenge_turn(question, trigger="thread_bridge")
    return None


def record_challenge_result(eval_state: dict, question: dict, turn_id: str, judge_result: dict) -> dict:
    if question.get("question_type") not in CHALLENGE_QUESTION_TYPES:
        return {}
    question_id = question.get("question_id")
    if not question_id:
        return {}
    state = eval_state.setdefault("challenge_states", {}).setdefault(question_id, {})
    result = _challenge_result(judge_result)
    state.update(
        {
            "status": "asked",
            "turn_id": turn_id,
            "result": result,
            "challenge_type": question.get("challenge_type"),
            "macro_ids": question.get("target_macro_ids", []),
            "thread_id": question.get("target_thread_id"),
            "challenge_scope": question.get("challenge_scope", "macro"),
        }
    )
    global_state = eval_state.setdefault("global_state", {})
    global_state["challenge_question_count"] = global_state.get("challenge_question_count", 0) + 1
    if result == "failed":
        global_state["challenge_failure_count"] = global_state.get("challenge_failure_count", 0) + 1
    elif result == "resisted":
        global_state["challenge_resisted_count"] = global_state.get("challenge_resisted_count", 0) + 1
    macro_id = question.get("macro_id")
    if macro_id:
        macro_state = eval_state.setdefault("macro_states", {}).setdefault(macro_id, {})
        macro_state["challenge_question_count"] = macro_state.get("challenge_question_count", 0) + 1
    thread_id = question.get("target_thread_id")
    if thread_id:
        thread_state = eval_state.setdefault("thread_states", {}).setdefault(thread_id, {})
        if question.get("question_type") == THREAD_CHALLENGE_QUESTION_TYPE:
            thread_state["thread_challenge_count"] = thread_state.get("thread_challenge_count", 0) + 1
            if question_id not in thread_state.setdefault("thread_challenge_question_ids", []):
                thread_state["thread_challenge_question_ids"].append(question_id)
            thread_state.setdefault("thread_challenge_results", []).append(
                {
                    "question_id": question_id,
                    "turn_id": turn_id,
                    "challenge_type": question.get("challenge_type"),
                    "target_failure_mode": question.get("target_failure_mode"),
                    "result": result,
                }
            )
        else:
            thread_state["challenge_question_count"] = thread_state.get("challenge_question_count", 0) + 1
    return {
        "challenge_question_id": question_id,
        "challenge_result": result,
        "challenge_scope": question.get("challenge_scope", "macro"),
    }


def _make_challenge_turn(question: dict, macro_id: str | None, trigger: str) -> dict:
    item = dict(question)
    item["question_type"] = CHALLENGE_QUESTION_TYPE
    item["macro_id"] = macro_id or _first_macro(question)
    item["challenge_trigger"] = trigger
    return item


def _make_thread_challenge_turn(question: dict, trigger: str) -> dict:
    item = dict(question)
    item["question_type"] = THREAD_CHALLENGE_QUESTION_TYPE
    item["challenge_scope"] = "thread"
    item["macro_id"] = item.get("macro_id") or _first_macro(item)
    item["thread_id"] = item.get("thread_id") or item.get("target_thread_id")
    item["thread_turn_id"] = item.get("thread_turn_id") or item.get("target_thread_turn_id") or item.get("insert_after_step")
    item["thread_role"] = "thread_challenge"
    item["challenge_trigger"] = trigger
    return item


def _already_asked(eval_state: dict, question: dict) -> bool:
    state = eval_state.get("challenge_states", {}).get(question.get("question_id"), {})
    return state.get("status") == "asked"


def _challenge_result(judge_result: dict) -> str:
    if judge_result.get("state") in {"MISLED", "HALLUCINATION", "CHALLENGE_FAIL"}:
        return "failed"
    if judge_result.get("state") in {"CHALLENGE_RESISTED"}:
        return "resisted"
    return "inconclusive"


def _macro_challenge_count(eval_state: dict, macro_id: str) -> int:
    return int(eval_state.get("macro_states", {}).get(macro_id, {}).get("challenge_question_count", 0) or 0)


def _thread_challenge_count(eval_state: dict, thread_id: str) -> int:
    return int(eval_state.get("thread_states", {}).get(thread_id, {}).get("challenge_question_count", 0) or 0)


def _thread_challenge_question_count(eval_state: dict, thread_id: str) -> int:
    return int(eval_state.get("thread_states", {}).get(thread_id, {}).get("thread_challenge_count", 0) or 0)


def _thread_challenge_preconditions_met(eval_state: dict, question: dict) -> bool:
    kc_ids = [kc_id for kc_id in question.get("target_kc_ids", []) if kc_id]
    if not kc_ids:
        return False
    states = eval_state.get("kc_states", {})
    for kc_id in kc_ids:
        status = states.get(kc_id, {}).get("status")
        if status in {"lit", "corrected"}:
            continue
        if _coverage_gap_exhausted(eval_state, kc_id):
            continue
        return False
    return True


def _coverage_gap_exhausted(eval_state: dict, kc_id: str) -> bool:
    for gap in eval_state.get("coverage_gaps", {}).values():
        if kc_id in (gap.get("target_kc_ids") or []) and gap.get("status") == "exhausted":
            return True
    return False


def _first_macro(question: dict) -> str | None:
    macros = question.get("target_macro_ids") or []
    return macros[0] if macros else None


def _env_positive_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer, got {raw!r}.") from exc
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value}.")
    return value


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_limit(name: str, default: int, maximum: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    text = raw.strip().lower()
    if text in {"all", "full", "unlimited", "-1"}:
        return maximum
    try:
        value = int(text)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer or 'all', got {raw!r}.") from exc
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer or 'all', got {value}.")
    return value
