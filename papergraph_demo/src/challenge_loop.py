from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path

from src.challenge_filter import (
    challenge_solver_configs,
    question_with_filter_metadata,
    run_single_challenge_question_trials,
)
from src.challenge_question_generator import generate_challenge_question_for_plan
from src.model_client import OpenAICompatClient
from src.progress import log, span
from src.prompt_loader import load_prompt, render_prompt


def build_challenge_questions_loop(
    challenge_plans: dict,
    client: OpenAICompatClient,
    paper_text: str,
    cache_path: Path,
    resume: bool = False,
    restart: bool = False,
) -> dict:
    if not client or not client.is_ready():
        raise RuntimeError("Challenge loop requires a configured online model client.")
    if not isinstance(paper_text, str) or not paper_text.strip():
        raise ValueError("Challenge loop requires non-empty full paper text.")
    plans = challenge_plans.get("challenge_plans", [])
    if not isinstance(plans, list) or not plans:
        raise ValueError("Challenge loop requires non-empty challenge_plans.")

    plan_by_id = _plan_by_id(plans)
    target_count = _env_positive_int("CHALLENGE_ACCEPT_TARGET", 10)
    max_attempts_per_plan = _env_positive_int("CHALLENGE_MAX_ATTEMPTS_PER_PLAN", 3)
    solver_configs = challenge_solver_configs(client.cfg.llm_model)
    signature = _loop_signature(challenge_plans, solver_configs, paper_text, target_count, max_attempts_per_plan)
    state = _load_cache(cache_path) if resume and not restart else {}
    if state.get("challenge_loop_signature") != signature:
        state = _initial_state(challenge_plans, signature, plans, target_count, max_attempts_per_plan)
        _write_json(cache_path, state)

    usability_tpl = load_prompt("judge_challenge_question_usability.txt")
    easiness_tpl = load_prompt("judge_challenge_plan_easiness.txt")

    while len(state["accepted_questions"]) < target_count:
        plan = _next_plan(state, plan_by_id, max_attempts_per_plan)
        if plan is None:
            break
        plan_id = plan["challenge_plan_id"]
        feedback = state["revision_feedback_by_plan_id"].get(plan_id, "")
        attempt = int(state["attempts_by_plan_id"].get(plan_id, 0)) + 1
        question_id = f"CHQ_{int(state['next_question_index']):04d}"
        state["next_question_index"] = int(state["next_question_index"]) + 1
        state["attempts_by_plan_id"][plan_id] = attempt

        with span("challenge loop attempt", plan_id=plan_id, attempt=attempt, accepted=len(state["accepted_questions"])):
            question = generate_challenge_question_for_plan(
                plan=plan,
                client=client,
                question_id=question_id,
                revision_feedback=feedback,
            )
            question["loop_metadata"] = {
                "attempt": attempt,
                "revision_feedback": feedback,
            }
            state["raw_questions"].append(question)
            usability = _judge_question_usability(plan, question, client, usability_tpl)
            question["usability_check"] = usability

            if not usability["usable"]:
                _record_rejection(
                    state,
                    question,
                    "unusable_question",
                    usability["reason"],
                    usability.get("revision_guidance", ""),
                )
                state["revision_feedback_by_plan_id"][plan_id] = _feedback_text(
                    "Previous question was not usable.",
                    usability["reason"],
                    usability.get("revision_guidance", ""),
                )
                _write_json(cache_path, state)
                log("challenge question unusable", plan_id=plan_id, question_id=question_id)
                continue

            trial_bundle = run_single_challenge_question_trials(question, client, paper_text)
            state["solver_trials"].append(trial_bundle)
            if trial_bundle["matched_target_failure_count"] > 0:
                accepted = question_with_filter_metadata(question, trial_bundle)
                accepted["accepted_by_challenge_loop"] = True
                accepted["filter_reason"] = "matched_target_failure"
                if trial_bundle["wrong_count"] == trial_bundle["solver_count"]:
                    accepted["needs_human_review"] = True
                    accepted["all_solvers_failed"] = True
                    state["human_review_questions"].append(dict(accepted))
                state["accepted_questions"].append(accepted)
                state["used_plan_ids"].append(plan_id)
                state["loop_events"].append(
                    {
                        "event": "accepted",
                        "plan_id": plan_id,
                        "question_id": question_id,
                        "attempt": attempt,
                        "matched_target_failure_count": trial_bundle["matched_target_failure_count"],
                    }
                )
                _write_json(cache_path, state)
                log("challenge question accepted", plan_id=plan_id, question_id=question_id)
                continue

            easiness = _judge_plan_easiness(plan, question, trial_bundle, client, easiness_tpl)
            question["easiness_check"] = easiness
            if easiness["plan_too_easy"]:
                state["blacklisted_plan_ids"].append(plan_id)
                _record_rejection(
                    state,
                    question,
                    "plan_too_easy",
                    easiness["reason"],
                    "",
                )
                state["loop_events"].append(
                    {
                        "event": "plan_blacklisted",
                        "plan_id": plan_id,
                        "question_id": question_id,
                        "attempt": attempt,
                        "reason": easiness["reason"],
                    }
                )
                _write_json(cache_path, state)
                log("challenge plan blacklisted", plan_id=plan_id, question_id=question_id)
                continue

            _record_rejection(
                state,
                question,
                "too_easy_question",
                easiness["reason"],
                easiness.get("revision_guidance", ""),
            )
            state["revision_feedback_by_plan_id"][plan_id] = _feedback_text(
                "Previous question was too easy.",
                easiness["reason"],
                easiness.get("revision_guidance", ""),
            )
            state["loop_events"].append(
                {
                    "event": "regenerate_question",
                    "plan_id": plan_id,
                    "question_id": question_id,
                    "attempt": attempt,
                    "reason": easiness["reason"],
                }
            )
            _write_json(cache_path, state)
            log("challenge question too easy; regenerating", plan_id=plan_id, question_id=question_id)

    return _result_from_state(challenge_plans, state, solver_configs)


def _judge_question_usability(plan: dict, question: dict, client: OpenAICompatClient, tpl: str) -> dict:
    payload = {
        "challenge_plan": _prompt_plan(plan),
        "question": {
            "question_id": question.get("question_id"),
            "question_text": question.get("question_text"),
            "surface_intent": question.get("surface_intent"),
        },
    }
    result = client.chat_json(
        system_prompt="You validate challenge questions for paper evaluation. Return JSON only.",
        user_prompt=render_prompt(
            tpl,
            usability_check_json=json.dumps(payload, ensure_ascii=False, indent=2),
        ),
        temperature=_env_float("CHALLENGE_META_JUDGE_TEMPERATURE", 0.1),
    )
    if not isinstance(result, dict) or "usable" not in result:
        raise ValueError("Challenge usability judge must return an object with usable.")
    return {
        "usable": bool(result["usable"]),
        "reason": _non_empty_text(result.get("reason"), "Challenge usability judge returned empty reason."),
        "revision_guidance": str(result.get("revision_guidance", "")).strip(),
    }


def _judge_plan_easiness(
    plan: dict,
    question: dict,
    trial_bundle: dict,
    client: OpenAICompatClient,
    tpl: str,
) -> dict:
    payload = {
        "challenge_plan": _prompt_plan(plan),
        "question": {
            "question_id": question.get("question_id"),
            "question_text": question.get("question_text"),
            "surface_intent": question.get("surface_intent"),
        },
        "solver_trial": trial_bundle,
    }
    result = client.chat_json(
        system_prompt="You diagnose failed challenge-question attempts for paper evaluation. Return JSON only.",
        user_prompt=render_prompt(
            tpl,
            easiness_check_json=json.dumps(payload, ensure_ascii=False, indent=2),
        ),
        temperature=_env_float("CHALLENGE_META_JUDGE_TEMPERATURE", 0.1),
    )
    if not isinstance(result, dict) or "plan_too_easy" not in result:
        raise ValueError("Challenge easiness judge must return an object with plan_too_easy.")
    return {
        "plan_too_easy": bool(result["plan_too_easy"]),
        "reason": _non_empty_text(result.get("reason"), "Challenge easiness judge returned empty reason."),
        "revision_guidance": str(result.get("revision_guidance", "")).strip(),
    }


def _next_plan(state: dict, plan_by_id: dict[str, dict], max_attempts_per_plan: int) -> dict | None:
    blacklisted = set(state["blacklisted_plan_ids"])
    accepted_plan_ids = set(state["used_plan_ids"])
    for plan_id in state["plan_order"]:
        if plan_id in blacklisted or plan_id in accepted_plan_ids:
            continue
        if int(state["attempts_by_plan_id"].get(plan_id, 0)) >= max_attempts_per_plan:
            continue
        plan = plan_by_id.get(plan_id)
        if plan is None:
            raise ValueError(f"Challenge loop state references missing plan_id={plan_id}.")
        return plan
    return None


def _initial_state(
    challenge_plans: dict,
    signature: dict,
    plans: list[dict],
    target_count: int,
    max_attempts_per_plan: int,
) -> dict:
    return {
        "paper_id": challenge_plans.get("paper_id", "unknown"),
        "schema_version": "v2",
        "challenge_loop_signature": signature,
        "target_count": target_count,
        "max_attempts_per_plan": max_attempts_per_plan,
        "plan_order": _random_plan_order(plans),
        "next_question_index": 1,
        "attempts_by_plan_id": {},
        "revision_feedback_by_plan_id": {},
        "used_plan_ids": [],
        "blacklisted_plan_ids": [],
        "raw_questions": [],
        "accepted_questions": [],
        "human_review_questions": [],
        "rejected_questions": [],
        "solver_trials": [],
        "loop_events": [],
    }


def _random_plan_order(plans: list[dict]) -> list[str]:
    plan_ids = [plan["challenge_plan_id"] for plan in plans]
    seed = os.getenv("CHALLENGE_RANDOM_SEED", "").strip()
    rng = random.Random(seed) if seed else random.SystemRandom()
    rng.shuffle(plan_ids)
    return plan_ids


def _result_from_state(challenge_plans: dict, state: dict, solver_configs: list[dict]) -> dict:
    raw_questions = state["raw_questions"]
    accepted = state["accepted_questions"]
    rejected = state["rejected_questions"]
    return {
        "paper_id": challenge_plans.get("paper_id", "unknown"),
        "schema_version": "v2",
        "source_challenge_plan_signature": _challenge_plan_signature(challenge_plans),
        "challenge_loop_signature": state["challenge_loop_signature"],
        "solver_configs": solver_configs,
        "plan_order": state["plan_order"],
        "blacklisted_plan_ids": state["blacklisted_plan_ids"],
        "challenge_questions_raw": raw_questions,
        "solver_trials": state["solver_trials"],
        "challenge_questions_filtered": accepted,
        "challenge_questions_need_human_review": state["human_review_questions"],
        "challenge_questions_rejected": rejected,
        "loop_events": state["loop_events"],
        "summary": {
            "plan_pool_count": len(challenge_plans.get("challenge_plans", [])),
            "target_count": state["target_count"],
            "max_attempts_per_plan": state["max_attempts_per_plan"],
            "raw_question_count": len(raw_questions),
            "solver_trial_count": len(state["solver_trials"]),
            "filtered_count": len(accepted),
            "human_review_count": len(state["human_review_questions"]),
            "rejected_count": len(rejected),
            "blacklisted_plan_count": len(state["blacklisted_plan_ids"]),
            "by_type": _count_by_type(accepted),
            "stop_reason": _stop_reason(challenge_plans, state),
        },
    }


def _stop_reason(challenge_plans: dict, state: dict) -> str:
    if len(state["accepted_questions"]) >= int(state["target_count"]):
        return "target_reached"
    available = set(plan["challenge_plan_id"] for plan in challenge_plans.get("challenge_plans", []))
    unavailable = set(state["used_plan_ids"]) | set(state["blacklisted_plan_ids"])
    exhausted = {
        plan_id
        for plan_id, attempts in state["attempts_by_plan_id"].items()
        if int(attempts) >= int(state["max_attempts_per_plan"])
    }
    if available <= (unavailable | exhausted):
        return "plan_pool_exhausted"
    return "stopped"


def _record_rejection(
    state: dict,
    question: dict,
    reason_code: str,
    reason: str,
    revision_guidance: str,
) -> None:
    item = dict(question)
    item["filter_reason"] = reason_code
    item["rejection_reason"] = reason
    item["revision_guidance"] = revision_guidance
    state["rejected_questions"].append(item)


def _feedback_text(prefix: str, reason: str, revision_guidance: str) -> str:
    parts = [prefix, f"Reason: {reason}"]
    if revision_guidance.strip():
        parts.append(f"Revision guidance: {revision_guidance.strip()}")
    return " ".join(parts)


def _loop_signature(
    challenge_plans: dict,
    solver_configs: list[dict],
    paper_text: str,
    target_count: int,
    max_attempts_per_plan: int,
) -> dict:
    return {
        "source_challenge_plan_signature": _challenge_plan_signature(challenge_plans),
        "solver_configs": solver_configs,
        "paper_text_sha256": hashlib.sha256(paper_text.encode("utf-8")).hexdigest(),
        "target_count": target_count,
        "max_attempts_per_plan": max_attempts_per_plan,
    }


def _challenge_plan_signature(challenge_plans: dict) -> dict:
    plans = challenge_plans.get("challenge_plans", [])
    return {
        "paper_id": challenge_plans.get("paper_id", "unknown"),
        "schema_version": challenge_plans.get("schema_version"),
        "source_graph_signature": challenge_plans.get("source_graph_signature"),
        "plan_ids": [plan.get("challenge_plan_id") for plan in plans],
        "plan_sources": [
            [
                plan.get("challenge_plan_id"),
                plan.get("challenge_type"),
                plan.get("target_failure_mode"),
                plan.get("source", {}).get("kc_ids", []),
                plan.get("source", {}).get("edge_ids", []),
                plan.get("trap_part", ""),
            ]
            for plan in plans
        ],
    }


def _prompt_plan(plan: dict) -> dict:
    return {
        "challenge_plan_id": plan.get("challenge_plan_id"),
        "challenge_type": plan.get("challenge_type"),
        "source": plan.get("source", {}),
        "true_part": plan.get("true_part"),
        "trap_part": plan.get("trap_part"),
        "expected_behavior": plan.get("expected_behavior"),
        "target_failure_mode": plan.get("target_failure_mode"),
        "evidence": plan.get("evidence", []),
        "metadata": plan.get("metadata", {}),
    }


def _plan_by_id(plans: list[dict]) -> dict[str, dict]:
    out = {}
    for plan in plans:
        plan_id = str(plan.get("challenge_plan_id", "")).strip()
        if not plan_id:
            raise ValueError("Every Challenge Plan must include challenge_plan_id.")
        if plan_id in out:
            raise ValueError(f"Duplicate challenge_plan_id={plan_id}.")
        out[plan_id] = plan
    return out


def _count_by_type(questions: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for question in questions:
        challenge_type = question.get("challenge_type", "unknown")
        counts[challenge_type] = counts.get(challenge_type, 0) + 1
    return dict(sorted(counts.items()))


def _non_empty_text(value: object, error: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(error)
    return text


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


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a float, got {raw!r}.") from exc
    if value < 0 or value > 2:
        raise ValueError(f"{name} must be in [0, 2], got {value}.")
    return value


def _load_cache(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Cannot read challenge loop cache {path}: {type(exc).__name__}: {exc}") from exc


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)
