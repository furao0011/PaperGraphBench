from __future__ import annotations

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from src.model_client import OpenAICompatClient
from src.progress import log, span
from src.prompt_loader import load_prompt, render_prompt


INTERNAL_LABELS = {
    "overclaim_challenge",
    "wrong_relation_challenge",
    "false_premise_challenge",
    "thread_wrong_bridge_challenge",
    "thread_overclaim_challenge",
    "thread_premise_mutation_challenge",
    "overclaim",
    "wrong_relation",
    "false_premise",
    "thread_wrong_bridge",
    "thread_overclaim",
    "thread_premise_mutation",
    "target_failure_mode",
    "challenge_type",
    "expected_behavior",
    "true_part",
    "trap_part",
}


def generate_raw_challenge_questions(
    challenge_plans: dict,
    client: OpenAICompatClient,
    cache_path: Path,
    resume: bool = False,
    restart: bool = False,
) -> dict:
    if not client or not client.is_ready():
        raise RuntimeError("Raw challenge question generation requires a configured online model client.")
    plans = challenge_plans.get("challenge_plans", [])
    if not isinstance(plans, list) or not plans:
        raise ValueError("Raw challenge question generation requires non-empty challenge_plans.")

    signature = _challenge_plan_signature(challenge_plans)
    cache = _load_cache(cache_path) if resume and not restart else {}
    if cache.get("challenge_plan_signature") != signature:
        cache = {
            "paper_id": challenge_plans.get("paper_id", "unknown"),
            "challenge_plan_signature": signature,
            "questions_by_plan_id": {},
        }
        _write_json(cache_path, cache)
    cache.setdefault("questions_by_plan_id", {})

    max_workers = min(_env_positive_int("CHALLENGE_QUESTION_WORKERS", 4), len(plans))
    completed: dict[str, dict] = {}
    futures = {}
    errors: list[str] = []

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for plan in plans:
            plan_id = str(plan.get("challenge_plan_id", "")).strip()
            if not plan_id:
                raise ValueError("Every Challenge Plan must include challenge_plan_id.")
            cached = cache["questions_by_plan_id"].get(plan_id)
            if cached:
                try:
                    completed[plan_id] = _normalize_question(plan, cached, 0)
                    log("challenge question cache hit", plan_id=plan_id)
                    continue
                except Exception:
                    cache["questions_by_plan_id"].pop(plan_id, None)
                    _write_json(cache_path, cache)
            futures[ex.submit(_generate_one, plan, client, _generation_prompt_for_plan(plan))] = plan

        for fut in as_completed(futures):
            plan = futures[fut]
            plan_id = plan.get("challenge_plan_id")
            try:
                question = fut.result()
                cache["questions_by_plan_id"][plan_id] = question
                _write_json(cache_path, cache)
                completed[plan_id] = question
                log("challenge question generated", plan_id=plan_id)
            except Exception as exc:
                errors.append(f"{plan_id}: {type(exc).__name__}: {exc}")
                log("challenge question generation error", plan_id=plan_id, error=f"{type(exc).__name__}: {exc}")

    if errors:
        raise RuntimeError("Raw challenge question generation failed: " + "; ".join(errors[:5]))

    questions = []
    for idx, plan in enumerate(plans, start=1):
        plan_id = plan["challenge_plan_id"]
        if plan_id not in completed:
            raise RuntimeError(f"Missing generated challenge question for plan_id={plan_id}.")
        questions.append(_normalize_question(plan, completed[plan_id], idx))

    return {
        "paper_id": challenge_plans.get("paper_id", "unknown"),
        "schema_version": "v2",
        "source_challenge_plan_signature": signature,
        "challenge_questions_raw": questions,
        "summary": {
            "raw_question_count": len(questions),
            "by_type": _count_by_type(questions),
        },
    }


def generate_challenge_question_for_plan(
    plan: dict,
    client: OpenAICompatClient,
    question_id: str,
    revision_feedback: str = "",
) -> dict:
    if not client or not client.is_ready():
        raise RuntimeError("Challenge question generation requires a configured online model client.")
    if not str(question_id).strip():
        raise ValueError("generate_challenge_question_for_plan requires a non-empty question_id.")
    tpl = load_prompt("generate_challenge_question.txt")
    raw = _generate_one(plan, client, tpl, revision_feedback=revision_feedback)
    raw["question_id"] = question_id
    return _normalize_question(plan, raw, 0)


def _generate_one(
    plan: dict,
    client: OpenAICompatClient,
    tpl: str,
    revision_feedback: str = "",
) -> dict:
    plan_id = plan["challenge_plan_id"]
    prompt_plan = _prompt_plan(plan)
    user_prompt = render_prompt(
        tpl,
        challenge_plan_json=json.dumps(prompt_plan, ensure_ascii=False, indent=2),
        revision_feedback=revision_feedback.strip() or "None.",
    )
    errors = []
    with span("generate raw challenge question", plan_id=plan_id, challenge_type=plan.get("challenge_type")):
        for attempt in range(1, 3):
            prompt = user_prompt
            if errors:
                prompt = (
                    user_prompt
                    + "\n\nYour previous response failed validation:\n"
                    + errors[-1]
                    + "\nRewrite the question without exposing KC IDs, edge IDs, challenge plan IDs, or internal labels. Return strict JSON only."
                )
            try:
                result = client.chat_json(
                    system_prompt="You generate natural Storybench-evaluation challenge questions. Return JSON only.",
                    user_prompt=prompt,
                    temperature=0.0 if attempt == 2 else 0.2,
                )
                return _normalize_question(plan, result, 0)
            except Exception as exc:
                errors.append(f"{type(exc).__name__}: {exc}")
                if attempt == 1:
                    continue
                raise
    raise RuntimeError(f"Challenge question generation failed for {plan_id}.")


def _prompt_plan(plan: dict) -> dict:
    return {
        "challenge_plan_id": plan.get("challenge_plan_id"),
        "challenge_scope": plan.get("challenge_scope", "macro"),
        "challenge_type": plan.get("challenge_type"),
        "thread_id": plan.get("thread_id"),
        "thread_type": plan.get("thread_type"),
        "preferred_insert_after_step": plan.get("preferred_insert_after_step"),
        "source": plan.get("source", {}),
        "canonical_thread_context": plan.get("canonical_thread_context", {}),
        "true_part": plan.get("true_part"),
        "trap_part": plan.get("trap_part"),
        "expected_behavior": plan.get("expected_behavior"),
        "target_failure_mode": plan.get("target_failure_mode"),
        "evidence": plan.get("evidence", []),
        "metadata": plan.get("metadata", {}),
        "modality_pool": plan.get("modality_pool", plan.get("metadata", {}).get("modality_pool", "text")),
        "asset_references": plan.get("asset_references") or plan.get("metadata", {}).get("asset_references", []),
    }


def _normalize_question(plan: dict, raw: dict, ordinal: int) -> dict:
    if not isinstance(raw, dict):
        raise ValueError(f"Challenge question for {plan.get('challenge_plan_id')} must be an object.")
    question_text = str(raw.get("question_text", "")).strip()
    if not question_text:
        raise ValueError(f"Challenge question for {plan.get('challenge_plan_id')} has empty question_text.")
    _assert_no_internal_labels(question_text, plan)
    source = plan.get("source", {})
    question_id = f"CHQ_{ordinal:04d}" if ordinal else str(raw.get("question_id") or "").strip()
    challenge_scope = str(plan.get("challenge_scope") or "macro").strip()
    asset_references = plan.get("asset_references") or plan.get("metadata", {}).get("asset_references", [])
    return {
        "question_id": question_id,
        "question_type": "thread_challenge_question" if challenge_scope == "thread" else "challenge_question",
        "source_plan_id": plan.get("challenge_plan_id"),
        "source_challenge_plan_id": plan.get("challenge_plan_id"),
        "challenge_scope": challenge_scope,
        "challenge_type": plan.get("challenge_type"),
        "question_text": question_text,
        "surface_intent": str(raw.get("surface_intent", "")).strip(),
        "target_kc_ids": source.get("kc_ids", []),
        "target_edge_ids": source.get("edge_ids", []),
        "target_thread_id": source.get("thread_id"),
        "target_thread_turn_id": source.get("thread_turn_id"),
        "thread_id": source.get("thread_id") if challenge_scope == "thread" else None,
        "thread_turn_id": source.get("thread_turn_id") if challenge_scope == "thread" else None,
        "thread_role": "thread_challenge" if challenge_scope == "thread" else None,
        "insert_after_step": plan.get("preferred_insert_after_step"),
        "canonical_thread_context": plan.get("canonical_thread_context", {}),
        "synthetic_thread_history": plan.get("metadata", {}).get("synthetic_thread_history", {}),
        "target_macro_ids": source.get("macro_ids", []),
        "target_asset_ids": source.get("asset_ids", []),
        "modality_pool": plan.get("modality_pool", plan.get("metadata", {}).get("modality_pool", "text")),
        "requires_multimodal_input": bool(asset_references),
        "asset_references": asset_references,
        "expected_behavior": plan.get("expected_behavior", ""),
        "target_failure_mode": plan.get("target_failure_mode", ""),
        "evidence": plan.get("evidence", []),
    }


def _generation_prompt_for_plan(plan: dict) -> str:
    if str(plan.get("challenge_scope") or "").strip() == "thread":
        return load_prompt("generate_thread_challenge_question.txt")
    return load_prompt("generate_challenge_question.txt")


def _assert_no_internal_labels(question_text: str, plan: dict) -> None:
    lower = question_text.lower()
    leaked = [label for label in INTERNAL_LABELS if label.lower() in lower]
    leaked.extend(re.findall(r"\bKC\d+\b|\bE\d+\b|\bCHP_\d+\b|\bCHQ_\d+\b|RT\d+(?:_STEP\d+)?", question_text))
    plan_id = str(plan.get("challenge_plan_id", "")).strip()
    if plan_id and plan_id in question_text:
        leaked.append(plan_id)
    if leaked:
        raise ValueError(
            f"Challenge question for {plan.get('challenge_plan_id')} leaks internal labels: {sorted(set(leaked))}"
        )


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
                plan.get("challenge_scope", "macro"),
                plan.get("challenge_type"),
                plan.get("target_failure_mode"),
                plan.get("thread_id"),
                plan.get("preferred_insert_after_step"),
                plan.get("source", {}).get("kc_ids", []),
                plan.get("source", {}).get("edge_ids", []),
                plan.get("source", {}).get("asset_ids", []),
                plan.get("modality_pool", plan.get("metadata", {}).get("modality_pool", "text")),
                plan.get("trap_part", ""),
            ]
            for plan in plans
        ],
    }


def _count_by_type(questions: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for question in questions:
        challenge_type = question.get("challenge_type", "unknown")
        counts[challenge_type] = counts.get(challenge_type, 0) + 1
    return dict(sorted(counts.items()))


def _load_cache(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Cannot read challenge question cache {path}: {type(exc).__name__}: {exc}") from exc


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


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
