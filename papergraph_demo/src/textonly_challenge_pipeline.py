from __future__ import annotations

import hashlib
import json
from pathlib import Path

from src.challenge_loop import build_challenge_questions_loop
from src.json_io import write_json_atomic
from src.model_client import OpenAICompatClient
from src.progress import log
from src.textonly_benchmark import CHALLENGE_TYPES, normalize_textonly_package


FAILURE_MODE_BY_TYPE = {
    "false_premise": "false_premise",
    "overclaim": "overclaim",
    "wrong_relation": "wrong_relation",
    "unsupported_generalization": "overclaim",
}


def build_filtered_textonly_package(
    *,
    paper_id: str,
    paper_text: str,
    multimodal_assets: list[dict],
    text_client: OpenAICompatClient,
    vision_client: OpenAICompatClient | None,
    cache_dir: Path,
    macro_count: int,
    text_plan_count: int,
    multimodal_plan_count: int,
    text_accept_count: int,
    multimodal_accept_count: int,
    temperature: float,
    restart: bool,
    generation_attempts: int = 3,
) -> dict:
    cache_dir.mkdir(parents=True, exist_ok=True)
    signature = _source_signature(
        paper_id,
        paper_text,
        multimodal_assets,
        macro_count,
        text_plan_count,
        multimodal_plan_count,
        text_client.cfg.llm_model,
        temperature,
        generation_attempts,
    )
    candidates = _load_or_generate_candidates(
        cache_dir / "generation_candidates.json",
        signature=signature,
        restart=restart,
        generator=lambda: _generate_candidates(
            paper_id=paper_id,
            paper_text=paper_text,
            multimodal_assets=multimodal_assets,
            client=text_client,
            macro_count=macro_count,
            text_plan_count=text_plan_count,
            multimodal_plan_count=multimodal_plan_count,
            temperature=temperature,
            generation_attempts=generation_attempts,
        ),
    )
    _refresh_multimodal_plan_asset_references(candidates["multimodal_challenge_plans"], multimodal_assets)
    text_plans = _plan_payload(paper_id, candidates["text_challenge_plans"], "text")
    multimodal_plans = _plan_payload(
        paper_id,
        candidates["multimodal_challenge_plans"],
        "multimodal",
    )
    _write_json(
        cache_dir / "challenge_plans.json",
        {
            "paper_id": paper_id,
            "schema_version": "textonly-v2",
            "text": text_plans,
            "multimodal": multimodal_plans,
        },
    )

    text_result = build_challenge_questions_loop(
        challenge_plans=text_plans,
        client=text_client,
        paper_text=paper_text,
        cache_path=cache_dir / "challenge_loop_text.json",
        resume=not restart,
        restart=restart,
        target_count=text_accept_count,
        question_id_prefix="TXT_CHQ",
    )
    if multimodal_accept_count:
        if vision_client is None or not vision_client.is_ready():
            raise RuntimeError("Multimodal no-graph challenge filtering requires the configured vision client.")
        multimodal_result = build_challenge_questions_loop(
            challenge_plans=multimodal_plans,
            client=text_client,
            paper_text=paper_text,
            cache_path=cache_dir / "challenge_loop_multimodal.json",
            resume=not restart,
            restart=restart,
            target_count=multimodal_accept_count,
            solver_client=vision_client,
            question_id_prefix="TXT_CHQM",
        )
    else:
        multimodal_result = _empty_loop_result(paper_id)

    _require_target(text_result, text_accept_count, "text")
    _require_target(multimodal_result, multimodal_accept_count, "multimodal")
    raw_text = _accepted_to_textonly_questions(text_result, text_plans, multimodal=False)
    raw_multimodal = _accepted_to_textonly_questions(
        multimodal_result,
        multimodal_plans,
        multimodal=True,
    )
    package = normalize_textonly_package(
        {
            "macro_questions": candidates["macro_questions"],
            "challenge_questions": raw_text,
            "multimodal_challenge_questions": raw_multimodal,
        },
        paper_id=paper_id,
        macro_count=macro_count,
        challenge_count=text_accept_count,
        multimodal_challenge_count=multimodal_accept_count,
        multimodal_assets=multimodal_assets,
    )
    _attach_filter_metadata(package["challenge_questions"], text_result)
    _attach_filter_metadata(package["multimodal_challenge_questions"], multimodal_result)
    package["challenge_pipeline"] = {
        "mode": "plan_solver_filter",
        "text": text_result.get("summary", {}),
        "multimodal": multimodal_result.get("summary", {}),
    }
    _write_pipeline_artifacts(cache_dir, paper_id, text_result, multimodal_result)
    return package


def _generate_candidates(
    *,
    paper_id: str,
    paper_text: str,
    multimodal_assets: list[dict],
    client: OpenAICompatClient,
    macro_count: int,
    text_plan_count: int,
    multimodal_plan_count: int,
    temperature: float,
    generation_attempts: int,
) -> dict:
    macros = _generate_validated_json(
        client=client,
        component="macro questions",
        system_prompt=(
            "Create main paper-reading questions directly from the supplied paper. "
            "Do not use or invent graph, KC, edge, macro-id, or thread metadata. Return JSON only."
        ),
        base_prompt=_macro_prompt(paper_id, paper_text, macro_count),
        temperature=temperature,
        attempts=generation_attempts,
        validator=lambda payload: _normalize_macro_candidates(payload, macro_count),
    )
    text_plans = _generate_validated_json(
        client=client,
        component="text challenge plans",
        system_prompt=(
            "Create challenge plans directly from the supplied paper. Plans describe traps, not final questions. "
            "Do not use graph, KC, edge, or thread metadata. Return JSON only."
        ),
        base_prompt=_plan_prompt(paper_id, paper_text, [], text_plan_count, multimodal=False),
        temperature=temperature,
        attempts=generation_attempts,
        validator=lambda payload: _normalize_plan_candidates(
            payload,
            text_plan_count,
            multimodal=False,
            multimodal_assets=[],
            id_prefix="TXT_PLAN_T",
        ),
    )
    if multimodal_plan_count:
        if not multimodal_assets:
            raise RuntimeError("Multimodal challenge plans requested but no usable multimodal assets exist.")
        multimodal_plans = _generate_multimodal_plans(
            client=client,
            system_prompt=(
                "Create multimodal challenge plans directly from the supplied paper and asset summaries. "
                "Every plan must bind one or two listed assets and explain its multimodal dependency. "
                "Plans describe traps, not final questions. Return JSON only."
            ),
            base_prompt=_plan_prompt(
                paper_id,
                paper_text,
                multimodal_assets,
                multimodal_plan_count,
                multimodal=True,
            ),
            temperature=temperature,
            attempts=generation_attempts,
            expected_count=multimodal_plan_count,
            multimodal_assets=multimodal_assets,
        )
    else:
        multimodal_plans = []
    return {
        "macro_questions": macros,
        "text_challenge_plans": text_plans,
        "multimodal_challenge_plans": multimodal_plans,
    }


def _generate_validated_json(
    *,
    client: OpenAICompatClient,
    component: str,
    system_prompt: str,
    base_prompt: str,
    temperature: float,
    attempts: int,
    validator,
):
    if attempts <= 0:
        raise ValueError("generation_attempts must be positive.")
    validation_error = ""
    for attempt in range(1, attempts + 1):
        prompt = base_prompt
        if validation_error:
            prompt += (
                "\n\nYour previous response failed strict schema validation:\n"
                f"{validation_error}\n"
                "Return a complete corrected replacement JSON object. "
                "Do not omit any requested item or field."
            )
        payload = client.chat_json(
            system_prompt=system_prompt,
            user_prompt=prompt,
            temperature=temperature,
        )
        try:
            return validator(payload)
        except (KeyError, TypeError, ValueError) as exc:
            validation_error = f"{type(exc).__name__}: {exc}"
            log(
                "text-only generation schema retry",
                component=component,
                attempt=attempt,
                attempts=attempts,
                error=validation_error,
            )
    raise RuntimeError(
        f"{component} remained invalid after {attempts} generation attempts: {validation_error}"
    )

def _generate_multimodal_plans(
    *,
    client: OpenAICompatClient,
    system_prompt: str,
    base_prompt: str,
    temperature: float,
    attempts: int,
    expected_count: int,
    multimodal_assets: list[dict],
) -> list[dict]:
    if attempts <= 0:
        raise ValueError("generation_attempts must be positive.")
    payload = client.chat_json(
        system_prompt=system_prompt,
        user_prompt=base_prompt,
        temperature=temperature,
    )
    raw_items = payload.get("challenge_plans") if isinstance(payload, dict) else None
    working = list(raw_items[:expected_count]) if isinstance(raw_items, list) else []
    working.extend([None] * (expected_count - len(working)))
    errors: dict[int, str] = {}

    for attempt in range(1, attempts + 1):
        errors = _invalid_multimodal_plan_errors(working, multimodal_assets)
        if not errors:
            return _normalize_plan_candidates(
                {"challenge_plans": working},
                expected_count,
                multimodal=True,
                multimodal_assets=multimodal_assets,
                id_prefix="TXT_PLAN_M",
            )
        error_text = "; ".join(f"#{position}: {error}" for position, error in errors.items())
        log(
            "text-only generation schema retry",
            component="multimodal challenge plans",
            attempt=attempt,
            attempts=attempts,
            invalid_positions=list(errors),
            error=error_text,
        )
        if attempt == attempts:
            break
        repair_payload = client.chat_json(
            system_prompt=system_prompt,
            user_prompt=_multimodal_plan_repair_prompt(base_prompt, working, errors),
            temperature=temperature,
        )
        repairs = repair_payload.get("challenge_plans") if isinstance(repair_payload, dict) else None
        repairs = repairs if isinstance(repairs, list) else []
        for repair_index, position in enumerate(errors):
            if repair_index < len(repairs):
                working[position - 1] = repairs[repair_index]

    error_text = "; ".join(f"#{position}: {error}" for position, error in errors.items())
    raise RuntimeError(
        f"multimodal challenge plans remained invalid after {attempts} generation attempts: {error_text}"
    )


def _invalid_multimodal_plan_errors(
    plans: list[object],
    multimodal_assets: list[dict],
) -> dict[int, str]:
    errors: dict[int, str] = {}
    for position, plan in enumerate(plans, start=1):
        try:
            _normalize_plan_candidates(
                {"challenge_plans": [plan]},
                1,
                multimodal=True,
                multimodal_assets=multimodal_assets,
                id_prefix="TXT_PLAN_CHECK",
            )
        except (KeyError, TypeError, ValueError) as exc:
            errors[position] = f"{type(exc).__name__}: {exc}"
    return errors


def _multimodal_plan_repair_prompt(
    base_prompt: str,
    plans: list[object],
    errors: dict[int, str],
) -> str:
    invalid_items = [
        {
            "position_in_original_list": position,
            "validation_error": errors[position],
            "invalid_plan": plans[position - 1],
        }
        for position in errors
    ]
    return (
        f"{base_prompt}\n\n"
        "REPAIR TASK: override the earlier output-count instruction for this response.\n"
        f"Only repair original positions {list(errors)}. Keep their order. "
        f"Return exactly {len(errors)} replacement objects as "
        "{\"challenge_plans\": [...]}; do not return the already-valid plans.\n"
        "Each replacement must include a valid non-empty asset_ids list and a non-empty "
        "multimodal_dependency, as well as every common required field.\n"
        "Invalid entries and exact validation errors:\n"
        f"{json.dumps(invalid_items, ensure_ascii=False, indent=2)}"
    )

def _normalize_macro_candidates(payload: dict, expected_count: int) -> list[dict]:
    items = payload.get("macro_questions") if isinstance(payload, dict) else None
    if not isinstance(items, list) or len(items) < expected_count:
        raise ValueError(f"Expected at least {expected_count} no-graph macro questions.")
    return items[:expected_count]


def _normalize_plan_candidates(
    payload: dict,
    expected_count: int,
    *,
    multimodal: bool,
    multimodal_assets: list[dict],
    id_prefix: str,
) -> list[dict]:
    items = payload.get("challenge_plans") if isinstance(payload, dict) else None
    if not isinstance(items, list) or len(items) < expected_count:
        raise ValueError(f"Expected at least {expected_count} {'multimodal' if multimodal else 'text'} challenge plans.")
    asset_index = {str(asset.get("asset_id")): asset for asset in multimodal_assets if asset.get("asset_id")}
    plans = []
    for index, raw in enumerate(items[:expected_count], start=1):
        if not isinstance(raw, dict):
            raise ValueError(f"Challenge plan #{index} must be an object.")
        challenge_type = _required_text(raw, "challenge_type", index).lower()
        if challenge_type not in CHALLENGE_TYPES:
            raise ValueError(f"Challenge plan #{index} has invalid challenge_type={challenge_type!r}.")
        asset_ids = _asset_ids(raw, asset_index, index) if multimodal else []
        asset_references = [asset_index[asset_id] for asset_id in asset_ids]
        multimodal_dependency = _required_text(raw, "multimodal_dependency", index) if multimodal else ""
        evidence = raw.get("evidence")
        if isinstance(evidence, str):
            evidence = [evidence]
        if not isinstance(evidence, list) or not any(str(item).strip() for item in evidence):
            raise ValueError(f"Challenge plan #{index} requires non-empty evidence.")
        plans.append(
            {
                "challenge_plan_id": f"{id_prefix}_{index:04d}",
                "challenge_scope": "macro",
                "challenge_type": challenge_type,
                "source": {
                    "kc_ids": [],
                    "edge_ids": [],
                    "macro_ids": [],
                    "asset_ids": asset_ids,
                },
                "true_part": _required_text(raw, "true_part", index),
                "trap_part": _required_text(raw, "trap_part", index),
                "expected_behavior": _required_text(raw, "expected_behavior", index),
                "target_failure_mode": FAILURE_MODE_BY_TYPE[challenge_type],
                "evidence": [str(item).strip() for item in evidence if str(item).strip()],
                "modality_pool": "multimodal" if multimodal else "text",
                "asset_references": asset_references,
                "metadata": {
                    "no_graph": True,
                    "modality_pool": "multimodal" if multimodal else "text",
                    "asset_references": asset_references,
                    "multimodal_dependency": multimodal_dependency,
                },
            }
        )
    return plans


def _refresh_multimodal_plan_asset_references(plans: list[dict], multimodal_assets: list[dict]) -> None:
    asset_index = {str(asset.get("asset_id")): asset for asset in multimodal_assets if asset.get("asset_id")}
    for plan in plans:
        if not isinstance(plan, dict):
            continue
        source = plan.get("source")
        if not isinstance(source, dict):
            continue
        raw_ids = source.get("asset_ids")
        if not isinstance(raw_ids, list) or not raw_ids:
            continue
        asset_ids = [str(asset_id).strip() for asset_id in raw_ids if str(asset_id).strip()]
        missing = [asset_id for asset_id in asset_ids if asset_id not in asset_index]
        if missing:
            raise ValueError(
                f"Cached multimodal challenge plan {plan.get('challenge_plan_id')} references unknown asset_ids={missing!r}."
            )
        refs = [asset_index[asset_id] for asset_id in asset_ids]
        plan["asset_references"] = refs
        metadata = plan.setdefault("metadata", {})
        if isinstance(metadata, dict):
            metadata["asset_references"] = refs

def _plan_payload(paper_id: str, plans: list[dict], modality_pool: str) -> dict:
    return {
        "paper_id": paper_id,
        "schema_version": "textonly-v2",
        "plan_builder": "paper_direct_no_graph",
        "modality_pool": modality_pool,
        "challenge_plans": plans,
        "summary": {"plan_count": len(plans), "modality_pool": modality_pool},
    }


def _accepted_to_textonly_questions(loop_result: dict, plan_payload: dict, *, multimodal: bool) -> list[dict]:
    plans = {plan["challenge_plan_id"]: plan for plan in plan_payload.get("challenge_plans", [])}
    questions = []
    for accepted in loop_result.get("challenge_questions_filtered", []):
        plan_id = str(accepted.get("source_plan_id") or accepted.get("source_challenge_plan_id") or "")
        plan = plans.get(plan_id)
        if plan is None:
            raise ValueError(f"Accepted challenge references unknown plan_id={plan_id!r}.")
        evidence = "\n".join(str(item) for item in plan.get("evidence", []) if str(item).strip())
        item = {
            "challenge_type": plan["challenge_type"],
            "question_text": accepted["question_text"],
            "forbidden_claim": plan["trap_part"],
            "expected_behavior": plan["expected_behavior"],
            "expected_answer": plan["true_part"],
            "evidence": evidence,
        }
        if multimodal:
            item["asset_ids"] = plan.get("source", {}).get("asset_ids", [])
            item["multimodal_dependency"] = plan.get("metadata", {}).get("multimodal_dependency") or (
                "The cited figure or table is required to verify the trap against the paper evidence."
            )
        questions.append(item)
    return questions


def _attach_filter_metadata(normalized: list[dict], loop_result: dict) -> None:
    accepted = loop_result.get("challenge_questions_filtered", [])
    if len(normalized) != len(accepted):
        raise ValueError("Normalized challenge count does not match accepted loop questions.")
    for question, source in zip(normalized, accepted):
        question["source_loop_question_id"] = source.get("question_id")
        question["source_challenge_plan_id"] = source.get("source_challenge_plan_id")
        question["accepted_by_challenge_loop"] = True
        question["solver_trial_summary"] = source.get("solver_trial_summary", {})
        question["needs_human_review"] = bool(source.get("needs_human_review"))


def _require_target(loop_result: dict, target: int, label: str) -> None:
    accepted = len(loop_result.get("challenge_questions_filtered", []))
    if accepted < target:
        summary = loop_result.get("summary", {})
        raise RuntimeError(
            f"No-graph {label} challenge loop accepted {accepted}/{target}; "
            f"stop_reason={summary.get('stop_reason')}."
        )


def _write_pipeline_artifacts(cache_dir: Path, paper_id: str, text_result: dict, multimodal_result: dict) -> None:
    _write_json(
        cache_dir / "challenge_questions_raw.json",
        {
            "paper_id": paper_id,
            "text": text_result.get("challenge_questions_raw", []),
            "multimodal": multimodal_result.get("challenge_questions_raw", []),
        },
    )
    _write_json(
        cache_dir / "challenge_questions_filtered.json",
        {
            "paper_id": paper_id,
            "text": text_result.get("challenge_questions_filtered", []),
            "multimodal": multimodal_result.get("challenge_questions_filtered", []),
        },
    )
    _write_json(
        cache_dir / "challenge_solver_trials.json",
        {
            "paper_id": paper_id,
            "text": text_result.get("solver_trials", []),
            "multimodal": multimodal_result.get("solver_trials", []),
        },
    )
    _write_json(
        cache_dir / "challenge_questions_need_human_review.json",
        {
            "paper_id": paper_id,
            "text": text_result.get("challenge_questions_need_human_review", []),
            "multimodal": multimodal_result.get("challenge_questions_need_human_review", []),
        },
    )
    _write_json(
        cache_dir / "challenge_questions_rejected.json",
        {
            "paper_id": paper_id,
            "text": text_result.get("challenge_questions_rejected", []),
            "multimodal": multimodal_result.get("challenge_questions_rejected", []),
        },
    )


def _load_or_generate_candidates(path: Path, *, signature: dict, restart: bool, generator) -> dict:
    if path.exists() and not restart:
        cached = json.loads(path.read_text(encoding="utf-8"))
        if cached.get("signature") == signature and isinstance(cached.get("candidates"), dict):
            return cached["candidates"]
    candidates = generator()
    _write_json(path, {"signature": signature, "candidates": candidates})
    return candidates


def _source_signature(
    paper_id: str,
    paper_text: str,
    assets: list[dict],
    macro_count: int,
    text_plan_count: int,
    multimodal_plan_count: int,
    generation_model: str,
    temperature: float,
    generation_attempts: int,
) -> dict:
    return {
        "pipeline_version": "textonly-challenge-v3",
        "paper_id": paper_id,
        "generation_model": generation_model,
        "temperature": temperature,
        "generation_attempts": generation_attempts,
        "paper_sha256": hashlib.sha256(paper_text.encode("utf-8")).hexdigest(),
        "asset_ids": [asset.get("asset_id") for asset in assets],
        "macro_count": macro_count,
        "text_plan_count": text_plan_count,
        "multimodal_plan_count": multimodal_plan_count,
    }


def _macro_prompt(paper_id: str, paper_text: str, count: int) -> str:
    return f"""
Create {count} main paper-reading questions for paper_id={paper_id!r}.
Return {{"macro_questions": [...]}}.
Each item must contain question_text and expected_points. expected_points must contain 3-5 objects with point_id, claim, evidence.
Questions should cover the paper's main problem, method, evidence, findings, and limitations without using graph terminology.

```paper
{paper_text}
```
""".strip()


def _plan_prompt(
    paper_id: str,
    paper_text: str,
    assets: list[dict],
    count: int,
    *,
    multimodal: bool,
) -> str:
    asset_payload = [
        {
            "asset_id": asset.get("asset_id"),
            "asset_type": asset.get("asset_type"),
            "caption": asset.get("caption"),
            "summary": asset.get("summary"),
            "supported_claims": asset.get("supported_claims", []),
            "possible_misreadings": asset.get("possible_misreadings", []),
        }
        for asset in assets
    ]
    allowed_asset_ids = [item["asset_id"] for item in asset_payload if item.get("asset_id")]
    item_schema = {
        "challenge_type": "false_premise | overclaim | wrong_relation | unsupported_generalization",
        "true_part": "paper-supported correction",
        "trap_part": "unsupported or misleading claim",
        "expected_behavior": "reject | qualify | correct",
        "evidence": ["specific paper evidence"],
    }
    if multimodal:
        item_schema["asset_ids"] = ["one or two ids from allowed_asset_ids"]
        item_schema["multimodal_dependency"] = "why the cited figure/table is needed to judge this trap"
        modality_instruction = f"""
Every one of the {count} plans must contain both:
- asset_ids: a non-empty JSON list containing one or two ids selected only from {json.dumps(allowed_asset_ids, ensure_ascii=False)}
- multimodal_dependency: a non-empty explanation tied to those exact assets
A plan without either field is invalid. Never use a caption, figure number, null, or an invented id as asset_ids.
""".strip()
    else:
        modality_instruction = "These plans are text-only and must not include asset_ids or multimodal_dependency."
    return f"""
Create exactly {count} {'multimodal' if multimodal else 'text'} challenge plans for paper_id={paper_id!r}.
Return one JSON object with exactly this top-level form: {{"challenge_plans": [ ... exactly {count} items ... ]}}.
Do not write final natural-language questions.

Every plan must match this item template:
{json.dumps(item_schema, ensure_ascii=False, indent=2)}

Requirements:
- challenge_type must be false_premise, overclaim, wrong_relation, or unsupported_generalization
- true_part states the paper-supported correction
- trap_part states the unsupported or misleading claim
- expected_behavior says reject, qualify, or correct
- evidence is a non-empty JSON list of specific paper evidence
- distribute plans across the four challenge types and avoid duplicate traps
{modality_instruction}

```paper
{paper_text}
```

Allowed assets and their summaries:
{json.dumps(asset_payload, ensure_ascii=False, indent=2)}
""".strip()

def _required_text(raw: dict, key: str, index: int) -> str:
    value = str(raw.get(key) or "").strip()
    if not value:
        raise ValueError(f"Challenge plan #{index} requires non-empty {key}.")
    return value


def _asset_ids(raw: dict, asset_index: dict[str, dict], index: int) -> list[str]:
    values = raw.get("asset_ids")
    if not isinstance(values, list) or not values:
        raise ValueError(f"Multimodal challenge plan #{index} requires asset_ids.")
    result = []
    for value in values[:2]:
        asset_id = str(value).strip()
        if asset_id not in asset_index:
            raise ValueError(f"Multimodal challenge plan #{index} references unknown asset_id={asset_id!r}.")
        if asset_id not in result:
            result.append(asset_id)
    return result


def _empty_loop_result(paper_id: str) -> dict:
    return {
        "paper_id": paper_id,
        "challenge_questions_filtered": [],
        "solver_trials": [],
        "challenge_questions_rejected": [],
        "summary": {"target_count": 0, "filtered_count": 0, "stop_reason": "disabled"},
    }


def _write_json(path: Path, payload: dict) -> None:
    write_json_atomic(path, payload)
