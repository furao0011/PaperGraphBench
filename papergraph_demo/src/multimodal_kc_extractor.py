from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.kc_type_registry import MULTIMODAL_KC_TYPES, valid_kc_type
from src.model_client import OpenAICompatClient
from src.progress import log, span
from src.prompt_loader import load_prompt, render_prompt


VALID_IMPORTANCE = {"critical", "normal"}

VALID_CLAIM_STRENGTH = {
    "direct_table_value",
    "computed_comparison",
    "caption_supported",
    "contextual_interpretation",
    "visually_indicated",
    "visible_label",
}


def extract_multimodal_kc_candidates(
    paper_id: str,
    asset_explanations_payload: dict,
    client: OpenAICompatClient,
    macro_spine: dict | None = None,
) -> dict:
    if not client or not client.is_ready():
        raise RuntimeError("Multimodal KC extraction requires a configured text LLM client.")
    explanations = asset_explanations_payload.get("asset_explanations", [])
    if not isinstance(explanations, list) or not explanations:
        raise ValueError("asset_explanations payload must contain a non-empty asset_explanations list.")

    tpl = load_prompt("extract_multimodal_kc_from_asset.txt")
    macro_bind_tpl = load_prompt("bind_multimodal_asset_macro.txt")
    macro_options = _macro_options(macro_spine or {})
    max_workers = min(_env_positive_int("MULTIMODAL_KC_WORKERS", 3), len(explanations))
    candidates_by_asset: dict[str, list[dict]] = {}
    empty_assets: dict[str, str] = {}
    errors: list[str] = []

    def run_one(explanation: dict) -> tuple[str, list[dict], str]:
        asset_id = str(explanation.get("asset_id", "")).strip()
        if not asset_id:
            raise ValueError("Every asset explanation must contain asset_id.")
        prompt = render_prompt(
            tpl,
            asset_explanation_json=json.dumps(_explanation_prompt_payload(explanation), ensure_ascii=False, indent=2),
        )
        with span("multimodal KC extraction", asset_id=asset_id):
            result = client.chat_json(
                system_prompt="You extract multimodal paper-evaluation KCs. Return JSON only.",
                user_prompt=prompt,
                temperature=0.1,
            )
        explanation_for_kc = explanation
        if result.get("kcs") and not _macro_id_or_empty(explanation.get("macro_id")):
            binding = _bind_asset_macro(
                explanation=explanation,
                kc_result=result,
                macro_options=macro_options,
                template=macro_bind_tpl,
                client=client,
            )
            explanation_for_kc = dict(explanation)
            explanation_for_kc["macro_id"] = binding["macro_id"]
            explanation_for_kc["macro_binding"] = binding
        candidates, empty_reason = _normalize_asset_result(explanation_for_kc, result)
        return asset_id, candidates, empty_reason

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(run_one, explanation): explanation for explanation in explanations}
        for fut in as_completed(futures):
            explanation = futures[fut]
            asset_id = str(explanation.get("asset_id", "")).strip()
            try:
                out_asset_id, candidates, empty_reason = fut.result()
                candidates_by_asset[out_asset_id] = candidates
                if not candidates:
                    empty_assets[out_asset_id] = empty_reason
                log("multimodal KC extraction completed", asset_id=out_asset_id, candidates=len(candidates))
            except Exception as exc:
                errors.append(f"{asset_id}: {type(exc).__name__}: {exc}")
                log("multimodal KC extraction error", asset_id=asset_id, error=f"{type(exc).__name__}: {exc}")

    if errors:
        raise RuntimeError("Multimodal KC extraction failed: " + "; ".join(errors[:5]))

    candidates: list[dict] = []
    for explanation in explanations:
        asset_id = str(explanation.get("asset_id", "")).strip()
        for item in candidates_by_asset.get(asset_id, []):
            candidates.append(
                {
                    "candidate_id": f"MMC_{len(candidates) + 1:04d}",
                    **item,
                }
            )
    if not candidates:
        raise RuntimeError(
            "Multimodal KC extraction produced no candidates. "
            f"Empty assets: {json.dumps(empty_assets, ensure_ascii=False)[:1200]}"
        )

    return {
        "paper_id": paper_id,
        "schema_version": "v1",
        "extraction_source": "multimodal_asset_explanations",
        "kc_candidates": candidates,
        "empty_assets": [
            {"asset_id": asset_id, "empty_reason": reason}
            for asset_id, reason in sorted(empty_assets.items())
        ],
        "summary": {
            "asset_count": len(explanations),
            "candidate_count": len(candidates),
            "empty_asset_count": len(empty_assets),
            "by_type": _count_by_field(candidates, "type"),
            "by_asset_type": _count_by_asset_type(candidates),
        },
    }


def _explanation_prompt_payload(explanation: dict) -> dict:
    return {
        "asset_id": explanation.get("asset_id"),
        "asset_type": explanation.get("asset_type"),
        "modality_class": explanation.get("modality_class"),
        "subtype": explanation.get("subtype"),
        "section_id": explanation.get("section_id"),
        "macro_id": explanation.get("macro_id"),
        "caption": explanation.get("caption"),
        "summary": explanation.get("summary"),
        "key_elements": explanation.get("key_elements", []),
        "relations": explanation.get("relations", []),
        "supported_claims": explanation.get("supported_claims", []),
        "possible_misreadings": explanation.get("possible_misreadings", []),
        "limitations": explanation.get("limitations", []),
        "needs_review": explanation.get("needs_review"),
        "confidence": explanation.get("confidence"),
    }


def _normalize_asset_result(explanation: dict, result: dict) -> tuple[list[dict], str]:
    raw_kcs = result.get("kcs", [])
    if not isinstance(raw_kcs, list):
        raise ValueError("Multimodal KC response must contain kcs list.")
    empty_reason = str(result.get("empty_reason", "")).strip()
    if not raw_kcs:
        if not empty_reason:
            raise ValueError(f"Asset {explanation.get('asset_id')} returned no KCs but did not include empty_reason.")
        return [], empty_reason

    asset_id = str(explanation.get("asset_id", "")).strip()
    asset_type = str(explanation.get("asset_type", "")).strip()
    section_id = str(explanation.get("section_id", "")).strip()
    macro_id = _macro_id_or_empty(explanation.get("macro_id"))
    if not macro_id:
        raise ValueError(f"Asset {asset_id} has no macro_id; cannot create multimodal KC.")
    possible_misreadings = explanation.get("possible_misreadings", [])
    per_asset_limit = _env_positive_int("MULTIMODAL_KC_PER_ASSET_LIMIT", 4)
    limitation_limit = _env_nonnegative_int("MULTIMODAL_LIMITATION_KC_PER_ASSET_LIMIT", 1)
    if len(raw_kcs) > per_asset_limit:
        raise ValueError(
            f"Asset {asset_id} returned {len(raw_kcs)} KCs, exceeding MULTIMODAL_KC_PER_ASSET_LIMIT={per_asset_limit}."
        )
    limitation_count = sum(
        1
        for item in raw_kcs
        if isinstance(item, dict) and str(item.get("type", "")).strip() == "multimodal_limitation"
    )
    if limitation_count > limitation_limit:
        raise ValueError(
            f"Asset {asset_id} returned {limitation_count} limitation KCs, exceeding "
            f"MULTIMODAL_LIMITATION_KC_PER_ASSET_LIMIT={limitation_limit}."
        )
    out = []
    for idx, item in enumerate(raw_kcs, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"KC #{idx} for asset {asset_id} must be an object.")
        claim = str(item.get("claim", "")).strip()
        evidence_basis = str(item.get("evidence_basis", "")).strip()
        raw_kc_type = str(item.get("type", "")).strip()
        kc_type = valid_kc_type(raw_kc_type, MULTIMODAL_KC_TYPES)
        importance = str(item.get("importance", "")).strip()
        claim_strength = str(item.get("claim_strength", "")).strip()
        if not claim or not evidence_basis:
            raise ValueError(f"KC #{idx} for asset {asset_id} must include claim and evidence_basis.")
        if kc_type is None:
            raise ValueError(f"KC #{idx} for asset {asset_id} has invalid type={raw_kc_type!r}.")
        if importance not in VALID_IMPORTANCE:
            raise ValueError(f"KC #{idx} for asset {asset_id} has invalid importance={importance!r}.")
        if claim_strength not in VALID_CLAIM_STRENGTH:
            raise ValueError(f"KC #{idx} for asset {asset_id} has invalid claim_strength={claim_strength!r}.")
        scope = item.get("scope") if isinstance(item.get("scope"), dict) else {}
        scope["generality"] = str(scope.get("generality", "limited_to_asset")).strip() or "limited_to_asset"
        scope["asset_id"] = asset_id
        evidence_text = _evidence_text(explanation, claim, evidence_basis)
        out.append(
            {
                "unit_id": f"MMU_{asset_id}",
                "source_window_id": asset_id,
                "section": explanation.get("section_id", ""),
                "section_id": section_id,
                "source_chunk_id": asset_id,
                "unit_title": f"{asset_id} {asset_type}",
                "unit_summary": explanation.get("summary", ""),
                "claim": claim,
                "evidence": evidence_text,
                "macro_id": macro_id,
                "type": kc_type,
                "importance": importance,
                "claim_strength": claim_strength,
                "scope": scope,
                "related_terms": _string_list(item.get("related_terms", [])),
                "modality": {
                    "is_multimodal": True,
                    "asset_id": asset_id,
                    "asset_type": asset_type,
                    "modality_class": explanation.get("modality_class"),
                    "subtype": explanation.get("subtype"),
                },
                "asset_id": asset_id,
                "asset_type": asset_type,
                "asset_caption": explanation.get("caption", ""),
                "asset_summary": explanation.get("summary", ""),
                "asset_evidence_basis": evidence_basis,
                "asset_source_basis": explanation.get("source_basis", []),
                "asset_possible_misreadings": possible_misreadings,
                "asset_needs_review": bool(explanation.get("needs_review", False)),
                "asset_confidence": explanation.get("confidence"),
                "asset_macro_binding": explanation.get("macro_binding"),
                "evidence_items": [
                    {
                        "section": explanation.get("section_id", ""),
                        "span_id": asset_id,
                        "text": evidence_text,
                        "modality": asset_type,
                        "asset_id": asset_id,
                        "evidence_basis": evidence_basis,
                    }
                ],
                "forbidden_claims": _forbidden_claims(asset_id, possible_misreadings),
            }
        )
    return out, empty_reason


def _evidence_text(explanation: dict, claim: str, evidence_basis: str) -> str:
    parts = [
        f"Asset {explanation.get('asset_id')} ({explanation.get('asset_type')}): {explanation.get('caption', '')}".strip(),
        f"Asset summary: {explanation.get('summary', '')}".strip(),
        f"Evidence basis: {evidence_basis}".strip(),
        f"Supported claim: {claim}".strip(),
    ]
    return "\n".join(part for part in parts if part)


def _macro_id_or_empty(value: object) -> str:
    if value is None:
        return ""
    macro_id = str(value).strip()
    if macro_id.lower() in {"none", "null", "nan"}:
        return ""
    return macro_id


def _bind_asset_macro(
    explanation: dict,
    kc_result: dict,
    macro_options: list[dict],
    template: str,
    client: OpenAICompatClient,
) -> dict:
    if not macro_options:
        raise ValueError(f"Asset {explanation.get('asset_id')} has no macro_id and Macro Spine is empty.")
    allowed = {str(item.get("macro_id")) for item in macro_options if str(item.get("macro_id") or "").strip()}
    payload = {
        "asset": {
            "asset_id": explanation.get("asset_id"),
            "asset_type": explanation.get("asset_type"),
            "section_id": explanation.get("section_id"),
            "caption": explanation.get("caption"),
            "summary": explanation.get("summary"),
            "key_elements": explanation.get("key_elements", []),
            "relations": explanation.get("relations", []),
            "supported_claims": explanation.get("supported_claims", []),
        },
        "extracted_kcs": kc_result.get("kcs", []),
        "macro_options": macro_options,
    }
    last_error = ""
    for attempt in range(1, 3):
        prompt = render_prompt(
            template,
            macro_binding_json=json.dumps(payload, ensure_ascii=False, indent=2),
            previous_error=last_error,
        )
        with span("bind multimodal asset macro", asset_id=explanation.get("asset_id"), attempt=attempt):
            result = client.chat_json(
                system_prompt="You bind one multimodal asset to exactly one existing paper Macro. Return JSON only.",
                user_prompt=prompt,
                temperature=0.0,
            )
        macro_id = _macro_id_or_empty(result.get("macro_id"))
        if macro_id in allowed:
            return {
                "macro_id": macro_id,
                "reason": str(result.get("reason") or "").strip(),
                "confidence": str(result.get("confidence") or "").strip(),
                "binding_source": "llm_forced_macro_binding",
            }
        last_error = (
            f"Previous macro_id={result.get('macro_id')!r} is invalid. "
            f"Choose exactly one macro_id from: {sorted(allowed)}."
        )
    raise ValueError(
        f"Asset {explanation.get('asset_id')} macro binding failed after retry: {last_error}"
    )


def _macro_options(macro_spine: dict) -> list[dict]:
    options = []
    for macro in macro_spine.get("macro_nodes", []):
        macro_id = _macro_id_or_empty(macro.get("macro_id"))
        if not macro_id:
            continue
        options.append(
            {
                "macro_id": macro_id,
                "title": macro.get("title"),
                "role": macro.get("role"),
                "summary": macro.get("summary"),
                "source_sections": macro.get("source_sections", []),
                "order": macro.get("order"),
            }
        )
    return options


def _forbidden_claims(asset_id: str, possible_misreadings: object) -> list[dict]:
    if not isinstance(possible_misreadings, list):
        return []
    out = []
    for idx, item in enumerate(possible_misreadings, start=1):
        if not isinstance(item, dict):
            continue
        claim = str(item.get("claim", "")).strip()
        why_wrong = str(item.get("why_wrong", "")).strip()
        if not claim or not why_wrong:
            continue
        out.append(
            {
                "claim_id": f"MM_FC_{asset_id}_{idx}",
                "claim": claim,
                "type": "multimodal_misreading",
                "severity": "medium",
                "why_wrong": why_wrong,
                "related_asset_ids": [asset_id],
            }
        )
    return out


def _string_list(values: object) -> list[str]:
    if not isinstance(values, list):
        return []
    return [str(value).strip() for value in values if str(value).strip()]


def _count_by_field(items: list[dict], field: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for item in items:
        key = str(item.get(field) or "unknown")
        out[key] = out.get(key, 0) + 1
    return out


def _count_by_asset_type(items: list[dict]) -> dict[str, int]:
    out: dict[str, int] = {}
    for item in items:
        key = str(item.get("modality", {}).get("asset_type") or "unknown")
        out[key] = out.get(key, 0) + 1
    return out


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


def _env_nonnegative_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a non-negative integer, got {raw!r}.") from exc
    if value < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {value}.")
    return value
