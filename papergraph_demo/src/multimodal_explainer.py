from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from src.model_client import ModelConfig, OpenAICompatClient
from src.progress import log, span
from src.prompt_loader import load_prompt, render_prompt


DASHSCOPE_COMPATIBLE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
VISION_MODEL = "qwen3.5-flash"
ALLOWED_SUPPORT_LEVELS = {
    "direct_table_value",
    "computed_comparison",
    "caption_supported",
    "contextual_interpretation",
    "visually_indicated",
    "visible_label",
}


def build_vision_client(
    embed_api_key: str,
    vision_api_key: str = "",
    vision_base_url: str = "",
    vision_model: str = "",
) -> OpenAICompatClient:
    api_key = vision_api_key.strip() or os.getenv("VISION_API_KEY", "").strip() or embed_api_key
    if not api_key:
        raise RuntimeError("Figure explanation requires EMBED_API_KEY or VISION_API_KEY.")
    return OpenAICompatClient(
        ModelConfig(
            api_key=api_key,
            base_url=vision_base_url.strip() or os.getenv("VISION_BASE_URL", DASHSCOPE_COMPATIBLE_BASE_URL),
            llm_model=vision_model.strip() or os.getenv("VISION_MODEL", VISION_MODEL),
            timeout_s=_env_positive_int("VISION_TIMEOUT_S", 300),
            max_retries=_env_nonnegative_int("VISION_MAX_RETRIES", 2),
        )
    )


def explain_multimodal_assets(
    paper_id: str,
    assets_payload: dict,
    text_client: OpenAICompatClient,
    vision_client: OpenAICompatClient,
) -> dict:
    assets = assets_payload.get("assets", [])
    if not isinstance(assets, list):
        raise ValueError("multimodal_assets payload must contain an assets list.")
    if not text_client or not text_client.is_ready():
        raise RuntimeError("Table asset explanation requires configured text LLM client.")
    figure_assets = [asset for asset in assets if asset.get("asset_type") in {"figure", "mixed"} and asset.get("image_paths")]
    if figure_assets and (not vision_client or not vision_client.is_ready()):
        raise RuntimeError("Figure asset explanation requires configured qwen3.5-flash vision client.")

    max_workers = min(_env_positive_int("MULTIMODAL_EXPLAIN_WORKERS", 3), max(1, len(assets)))
    explanations_by_asset: dict[str, dict] = {}
    errors: list[str] = []

    def run_one(asset: dict) -> tuple[str, dict]:
        asset_id = str(asset.get("asset_id") or "")
        if not asset_id:
            raise ValueError("Every multimodal asset must contain asset_id.")
        if asset.get("asset_type") == "table":
            explanation = _explain_table(asset, text_client)
        elif asset.get("asset_type") in {"figure", "mixed"}:
            explanation = _explain_figure(asset, vision_client)
        else:
            raise ValueError(f"Unsupported asset_type for explanation: {asset.get('asset_type')}")
        return asset_id, explanation

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(run_one, asset): asset for asset in assets}
        for fut in as_completed(futures):
            asset = futures[fut]
            asset_id = str(asset.get("asset_id") or "")
            try:
                out_asset_id, explanation = fut.result()
                explanations_by_asset[out_asset_id] = explanation
                log(
                    "multimodal asset explained",
                    asset_id=out_asset_id,
                    asset_type=asset.get("asset_type"),
                    claims=len(explanation.get("supported_claims", [])),
                )
            except Exception as exc:
                errors.append(f"{asset_id}: {type(exc).__name__}: {exc}")
                log("multimodal asset explanation error", asset_id=asset_id, error=f"{type(exc).__name__}: {exc}")

    if errors:
        raise RuntimeError("Multimodal asset explanation failed: " + "; ".join(errors[:5]))

    explanations = [explanations_by_asset[str(asset.get("asset_id"))] for asset in assets]
    return {
        "paper_id": paper_id,
        "schema_version": "v1",
        "table_explainer_model": text_client.cfg.llm_model,
        "figure_explainer_model": vision_client.cfg.llm_model,
        "figure_explainer_base_url": vision_client.cfg.base_url,
        "asset_explanations": explanations,
        "summary": {
            "asset_explanation_count": len(explanations),
            "by_asset_type": _count_by_field(explanations, "asset_type"),
            "needs_review_count": sum(1 for item in explanations if item.get("needs_review")),
            "supported_claim_count": sum(len(item.get("supported_claims", [])) for item in explanations),
            "possible_misreading_count": sum(len(item.get("possible_misreadings", [])) for item in explanations),
        },
    }


def _explain_table(asset: dict, client: OpenAICompatClient) -> dict:
    tpl = load_prompt("explain_table_asset.txt")
    prompt = render_prompt(
        tpl,
        asset_json=json.dumps(_asset_prompt_payload(asset), ensure_ascii=False, indent=2),
    )
    with span("explain table asset", asset_id=asset.get("asset_id")):
        result = client.chat_json(
            system_prompt="You explain paper table assets as strict, evidence-backed JSON. Return JSON only.",
            user_prompt=prompt,
            temperature=0.1,
        )
    return _normalize_explanation(asset, result, model=client.cfg.llm_model, base_url=client.cfg.base_url)


def _explain_figure(asset: dict, client: OpenAICompatClient) -> dict:
    image_paths = [str(path) for path in asset.get("image_paths", []) if str(path).strip()]
    if not image_paths:
        raise ValueError(f"Figure asset {asset.get('asset_id')} has no image_paths.")
    for image_path in image_paths:
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Figure image path does not exist: {image_path}")
    tpl = load_prompt("explain_figure_asset.txt")
    prompt = render_prompt(
        tpl,
        asset_json=json.dumps(_asset_prompt_payload(asset), ensure_ascii=False, indent=2),
    )
    with span("explain figure asset", asset_id=asset.get("asset_id"), images=len(image_paths)):
        result = client.chat_json_with_images(
            system_prompt="You explain paper figure assets using the attached image and return JSON only.",
            user_prompt=prompt,
            image_paths=image_paths,
            temperature=0.1,
        )
    return _normalize_explanation(asset, result, model=client.cfg.llm_model, base_url=client.cfg.base_url)


def _asset_prompt_payload(asset: dict) -> dict:
    payload = {
        "asset_id": asset.get("asset_id"),
        "asset_type": asset.get("asset_type"),
        "modality_class": asset.get("modality_class"),
        "subtype": asset.get("subtype"),
        "section_id": asset.get("section_id"),
        "section_title": asset.get("section_title"),
        "macro_id": asset.get("macro_id"),
        "caption": asset.get("caption"),
        "nearby_context": asset.get("nearby_context"),
        "source_block_ids": asset.get("source_block_ids", []),
        "source_basis": asset.get("source_basis", []),
    }
    if asset.get("asset_type") == "table":
        payload["normalized_markdown"] = asset.get("normalized_markdown", "")
        payload["table_shape"] = asset.get("table_shape", {})
    else:
        payload["image_paths"] = asset.get("image_paths", [])
        payload["panel_structure"] = asset.get("panel_structure", [])
    return payload


def _normalize_explanation(asset: dict, result: dict, model: str, base_url: str) -> dict:
    if not isinstance(result, dict):
        raise ValueError(f"Explanation for {asset.get('asset_id')} must be a JSON object.")
    supported_claims = _normalize_supported_claims(asset, result.get("supported_claims", []))
    possible_misreadings = _normalize_possible_misreadings(result.get("possible_misreadings", []))
    limitations = _string_list(result.get("limitations", []), "limitations")
    confidence = _confidence(result.get("confidence"))
    needs_review = bool(result.get("needs_review", False))
    summary = str(result.get("summary", "")).strip()
    if not summary:
        raise ValueError(f"Explanation for {asset.get('asset_id')} must contain non-empty summary.")
    return {
        "asset_id": asset.get("asset_id"),
        "asset_type": asset.get("asset_type"),
        "modality_class": asset.get("modality_class"),
        "subtype": asset.get("subtype"),
        "section_id": asset.get("section_id"),
        "macro_id": asset.get("macro_id"),
        "caption": asset.get("caption", ""),
        "summary": summary,
        "key_elements": _object_list(result.get("key_elements", []), "key_elements"),
        "relations": _object_list(result.get("relations", []), "relations"),
        "supported_claims": supported_claims,
        "possible_misreadings": possible_misreadings,
        "limitations": limitations,
        "needs_review": needs_review,
        "confidence": confidence,
        "source_basis": asset.get("source_basis", []),
        "explanation_model": model,
        "explanation_base_url": base_url,
    }


def _normalize_supported_claims(asset: dict, values: object) -> list[dict]:
    items = _object_list(values, "supported_claims")
    out = []
    for idx, item in enumerate(items, start=1):
        claim = str(item.get("claim", "")).strip()
        support_level = str(item.get("support_level", "")).strip()
        evidence_basis = str(item.get("evidence_basis", "")).strip()
        if not claim or not support_level or not evidence_basis:
            raise ValueError(f"supported_claims[{idx}] for {asset.get('asset_id')} must include claim/support_level/evidence_basis.")
        if support_level not in ALLOWED_SUPPORT_LEVELS:
            raise ValueError(
                f"supported_claims[{idx}] for {asset.get('asset_id')} has invalid support_level: {support_level}"
            )
        scope = item.get("scope") if isinstance(item.get("scope"), dict) else {}
        scope.setdefault("generality", "limited_to_asset")
        scope.setdefault("asset_id", asset.get("asset_id"))
        out.append(
            {
                "claim": claim,
                "support_level": support_level,
                "evidence_basis": evidence_basis,
                "scope": scope,
            }
        )
    if not out:
        raise ValueError(f"Explanation for {asset.get('asset_id')} must include at least one supported_claim.")
    return out


def _normalize_possible_misreadings(values: object) -> list[dict]:
    items = _object_list(values, "possible_misreadings")
    out = []
    for idx, item in enumerate(items, start=1):
        claim = str(item.get("claim", "")).strip()
        why_wrong = str(item.get("why_wrong", "")).strip()
        if not claim or not why_wrong:
            raise ValueError(f"possible_misreadings[{idx}] must include claim and why_wrong.")
        out.append({"claim": claim, "why_wrong": why_wrong})
    return out


def _object_list(values: object, field: str) -> list[dict[str, Any]]:
    if values is None:
        return []
    if not isinstance(values, list):
        raise ValueError(f"{field} must be a list.")
    for item in values:
        if not isinstance(item, dict):
            raise ValueError(f"{field} must contain objects only.")
    return values


def _string_list(values: object, field: str) -> list[str]:
    if values is None:
        return []
    if not isinstance(values, list):
        raise ValueError(f"{field} must be a list.")
    out = [str(item).strip() for item in values if str(item).strip()]
    return out


def _confidence(value: object) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        raise ValueError("confidence must be a number between 0 and 1.")
    if parsed < 0 or parsed > 1:
        raise ValueError("confidence must be between 0 and 1.")
    return parsed


def _count_by_field(items: list[dict], field: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for item in items:
        key = str(item.get(field) or "unknown")
        out[key] = out.get(key, 0) + 1
    return out


def _env_positive_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def _env_nonnegative_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value >= 0 else default
