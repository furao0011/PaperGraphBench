from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.model_client import OpenAICompatClient
from src.progress import log, span
from src.prompt_loader import load_prompt, render_prompt


ALLOWED_ASSET_TYPES = {"table", "figure", "mixed", "noise"}
ALLOWED_MODALITY_CLASSES = {"evidential", "explanatory", "unknown"}


def analyze_multimodal_html_groups(
    paper_id: str,
    asset_groups: dict,
    client: OpenAICompatClient,
) -> dict:
    if not client or not client.is_ready():
        raise RuntimeError("Multimodal HTML group analysis requires the configured text LLM client.")
    groups = asset_groups.get("asset_groups", [])
    if not isinstance(groups, list):
        raise ValueError("asset_groups payload must contain an asset_groups list.")

    tpl = load_prompt("analyze_multimodal_html_group.txt")
    max_workers = min(_env_positive_int("MULTIMODAL_GROUP_ANALYZE_WORKERS", 3), max(1, len(groups)))
    analyzed_by_id: dict[str, dict] = {}
    errors: list[str] = []

    def run_one(group: dict) -> tuple[str, dict]:
        group_id = str(group.get("asset_group_id") or "")
        prompt = render_prompt(
            tpl,
            group_json=json.dumps(_group_prompt_payload(group), ensure_ascii=False, indent=2),
        )
        with span("analyze multimodal html group", group_id=group_id):
            result = client.chat_json(
                system_prompt=(
                    "You extract structured multimodal assets from consecutive OCR HTML blocks. "
                    "Return strict JSON only."
                ),
                user_prompt=prompt,
                temperature=0.1,
            )
        return group_id, _normalize_group_analysis(group, result)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(run_one, group): group for group in groups}
        for fut in as_completed(futures):
            group = futures[fut]
            group_id = str(group.get("asset_group_id") or "")
            try:
                out_group_id, analysis = fut.result()
                analyzed_by_id[out_group_id] = analysis
                log(
                    "multimodal html group analyzed",
                    group_id=out_group_id,
                    assets=len(analysis.get("group_assets", [])),
                )
            except Exception as exc:
                errors.append(f"{group_id}: {type(exc).__name__}: {exc}")
                log("multimodal html group analysis error", group_id=group_id, error=f"{type(exc).__name__}: {exc}")

    if errors:
        raise RuntimeError("Multimodal HTML group analysis failed: " + "; ".join(errors[:5]))

    out_groups = []
    for group in groups:
        item = dict(group)
        group_id = str(item.get("asset_group_id") or "")
        item["llm_extraction"] = analyzed_by_id[group_id]
        out_groups.append(item)

    return {
        "paper_id": paper_id,
        "schema_version": "v1_llm_html_island_analysis",
        "asset_groups": out_groups,
        "summary": {
            **asset_groups.get("summary", {}),
            "llm_analyzed_group_count": len(out_groups),
            "llm_extracted_asset_count": sum(
                len(group.get("llm_extraction", {}).get("group_assets", []))
                for group in out_groups
            ),
        },
    }


def _group_prompt_payload(group: dict) -> dict:
    return {
        "asset_group_id": group.get("asset_group_id"),
        "section_id": group.get("section_id"),
        "section_title": group.get("section_title"),
        "asset_group_type": group.get("asset_group_type"),
        "html_blocks": [
            {
                "block_id": block.get("block_id"),
                "block_type": block.get("block_type"),
                "text": _limit(str(block.get("text", "")), 3000),
                "html": _limit(str(block.get("html", "")), 12000),
                "image_path": block.get("image_path"),
                "resolved_image_path": block.get("resolved_image_path"),
            }
            for block in group.get("html_blocks", [])
        ],
        "media_blocks": [
            {
                "block_id": block.get("block_id"),
                "media_type": block.get("media_type"),
                "image_path": block.get("image_path"),
                "resolved_image_path": block.get("resolved_image_path"),
                "text": _limit(str(block.get("text", "")), 1000),
            }
            for block in group.get("media_blocks", [])
        ],
    }


def _normalize_group_analysis(group: dict, result: dict) -> dict:
    raw_assets = result.get("group_assets")
    if not isinstance(raw_assets, list):
        raise ValueError("Group analysis response must contain group_assets list.")
    block_ids = {block.get("block_id") for block in group.get("html_blocks", [])}
    media_ids = {block.get("block_id") for block in group.get("media_blocks", [])}
    normalized_assets = []
    for idx, raw in enumerate(raw_assets, start=1):
        if not isinstance(raw, dict):
            raise ValueError("Every group asset must be an object.")
        asset_type = str(raw.get("asset_type", "")).strip().lower()
        if asset_type not in ALLOWED_ASSET_TYPES:
            raise ValueError(f"Invalid asset_type for group {group.get('asset_group_id')}: {asset_type}")
        modality_class = str(raw.get("modality_class", "unknown")).strip().lower() or "unknown"
        if modality_class not in ALLOWED_MODALITY_CLASSES:
            raise ValueError(f"Invalid modality_class for group {group.get('asset_group_id')}: {modality_class}")
        source_block_ids = _valid_ids(raw.get("source_block_ids", []), block_ids, "source_block_ids")
        media_block_ids = _valid_ids(raw.get("media_block_ids", []), media_ids, "media_block_ids")
        table_block_ids = _valid_ids(raw.get("table_block_ids", []), block_ids, "table_block_ids")
        image_block_ids = _valid_ids(raw.get("image_block_ids", []), block_ids, "image_block_ids")
        if asset_type != "noise" and not media_block_ids:
            raise ValueError(f"Non-noise asset in group {group.get('asset_group_id')} must include media_block_ids.")
        normalized_assets.append(
            {
                "asset_local_id": str(raw.get("asset_local_id") or f"A{idx}").strip(),
                "asset_type": asset_type,
                "modality_class": modality_class,
                "subtype_hint": str(raw.get("subtype_hint", "unknown")).strip() or "unknown",
                "caption": str(raw.get("caption", "")).strip(),
                "caption_kind": str(raw.get("caption_kind", "")).strip().lower() or None,
                "caption_number": str(raw.get("caption_number", "")).strip() or None,
                "source_block_ids": source_block_ids,
                "media_block_ids": media_block_ids,
                "table_block_ids": table_block_ids,
                "image_block_ids": image_block_ids,
                "description_from_html": str(raw.get("description_from_html", "")).strip(),
                "notes": str(raw.get("notes", "")).strip(),
            }
        )
    return {
        "group_assets": normalized_assets,
        "group_notes": str(result.get("group_notes", "")).strip(),
        "analyzer": "text_llm_v1",
    }


def _valid_ids(values: object, allowed: set[object], field: str) -> list[str]:
    if values is None:
        return []
    if not isinstance(values, list):
        raise ValueError(f"{field} must be a list.")
    out = []
    for value in values:
        item = str(value).strip()
        if item and item not in allowed:
            raise ValueError(f"{field} references unknown block_id: {item}")
        if item:
            out.append(item)
    return list(dict.fromkeys(out))


def _limit(text: str, max_chars: int) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...[truncated]"


def _env_positive_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default
