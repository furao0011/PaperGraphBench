from __future__ import annotations

import re

from src.table_html_normalizer import normalize_table_html


def normalize_multimodal_assets(paper_id: str, asset_groups: dict, macro_spine: dict) -> dict:
    groups = asset_groups.get("asset_groups", [])
    if not isinstance(groups, list):
        raise ValueError("asset_groups payload must contain an asset_groups list.")

    assets: list[dict] = []
    counters = {"table": 0, "figure": 0, "mixed": 0}
    macro_index = _build_macro_index(macro_spine)

    for group in groups:
        extraction = group.get("llm_extraction")
        if not isinstance(extraction, dict):
            raise ValueError(
                f"Asset group {group.get('asset_group_id')} must contain llm_extraction before normalization."
            )
        html_by_id = {block.get("block_id"): block for block in group.get("html_blocks", [])}
        media_by_id = {block.get("block_id"): block for block in group.get("media_blocks", [])}
        macro_resolution = _resolve_macro(group, macro_index)

        for group_asset in extraction.get("group_assets", []):
            asset_type = group_asset.get("asset_type")
            if asset_type == "noise":
                continue
            if asset_type not in {"table", "figure", "mixed"}:
                raise ValueError(f"Unsupported analyzed asset_type: {asset_type}")
            counters[asset_type] += 1
            asset_id = _asset_id(asset_type, counters[asset_type])
            if asset_type == "table":
                assets.append(_table_asset(asset_id, group, group_asset, html_by_id, macro_resolution))
            elif asset_type == "figure":
                assets.append(_figure_asset(asset_id, group, group_asset, media_by_id, macro_resolution))
            else:
                assets.append(_mixed_asset(asset_id, group, group_asset, html_by_id, media_by_id, macro_resolution))

    return {
        "paper_id": paper_id,
        "schema_version": "v2_llm_structured_assets",
        "assets": assets,
        "summary": {
            "asset_count": len(assets),
            "by_asset_type": _count_by_field(assets, "asset_type"),
            "by_modality_class": _count_by_field(assets, "modality_class"),
            "macro_unresolved_count": sum(1 for asset in assets if asset.get("macro_resolution_status") != "resolved"),
        },
    }


def _table_asset(
    asset_id: str,
    group: dict,
    group_asset: dict,
    html_by_id: dict,
    macro_resolution: dict,
) -> dict:
    table_ids = group_asset.get("table_block_ids") or group_asset.get("media_block_ids") or []
    table_blocks = [_required_block(html_by_id, block_id, asset_id) for block_id in table_ids]
    if not table_blocks:
        raise ValueError(f"Table asset {asset_id} has no table blocks.")
    normalized_tables = [normalize_table_html(str(block.get("html") or "")) for block in table_blocks]
    markdown_parts = [item["normalized_markdown"] for item in normalized_tables if item.get("normalized_markdown")]
    first_shape = normalized_tables[0]["table_shape"]
    return {
        "asset_id": asset_id,
        "asset_type": "table",
        "modality_class": _modality_class(group_asset, default="evidential"),
        "subtype": _subtype(group_asset, fallback="unknown_table"),
        "section_id": group.get("section_id"),
        "section_title": group.get("section_title"),
        "macro_id": macro_resolution["macro_id"],
        "macro_candidates": macro_resolution["macro_candidates"],
        "macro_resolution_status": macro_resolution["status"],
        "source_group_id": group.get("asset_group_id"),
        "source_block_ids": group_asset.get("source_block_ids", []),
        "media_block_ids": group_asset.get("media_block_ids", []),
        "caption": group_asset.get("caption", ""),
        "caption_kind": group_asset.get("caption_kind"),
        "caption_number": group_asset.get("caption_number"),
        "html": "\n\n".join(str(block.get("html") or "") for block in table_blocks),
        "normalized_markdown": "\n\n".join(markdown_parts),
        "table_grid": normalized_tables[0]["grid"] if len(normalized_tables) == 1 else [item["grid"] for item in normalized_tables],
        "nearby_context": group_asset.get("description_from_html", ""),
        "table_shape": first_shape if len(normalized_tables) == 1 else {"tables": [item["table_shape"] for item in normalized_tables]},
        "source_basis": ["llm_html_island_extraction", "html_table"],
        "llm_group_asset": group_asset,
        "diagnostics": {
            "has_caption": bool(group_asset.get("caption")),
            "table_block_count": len(table_blocks),
        },
    }


def _figure_asset(
    asset_id: str,
    group: dict,
    group_asset: dict,
    media_by_id: dict,
    macro_resolution: dict,
) -> dict:
    image_ids = group_asset.get("image_block_ids") or group_asset.get("media_block_ids") or []
    image_blocks = [_required_block(media_by_id, block_id, asset_id) for block_id in image_ids]
    image_paths = [block.get("resolved_image_path") for block in image_blocks if block.get("resolved_image_path")]
    if len(image_paths) != len(image_blocks):
        missing = [block.get("block_id") for block in image_blocks if not block.get("resolved_image_path")]
        raise ValueError(f"Figure asset {asset_id} has image blocks without resolved paths: {missing}")
    return {
        "asset_id": asset_id,
        "asset_type": "figure",
        "modality_class": _modality_class(group_asset, default="unknown"),
        "subtype": _subtype(group_asset, fallback="unknown_figure"),
        "section_id": group.get("section_id"),
        "section_title": group.get("section_title"),
        "macro_id": macro_resolution["macro_id"],
        "macro_candidates": macro_resolution["macro_candidates"],
        "macro_resolution_status": macro_resolution["status"],
        "source_group_id": group.get("asset_group_id"),
        "source_block_ids": group_asset.get("source_block_ids", []),
        "media_block_ids": group_asset.get("media_block_ids", []),
        "image_paths": image_paths,
        "attachments": _image_attachments(asset_id, image_paths, group_asset.get("caption", "")),
        "caption": group_asset.get("caption", ""),
        "caption_kind": group_asset.get("caption_kind"),
        "caption_number": group_asset.get("caption_number"),
        "nearby_context": group_asset.get("description_from_html", ""),
        "panel_structure": _panel_structure(group_asset.get("caption", ""), image_blocks),
        "source_basis": ["llm_html_island_extraction", "image_file"],
        "llm_group_asset": group_asset,
        "diagnostics": {
            "has_caption": bool(group_asset.get("caption")),
            "image_count": len(image_blocks),
        },
    }


def _mixed_asset(
    asset_id: str,
    group: dict,
    group_asset: dict,
    html_by_id: dict,
    media_by_id: dict,
    macro_resolution: dict,
) -> dict:
    table_ids = group_asset.get("table_block_ids") or []
    image_ids = group_asset.get("image_block_ids") or []
    table_blocks = [_required_block(html_by_id, block_id, asset_id) for block_id in table_ids]
    image_blocks = [_required_block(media_by_id, block_id, asset_id) for block_id in image_ids]
    normalized_tables = [normalize_table_html(str(block.get("html") or "")) for block in table_blocks]
    return {
        "asset_id": asset_id,
        "asset_type": "mixed",
        "modality_class": _modality_class(group_asset, default="unknown"),
        "subtype": _subtype(group_asset, fallback="unknown_mixed"),
        "section_id": group.get("section_id"),
        "section_title": group.get("section_title"),
        "macro_id": macro_resolution["macro_id"],
        "macro_candidates": macro_resolution["macro_candidates"],
        "macro_resolution_status": macro_resolution["status"],
        "source_group_id": group.get("asset_group_id"),
        "source_block_ids": group_asset.get("source_block_ids", []),
        "media_block_ids": group_asset.get("media_block_ids", []),
        "image_paths": [block.get("resolved_image_path") for block in image_blocks if block.get("resolved_image_path")],
        "attachments": _image_attachments(
            asset_id,
            [block.get("resolved_image_path") for block in image_blocks if block.get("resolved_image_path")],
            group_asset.get("caption", ""),
        ),
        "caption": group_asset.get("caption", ""),
        "caption_kind": group_asset.get("caption_kind"),
        "caption_number": group_asset.get("caption_number"),
        "normalized_markdown": "\n\n".join(item["normalized_markdown"] for item in normalized_tables),
        "nearby_context": group_asset.get("description_from_html", ""),
        "source_basis": ["llm_html_island_extraction", "html_table", "image_file"],
        "llm_group_asset": group_asset,
        "diagnostics": {
            "has_caption": bool(group_asset.get("caption")),
            "table_block_count": len(table_blocks),
            "image_count": len(image_blocks),
        },
    }


def _required_block(blocks: dict, block_id: str, asset_id: str) -> dict:
    block = blocks.get(block_id)
    if not block:
        raise ValueError(f"Asset {asset_id} references unknown block_id: {block_id}")
    return block


def _asset_id(asset_type: str, idx: int) -> str:
    if asset_type == "table":
        return f"TAB_{idx:03d}"
    if asset_type == "figure":
        return f"FIG_{idx:03d}"
    return f"MIX_{idx:03d}"


def _image_attachments(asset_id: str, image_paths: list[str], caption: str) -> list[dict]:
    return [
        {
            "type": "image",
            "asset_id": asset_id,
            "path": image_path,
            "caption": caption,
        }
        for image_path in image_paths
    ]


def _modality_class(group_asset: dict, default: str) -> str:
    value = str(group_asset.get("modality_class") or "").strip().lower()
    return value if value in {"evidential", "explanatory", "unknown"} else default


def _subtype(group_asset: dict, fallback: str) -> str:
    value = str(group_asset.get("subtype_hint") or "").strip()
    return value if value and value != "unknown" else fallback


def _build_macro_index(macro_spine: dict) -> dict[str, list[dict]]:
    index: dict[str, list[dict]] = {}
    for macro in macro_spine.get("macro_nodes", []):
        for source in macro.get("source_sections", []):
            key = _norm(source)
            index.setdefault(key, []).append(
                {
                    "macro_id": macro.get("macro_id"),
                    "title": macro.get("title"),
                    "role": macro.get("role"),
                    "order": macro.get("order", 0),
                }
            )
    for candidates in index.values():
        candidates.sort(key=lambda item: item.get("order", 0))
    return index


def _resolve_macro(group: dict, macro_index: dict[str, list[dict]]) -> dict:
    candidates = []
    for key in (_norm(group.get("section_id")), _norm(group.get("section_title"))):
        candidates.extend(macro_index.get(key, []))
    unique = []
    seen = set()
    for item in candidates:
        macro_id = item.get("macro_id")
        if macro_id and macro_id not in seen:
            seen.add(macro_id)
            unique.append(item)
    if len(unique) == 1:
        return {"macro_id": unique[0]["macro_id"], "macro_candidates": unique, "status": "resolved"}
    if not unique:
        return {"macro_id": None, "macro_candidates": [], "status": "unresolved"}
    return {"macro_id": None, "macro_candidates": unique, "status": "ambiguous"}


def _panel_structure(caption: str, image_blocks: list[dict]) -> list[dict]:
    labels = re.findall(r"\(([a-zA-Z])\)", caption or "")
    panels = []
    for idx, block in enumerate(image_blocks, start=1):
        panel_label = f"({labels[idx - 1]})" if idx - 1 < len(labels) else None
        panels.append(
            {
                "panel_id": f"panel_{idx}",
                "panel_label": panel_label,
                "source_block_id": block.get("block_id"),
                "description_hint": "",
            }
        )
    return panels


def _count_by_field(items: list[dict], field: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for item in items:
        key = str(item.get(field) or "unknown")
        out[key] = out.get(key, 0) + 1
    return out


def _norm(text: object) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())
