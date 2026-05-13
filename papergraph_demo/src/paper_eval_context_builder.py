from __future__ import annotations

import re

from src.paper_parser import _drop_author_affiliation_block, _drop_code_link_lines, _drop_reference_tail


TEXT_BLOCK_TYPES = {"heading", "paragraph"}
MEDIA_BLOCK_TYPES = {"table_html", "image_html", "markdown_image"}
CAPTION_BLOCK_TYPES = {"caption_html", "caption_text"}


def build_eval_paper_context(
    paper_id: str,
    blocks: list[dict],
    assets_payload: dict | None,
    explanations_payload: dict | None,
) -> dict:
    assets = assets_payload.get("assets", []) if isinstance(assets_payload, dict) else []
    explanations = explanations_payload.get("asset_explanations", []) if isinstance(explanations_payload, dict) else []
    explanation_by_asset = {
        str(item.get("asset_id")): item
        for item in explanations
        if isinstance(item, dict) and item.get("asset_id")
    }
    insertion_by_block, covered_asset_blocks = _asset_insertions(assets, explanation_by_asset)

    parts: list[str] = []
    inserted_assets: list[str] = []
    for block in sorted(blocks, key=lambda item: int(item.get("order") or 0)):
        block_id = block.get("block_id")
        if block_id in insertion_by_block:
            asset = insertion_by_block[block_id]
            parts.append(_render_asset(asset, explanation_by_asset.get(str(asset.get("asset_id")))))
            inserted_assets.append(str(asset.get("asset_id")))
            continue
        if block_id in covered_asset_blocks:
            continue
        if _is_reference_heading(block):
            break
        rendered = _render_text_block(block)
        if rendered:
            parts.append(rendered)

    text = "\n\n".join(part.strip() for part in parts if part and part.strip())
    text = _clean_eval_context_text(text)
    return {
        "paper_id": paper_id,
        "schema_version": "v1_eval_paper_context",
        "text": text,
        "summary": {
            "block_count": len(blocks),
            "asset_count": len(assets),
            "inserted_asset_count": len(inserted_assets),
            "inserted_asset_ids": inserted_assets,
        },
    }


def _asset_insertions(assets: list[dict], explanation_by_asset: dict[str, dict]) -> tuple[dict[str, dict], set[str]]:
    insertion_by_block: dict[str, dict] = {}
    covered: set[str] = set()
    for asset in assets:
        if not isinstance(asset, dict):
            continue
        block_ids = _asset_block_ids(asset)
        if not block_ids:
            raise ValueError(f"Multimodal asset {asset.get('asset_id')} has no source/media/table block ids.")
        first_block = block_ids[0]
        if first_block in insertion_by_block:
            raise ValueError(f"Multiple multimodal assets start at block {first_block}.")
        insertion_by_block[first_block] = asset
        covered.update(block_ids)
        explanation = explanation_by_asset.get(str(asset.get("asset_id")))
        if explanation:
            covered.update(_string_list(explanation.get("source_block_ids", [])))
    return insertion_by_block, covered


def _asset_block_ids(asset: dict) -> list[str]:
    preferred = (
        _string_list(asset.get("media_block_ids", []))
        or _string_list(asset.get("table_block_ids", []))
        or _string_list(asset.get("image_block_ids", []))
        or _string_list(asset.get("source_block_ids", []))
    )
    all_source = _string_list(asset.get("source_block_ids", []))
    return list(dict.fromkeys(preferred + all_source))


def _render_asset(asset: dict, explanation: dict | None) -> str:
    asset_type = str(asset.get("asset_type") or "").strip().lower()
    if asset_type == "figure":
        return _render_figure(asset, explanation)
    if asset_type == "table":
        return _render_table(asset, explanation)
    raise ValueError(f"Unsupported multimodal asset type for eval context: {asset_type!r}")


def _render_figure(asset: dict, explanation: dict | None) -> str:
    label = _asset_label(asset, "Figure")
    summary = _first_nonempty(
        (explanation or {}).get("summary"),
        asset.get("nearby_context"),
        asset.get("caption"),
    )
    lines = [
        f"[{label} was here. The original paper contained a figure at this position.",
    ]
    if summary:
        lines[-1] += f" Brief visual description: {summary}"
    lines[-1] += "]"
    caption = _first_nonempty(asset.get("caption"), (explanation or {}).get("caption"))
    if caption:
        lines.append(f"Caption: {caption}")
    key_elements = (explanation or {}).get("key_elements") or []
    if key_elements:
        lines.append("Key visible elements:")
        for item in key_elements[:8]:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "").strip()
            role = str(item.get("role") or "").strip()
            if name and role:
                lines.append(f"- {name}: {role}")
            elif name:
                lines.append(f"- {name}")
    return "\n".join(lines)


def _render_table(asset: dict, explanation: dict | None) -> str:
    label = _asset_label(asset, "Table")
    lines = [f"[{label} was here. The original paper contained a table at this position. The normalized table content is preserved below.]"]
    caption = _first_nonempty(asset.get("caption"), (explanation or {}).get("caption"))
    if caption:
        lines.append(f"Caption: {caption}")
    summary = _first_nonempty((explanation or {}).get("summary"), asset.get("nearby_context"))
    if summary:
        lines.append(f"Summary: {summary}")
    markdown = str(asset.get("normalized_markdown") or "").strip()
    if not markdown:
        raise ValueError(f"Table asset {asset.get('asset_id')} has no normalized_markdown for eval context.")
    lines.append(markdown)
    return "\n\n".join(lines)


def _render_text_block(block: dict) -> str:
    block_type = block.get("block_type")
    text = str(block.get("text") or "").strip()
    if not text:
        return ""
    if block_type in TEXT_BLOCK_TYPES or block_type in CAPTION_BLOCK_TYPES:
        return text
    if block_type in MEDIA_BLOCK_TYPES or block_type == "html_div":
        return ""
    return text


def _asset_label(asset: dict, default_prefix: str) -> str:
    caption_kind = str(asset.get("caption_kind") or "").strip()
    caption_number = str(asset.get("caption_number") or "").strip()
    if caption_kind and caption_number:
        return f"{caption_kind.capitalize()} {caption_number}"
    return f"{default_prefix} {asset.get('asset_id')}"


def _clean_eval_context_text(text: str) -> str:
    text = _drop_author_affiliation_block(text)
    text = _drop_reference_tail(text)
    text = _drop_code_link_lines(text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def _is_reference_heading(block: dict) -> bool:
    if block.get("block_type") != "heading":
        return False
    text = str(block.get("text") or "").strip().lower()
    return bool(re.match(r"^#{1,6}\s*(references|bibliography|appendix|acknowledg(e)?ments?)\b", text))


def _first_nonempty(*values: object) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _string_list(values: object) -> list[str]:
    if not isinstance(values, list):
        return []
    return [str(value).strip() for value in values if str(value or "").strip()]
