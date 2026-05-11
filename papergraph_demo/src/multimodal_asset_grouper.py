from __future__ import annotations


HTML_ISLAND_BLOCK_TYPES = {
    "table_html",
    "image_html",
    "markdown_image",
    "caption_html",
    "html_div",
}

MEDIA_BLOCK_TYPES = {"table_html", "image_html", "markdown_image"}


def group_multimodal_assets(paper_id: str, blocks: list[dict], sections: list[dict]) -> dict:
    """
    Group OCR multimodal evidence by the input's structural invariant:
    consecutive HTML/media blocks belong to one island; ordinary text breaks it.

    This deliberately avoids guessing caption/media associations. The island is
    the stable boundary; an LLM analyzer handles the structure inside each group.
    """
    section_titles = {sec.get("section_id"): sec.get("title") for sec in sections}
    groups: list[dict] = []
    idx = 0
    while idx < len(blocks):
        block = blocks[idx]
        if block.get("block_type") not in HTML_ISLAND_BLOCK_TYPES:
            idx += 1
            continue

        island: list[dict] = []
        while idx < len(blocks) and blocks[idx].get("block_type") in HTML_ISLAND_BLOCK_TYPES:
            island.append(blocks[idx])
            idx += 1

        if not any(item.get("block_type") in MEDIA_BLOCK_TYPES for item in island):
            continue

        section_id = _first_nonempty([item.get("section_id") for item in island])
        section_title = section_titles.get(section_id) or _first_nonempty([item.get("section_title") for item in island])
        media_blocks = [_media_block(item) for item in island if item.get("block_type") in MEDIA_BLOCK_TYPES]
        html_blocks = [_html_block(item) for item in island]
        group_type = _asset_group_type(media_blocks)
        groups.append(
            {
                "asset_group_id": f"MAG_{section_id or 'SNA'}_{len(groups) + 1:04d}",
                "section_id": section_id,
                "section_title": section_title,
                "asset_group_type": group_type,
                "source_block_ids": [item.get("block_id") for item in island],
                "html_blocks": html_blocks,
                "media_blocks": media_blocks,
                # Kept for compatibility with old inspection code; semantic
                # caption binding now comes from multimodal_html_group_analyzer.
                "caption_blocks": [],
                "nearby_context_blocks": [],
                "diagnostics": {
                    "html_island_block_count": len(island),
                    "media_block_count": len(media_blocks),
                    "contains_table": any(item.get("media_type") == "table" for item in media_blocks),
                    "contains_image": any(item.get("media_type") == "image" for item in media_blocks),
                },
            }
        )
    return {
        "paper_id": paper_id,
        "schema_version": "v2_html_islands",
        "asset_groups": groups,
        "summary": {
            "asset_group_count": len(groups),
            "by_type": _count_by_field(groups, "asset_group_type"),
        },
    }


def _html_block(block: dict) -> dict:
    return {
        "block_id": block.get("block_id"),
        "block_type": block.get("block_type"),
        "text": block.get("text", ""),
        "html": block.get("html", ""),
        "image_path": block.get("image_path"),
        "resolved_image_path": block.get("resolved_image_path"),
        "caption_kind_hint": block.get("caption_kind"),
        "caption_number_hint": block.get("caption_number"),
    }


def _media_block(block: dict) -> dict:
    block_type = block.get("block_type")
    media_type = "table" if block_type == "table_html" else "image"
    return {
        "block_id": block.get("block_id"),
        "media_type": media_type,
        "html": block.get("html", ""),
        "image_path": block.get("image_path"),
        "resolved_image_path": block.get("resolved_image_path"),
        "text": block.get("text", ""),
    }


def _asset_group_type(media_blocks: list[dict]) -> str:
    kinds = {item.get("media_type") for item in media_blocks}
    if kinds == {"table"}:
        return "table_group"
    if kinds == {"image"}:
        return "figure_group"
    return "mixed_group"


def _count_by_field(items: list[dict], field: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for item in items:
        key = str(item.get(field) or "unknown")
        out[key] = out.get(key, 0) + 1
    return out


def _first_nonempty(values: list[object]) -> object:
    for value in values:
        if value:
            return value
    return None
