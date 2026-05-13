from __future__ import annotations

import re


def align_blocks_to_sections(paper_id: str, blocks: list[dict], sections: list[dict]) -> dict:
    if not isinstance(blocks, list):
        raise ValueError("blocks must be a list.")
    if not sections:
        raise ValueError("Cannot align Storybench blocks without sections.")

    section_by_title = {_norm(sec.get("title", "")): sec for sec in sections if sec.get("title")}
    section_by_id = {str(sec.get("section_id", "")).strip(): sec for sec in sections if sec.get("section_id")}
    aligned: list[dict] = []
    current_section: dict | None = None

    for block in blocks:
        item = dict(block)
        if item.get("block_type") == "heading":
            title = _heading_title(str(item.get("text") or ""))
            current_section = section_by_title.get(_norm(title)) or section_by_id.get(title)
        if current_section is not None:
            item["section_id"] = current_section.get("section_id")
            item["section_title"] = current_section.get("title")
        aligned.append(item)

    _infer_unassigned_media_sections(aligned)
    diagnostics = _diagnostics(aligned)
    return {
        "paper_id": paper_id,
        "schema_version": "v1",
        "blocks": aligned,
        "summary": {
            "block_count": len(aligned),
            "by_type": _count_by_type(aligned),
        },
        "diagnostics": diagnostics,
    }


def _infer_unassigned_media_sections(blocks: list[dict]) -> None:
    for idx, block in enumerate(blocks):
        if block.get("section_id") or block.get("block_type") not in {
            "table_html",
            "image_html",
            "markdown_image",
            "caption_html",
            "caption_text",
        }:
            continue
        neighbor = _nearest_sectioned_neighbor(blocks, idx)
        if neighbor:
            block["section_id"] = neighbor.get("section_id")
            block["section_title"] = neighbor.get("section_title")
            block["section_alignment"] = "nearest_neighbor"


def _nearest_sectioned_neighbor(blocks: list[dict], idx: int) -> dict | None:
    for distance in range(1, 6):
        before = idx - distance
        after = idx + distance
        if before >= 0 and blocks[before].get("section_id"):
            return blocks[before]
        if after < len(blocks) and blocks[after].get("section_id"):
            return blocks[after]
    return None


def _diagnostics(blocks: list[dict]) -> dict:
    media_types = {"table_html", "image_html", "markdown_image"}
    caption_types = {"caption_html", "caption_text"}
    return {
        "unassigned_asset_block_count": sum(
            1
            for block in blocks
            if block.get("block_type") in media_types | caption_types and not block.get("section_id")
        ),
        "image_path_missing_count": sum(
            1
            for block in blocks
            if block.get("block_type") in {"image_html", "markdown_image"}
            and not block.get("resolved_image_path")
        ),
        "caption_without_media_count": _caption_without_media_count(blocks),
        "media_without_caption_count": _media_without_caption_count(blocks),
    }


def _caption_without_media_count(blocks: list[dict]) -> int:
    count = 0
    for idx, block in enumerate(blocks):
        if block.get("block_type") not in {"caption_html", "caption_text"}:
            continue
        kind = block.get("caption_kind")
        if not _nearby_media(blocks, idx, kind):
            count += 1
    return count


def _media_without_caption_count(blocks: list[dict]) -> int:
    count = 0
    for idx, block in enumerate(blocks):
        if block.get("block_type") not in {"table_html", "image_html", "markdown_image"}:
            continue
        kind = "table" if block.get("block_type") == "table_html" else "figure"
        if not _nearby_caption(blocks, idx, kind):
            count += 1
    return count


def _nearby_media(blocks: list[dict], idx: int, caption_kind: str | None) -> bool:
    wanted = {"table_html"} if caption_kind == "table" else {"image_html", "markdown_image"}
    for distance in range(1, 4):
        if idx - distance >= 0 and blocks[idx - distance].get("block_type") in wanted:
            return True
        if idx + distance < len(blocks) and blocks[idx + distance].get("block_type") in wanted:
            return True
    return False


def _nearby_caption(blocks: list[dict], idx: int, kind: str) -> bool:
    for distance in range(1, 4):
        for pos in (idx - distance, idx + distance):
            if 0 <= pos < len(blocks):
                block = blocks[pos]
                if block.get("block_type") in {"caption_html", "caption_text"} and block.get("caption_kind") == kind:
                    return True
    return False


def _count_by_type(blocks: list[dict]) -> dict[str, int]:
    out: dict[str, int] = {}
    for block in blocks:
        key = str(block.get("block_type") or "unknown")
        out[key] = out.get(key, 0) + 1
    return out


def _heading_title(text: str) -> str:
    return re.sub(r"^\s*#+\s*", "", text).strip()


def _norm(text: object) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())
