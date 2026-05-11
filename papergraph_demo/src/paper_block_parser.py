from __future__ import annotations

import html
import re
from pathlib import Path

from src.paper_parser import _clean_markdown


_CAPTION_RE = re.compile(r"^\s*(Figure|Table)\s+([0-9]+[A-Za-z]?)\b[:.\s-]*", re.IGNORECASE)
_IMG_SRC_RE = re.compile(r"\bsrc\s*=\s*(['\"])(.*?)\1", re.IGNORECASE | re.DOTALL)
_MD_IMAGE_RE = re.compile(r"!\[[^\]]*]\(([^)]+)\)")


def load_paper_bundle_from_dir(directory: Path) -> dict:
    md_files = sorted(directory.glob("doc_*.md"))
    if not md_files:
        md_files = sorted(directory.glob("*.md"))
    if not md_files:
        raise FileNotFoundError(f"No markdown files found in: {directory}")

    clean_parts: list[str] = []
    blocks: list[dict] = []
    for page_index, md_file in enumerate(md_files, start=1):
        content = md_file.read_text(encoding="utf-8")
        cleaned = _clean_markdown(content)
        clean_parts.append(f"\n\n<!-- page {page_index}: {md_file.name} -->\n\n{cleaned}")
        blocks.extend(parse_markdown_blocks(content, page_index, md_file, directory))

    return {
        "source_type": "directory",
        "source_path": str(directory),
        "clean_text": "".join(clean_parts).strip(),
        "blocks": _renumber_global_order(blocks),
    }


def load_paper_bundle_from_file(path: Path) -> dict:
    content = path.read_text(encoding="utf-8")
    return {
        "source_type": "file",
        "source_path": str(path),
        "clean_text": _clean_markdown(content),
        "blocks": _renumber_global_order(parse_markdown_blocks(content, 1, path, path.parent)),
    }


def parse_markdown_blocks(markdown: str, page_index: int, source_file: Path, base_dir: Path) -> list[dict]:
    page_id = f"P{page_index:03d}"
    blocks: list[dict] = []
    paragraph: list[str] = []

    def flush_paragraph() -> None:
        if not paragraph:
            return
        raw = "\n".join(paragraph).strip()
        paragraph.clear()
        if raw:
            blocks.append(_make_block(raw, page_id, source_file, base_dir, len(blocks) + 1))

    lines = markdown.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if not stripped:
            flush_paragraph()
            i += 1
            continue

        if _starts_html_block(stripped, "table"):
            flush_paragraph()
            raw_lines = [line]
            while "</table>" not in lines[i].lower() and i + 1 < len(lines):
                i += 1
                raw_lines.append(lines[i])
            raw = "\n".join(raw_lines).strip()
            blocks.append(_make_block(raw, page_id, source_file, base_dir, len(blocks) + 1))
            i += 1
            continue

        if _starts_html_block(stripped, "div"):
            flush_paragraph()
            raw_lines = [line]
            while "</div>" not in lines[i].lower() and i + 1 < len(lines):
                i += 1
                raw_lines.append(lines[i])
            raw = "\n".join(raw_lines).strip()
            blocks.append(_make_block(raw, page_id, source_file, base_dir, len(blocks) + 1))
            i += 1
            continue

        if stripped.lower().startswith("<img"):
            flush_paragraph()
            blocks.append(_make_block(line.strip(), page_id, source_file, base_dir, len(blocks) + 1))
            i += 1
            continue

        if _MD_IMAGE_RE.search(stripped):
            flush_paragraph()
            blocks.append(_make_block(line.strip(), page_id, source_file, base_dir, len(blocks) + 1))
            i += 1
            continue

        if re.match(r"^\s{0,3}#{1,6}\s+.+$", line):
            flush_paragraph()
            blocks.append(_make_block(line.strip(), page_id, source_file, base_dir, len(blocks) + 1))
            i += 1
            continue

        paragraph.append(line)
        i += 1

    flush_paragraph()
    return blocks


def _make_block(raw: str, page_id: str, source_file: Path, base_dir: Path, page_order: int) -> dict:
    text = _html_text(raw)
    image_path = _extract_image_path(raw)
    resolved_image_path = _resolve_image_path(image_path, base_dir) if image_path else None
    caption = _caption_info(text)
    block_type = _classify_block(raw, text, image_path, caption)
    return {
        "block_id": f"{page_id}_B{page_order:04d}",
        "page_id": page_id,
        "source_file": source_file.name,
        "page_order": page_order,
        "order": 0,
        "block_type": block_type,
        "text": text,
        "html": raw if _looks_like_html(raw) else "",
        "image_path": image_path,
        "resolved_image_path": str(resolved_image_path) if resolved_image_path else None,
        "caption_kind": caption["kind"],
        "caption_number": caption["number"],
        "section_id": None,
        "section_title": None,
    }


def _classify_block(raw: str, text: str, image_path: str | None, caption: dict) -> str:
    stripped = raw.strip()
    lower = stripped.lower()
    if re.match(r"^\s{0,3}#{1,6}\s+.+$", stripped):
        return "heading"
    if lower.startswith("<table"):
        return "table_html"
    if image_path and lower.startswith("<div"):
        return "image_html"
    if lower.startswith("<img"):
        return "image_html"
    if image_path and _MD_IMAGE_RE.search(stripped):
        return "markdown_image"
    if lower.startswith("<div") and caption["kind"] != "none":
        return "caption_html"
    if lower.startswith("<div"):
        return "html_div"
    if caption["kind"] != "none" and len(text.split()) <= 80:
        return "caption_text"
    return "paragraph"


def _caption_info(text: str) -> dict:
    match = _CAPTION_RE.search(text or "")
    if not match:
        return {"kind": "none", "number": None}
    return {"kind": match.group(1).lower(), "number": match.group(2)}


def _html_text(raw: str) -> str:
    text = re.sub(r"<[^>]+>", " ", raw, flags=re.DOTALL)
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _extract_image_path(raw: str) -> str | None:
    match = _IMG_SRC_RE.search(raw)
    if match:
        return html.unescape(match.group(2).strip())
    match = _MD_IMAGE_RE.search(raw)
    if match:
        return html.unescape(match.group(1).strip())
    return None


def _resolve_image_path(image_path: str, base_dir: Path) -> Path:
    raw = Path(image_path)
    if raw.is_absolute():
        resolved = raw
    else:
        resolved = base_dir / raw
    if not resolved.exists():
        raise FileNotFoundError(f"Image referenced by OCR markdown does not exist: {resolved}")
    return resolved.resolve()


def _starts_html_block(stripped: str, tag: str) -> bool:
    return bool(re.match(rf"^<{tag}\b", stripped, flags=re.IGNORECASE))


def _looks_like_html(raw: str) -> bool:
    return bool(re.match(r"^\s*<\w+", raw))


def _renumber_global_order(blocks: list[dict]) -> list[dict]:
    for order, block in enumerate(blocks, start=1):
        block["order"] = order
    return blocks
