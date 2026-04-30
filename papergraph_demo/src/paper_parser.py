from __future__ import annotations

import re
from pathlib import Path


def load_paper_text(path: Path) -> str:
    return _clean_markdown(path.read_text(encoding="utf-8"))


def load_paper_text_from_dir(directory: Path) -> str:
    md_files = sorted(directory.glob("doc_*.md"))
    if not md_files:
        md_files = sorted(directory.glob("*.md"))
    if not md_files:
        raise FileNotFoundError(f"No markdown files found in: {directory}")

    parts: list[str] = []
    for idx, md_file in enumerate(md_files, start=1):
        content = md_file.read_text(encoding="utf-8")
        cleaned = _clean_markdown(content)
        parts.append(f"\n\n<!-- page {idx}: {md_file.name} -->\n\n{cleaned}")
    return "".join(parts).strip()


def _clean_markdown(text: str) -> str:
    # Remove heavy HTML render noise from OCR markdown.
    text = re.sub(r"<div[^>]*>.*?</div>", " ", text, flags=re.DOTALL)
    text = re.sub(r"<table[^>]*>.*?</table>", " ", text, flags=re.DOTALL)
    text = re.sub(r"<img[^>]*>", " ", text, flags=re.DOTALL)
    text = re.sub(r"!\[[^\]]*\]\([^)]+\)", " ", text)
    text = re.sub(r"\bFigure\s+\d+[:\.\s].*", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\bTable\s+\d+[:\.\s].*", " ", text, flags=re.IGNORECASE)
    text = _drop_author_affiliation_block(text)
    text = _drop_reference_tail(text)
    text = _drop_code_link_lines(text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def _drop_author_affiliation_block(text: str) -> str:
    """
    Keep title and scientific content; drop author/affiliation/email lines
    between title and abstract/introduction when present.
    """
    lines = text.splitlines()
    if not lines:
        return text

    start_idx = None
    for i, ln in enumerate(lines):
        l = ln.strip().lower()
        if l.startswith("## abstract") or re.match(r"^##\s*\d+\.?\s*introduction", l):
            start_idx = i
            break
    if start_idx is None or start_idx <= 1:
        return text

    cleaned = [lines[0]]
    for ln in lines[1:start_idx]:
        l = ln.strip()
        low = l.lower()
        if not l:
            continue
        if "@" in l:
            continue
        if re.search(r"\$\s*\^\{?\d+", l):
            continue
        if any(k in low for k in ["university", "school of", "department", "china", "singapore"]):
            continue
        # Keep potential useful short lines, but most meta block gets filtered.
        if len(l.split()) <= 2:
            continue
        cleaned.append(ln)

    cleaned.extend(lines[start_idx:])
    return "\n".join(cleaned)


def _drop_reference_tail(text: str) -> str:
    """
    Remove references/bibliography and trailing metadata sections.
    """
    pat = re.compile(
        r"\n#{1,6}\s*(references|bibliography|appendix|acknowledg(e)?ments?)\b.*$",
        flags=re.IGNORECASE | re.DOTALL,
    )
    m = pat.search(text)
    if m:
        return text[: m.start()].strip()
    return text


def _drop_code_link_lines(text: str) -> str:
    lines = []
    for ln in text.splitlines():
        l = ln.strip().lower()
        if l.startswith("code ") or l.startswith("code:") or "github.com" in l:
            continue
        lines.append(ln)
    return "\n".join(lines)


def split_into_sections(paper_text: str) -> list[dict]:
    """
    Guidance-aligned section splitting:
    outputs [{"section_id":"S1","title":"Abstract","text":"..."}]
    """
    lines = paper_text.splitlines()
    sections: list[dict] = []
    current_title = "Preamble"
    current_buf: list[str] = []

    def flush() -> None:
        nonlocal current_buf, current_title
        txt = "\n".join(current_buf).strip()
        if txt:
            sections.append(
                {
                    "section_id": f"S{len(sections) + 1}",
                    "title": current_title,
                    "text": txt,
                }
            )
        current_buf = []

    header_pat = re.compile(r"^\s{0,3}#{1,6}\s+(.+?)\s*$")
    for ln in lines:
        m = header_pat.match(ln)
        if m:
            title = m.group(1).strip()
            if _is_noise_heading(title):
                continue
            flush()
            current_title = title
        else:
            current_buf.append(ln)
    flush()

    if not sections:
        sections = [{"section_id": "S1", "title": "FullText", "text": paper_text.strip()}]
    return _merge_short_sections(sections)


def _is_noise_heading(title: str) -> bool:
    title_l = title.strip().lower()
    if re.match(r"^page\s+\d+\b", title_l):
        return True
    if title_l in {"acknowledgments", "acknowledgements"}:
        return True
    return False


def _merge_short_sections(sections: list[dict]) -> list[dict]:
    merged: list[dict] = []
    for sec in sections:
        text = sec.get("text", "").strip()
        if not text:
            continue
        title = sec.get("title", "")
        # OCR page turns often leave a tiny orphan before the next real heading.
        if merged and len(text) < 180 and not _looks_like_major_section(title):
            merged[-1]["text"] = (merged[-1]["text"].rstrip() + "\n\n" + text).strip()
            continue
        merged.append({"section_id": "", "title": title, "text": text})
    for idx, sec in enumerate(merged, start=1):
        sec["section_id"] = f"S{idx}"
    return merged


def _looks_like_major_section(title: str) -> bool:
    t = title.lower()
    return bool(
        re.match(r"^(\d+(\.\d+)*)\s+", t)
        or t in {"abstract", "introduction", "conclusion", "limitations", "references"}
    )
