from __future__ import annotations

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.model_client import OpenAICompatClient
from src.progress import log, span
from src.prompt_loader import load_prompt, render_prompt


EXPECTED_KC_DENSITIES = {"low", "medium", "high"}


def decompose_extraction_units(
    paper_id: str,
    sections: list[dict],
    client: OpenAICompatClient,
) -> dict:
    if not client or not client.is_ready():
        raise RuntimeError("Extraction Unit decomposition requires a configured online model client.")
    if not sections:
        raise ValueError("Cannot decompose extraction units from empty sections.")

    windows = _build_section_windows(sections)
    if not windows:
        raise ValueError("No valid section windows were built for Extraction Unit decomposition.")

    tpl = load_prompt("decompose_extraction_units.txt")
    unit_max_chars_soft = _env_positive_int("UNIT_MAX_CHARS_SOFT", 2500)
    max_workers = min(_env_positive_int("UNIT_DECOMP_WORKERS", 3), len(windows))

    units_by_window: dict[str, list[dict]] = {}
    skipped_spans_by_window: dict[str, list[dict]] = {}
    errors: list[str] = []

    def run_one(window: dict) -> tuple[str, list[dict], list[dict]]:
        window_id = window["source_window_id"]
        user_prompt = render_prompt(
            tpl,
            unit_max_chars_soft=str(unit_max_chars_soft),
            window_json=json.dumps(_window_prompt_payload(window), ensure_ascii=False, indent=2),
        )
        with span(
            "decompose extraction units",
            section_id=window.get("section_id"),
            window_id=window_id,
            chars=len(window.get("text", "")),
        ):
            result = client.chat_json(
                system_prompt="You decompose paper sections into semantic Extraction Units. Return JSON only.",
                user_prompt=user_prompt,
                temperature=0.1,
            )
        units, skipped_spans = _normalize_window_result(window, result)
        return window_id, units, skipped_spans

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(run_one, window): window for window in windows}
        for fut in as_completed(futures):
            window = futures[fut]
            window_id = window["source_window_id"]
            try:
                out_window_id, units, skipped_spans = fut.result()
                units_by_window[out_window_id] = units
                skipped_spans_by_window[out_window_id] = skipped_spans
                log(
                    "extraction unit window completed",
                    section_id=window.get("section_id"),
                    window_id=window_id,
                    units=len(units),
                    skipped=len(skipped_spans),
                )
            except Exception as exc:
                errors.append(f"{window_id}: {type(exc).__name__}: {exc}")
                log("extraction unit window error", window_id=window_id, error=f"{type(exc).__name__}: {exc}")

    if errors:
        raise RuntimeError("Extraction Unit decomposition failed: " + "; ".join(errors[:5]))

    units = _assign_unit_ids(windows, units_by_window)
    skipped_spans = [
        {
            "section_id": window.get("section_id"),
            "section_title": window.get("section_title"),
            "source_window_id": window["source_window_id"],
            **span_item,
        }
        for window in windows
        for span_item in skipped_spans_by_window.get(window["source_window_id"], [])
    ]
    _validate_section_coverage(sections, units, skipped_spans)

    payload = {
        "paper_id": paper_id,
        "units": units,
        "skipped_spans": skipped_spans,
        "config": {
            "unit_decomp_max_chars": _env_positive_int("UNIT_DECOMP_MAX_CHARS", 12000),
            "unit_decomp_window_chars": _env_positive_int("UNIT_DECOMP_WINDOW_CHARS", 10000),
            "unit_decomp_window_overlap_paragraphs": _env_nonnegative_int(
                "UNIT_DECOMP_WINDOW_OVERLAP_PARAGRAPHS",
                1,
            ),
            "unit_max_chars_soft": unit_max_chars_soft,
            "unit_decomp_workers": max_workers,
        },
    }
    log("extraction units decomposed", units=len(units), skipped=len(skipped_spans))
    return payload


def _build_section_windows(sections: list[dict]) -> list[dict]:
    max_chars = _env_positive_int("UNIT_DECOMP_MAX_CHARS", 12000)
    window_chars = _env_positive_int("UNIT_DECOMP_WINDOW_CHARS", 10000)
    overlap_paragraphs = _env_nonnegative_int("UNIT_DECOMP_WINDOW_OVERLAP_PARAGRAPHS", 1)
    if window_chars > max_chars:
        raise ValueError("UNIT_DECOMP_WINDOW_CHARS must be <= UNIT_DECOMP_MAX_CHARS.")

    windows: list[dict] = []
    for section_index, section in enumerate(sections):
        section_id = str(section.get("section_id", "")).strip()
        title = str(section.get("title", "")).strip()
        text = str(section.get("text", "")).strip()
        if not section_id:
            raise ValueError("Every section must contain a non-empty section_id.")
        if not text:
            continue
        if len(text) <= max_chars:
            windows.append(
                {
                    "section_id": section_id,
                    "section_title": title,
                    "section_index": section_index,
                    "source_window_id": f"{section_id}_W1",
                    "window_order": 1,
                    "text": text,
                }
            )
            continue
        for window_order, window_text in enumerate(
            _coarse_windows(text, window_chars, overlap_paragraphs),
            start=1,
        ):
            windows.append(
                {
                    "section_id": section_id,
                    "section_title": title,
                    "section_index": section_index,
                    "source_window_id": f"{section_id}_W{window_order}",
                    "window_order": window_order,
                    "text": window_text,
                }
            )
    return windows


def _coarse_windows(text: str, limit: int, overlap_paragraphs: int) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paragraphs:
        return [text]

    windows: list[str] = []
    current: list[str] = []
    for paragraph in paragraphs:
        if len(paragraph) > limit:
            if current:
                windows.append("\n\n".join(current).strip())
                current = []
            windows.extend(_split_long_paragraph(paragraph, limit))
            continue
        candidate = "\n\n".join(current + [paragraph]).strip()
        if current and len(candidate) > limit:
            windows.append("\n\n".join(current).strip())
            current = current[-overlap_paragraphs:] if overlap_paragraphs > 0 else []
            current.append(paragraph)
        else:
            current.append(paragraph)
    if current:
        windows.append("\n\n".join(current).strip())
    return [window for window in windows if window]


def _split_long_paragraph(text: str, limit: int) -> list[str]:
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + limit)
        if end < len(text):
            split = max(text.rfind(". ", start, end), text.rfind("; ", start, end), text.rfind("\n", start, end))
            if split > start + limit // 2:
                end = split + 1
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = end
    return [chunk for chunk in chunks if chunk]


def _window_prompt_payload(window: dict) -> dict:
    return {
        "section_id": window.get("section_id"),
        "section_title": window.get("section_title"),
        "source_window_id": window.get("source_window_id"),
        "window_order": window.get("window_order"),
        "text": window.get("text"),
    }


def _normalize_window_result(window: dict, result: dict) -> tuple[list[dict], list[dict]]:
    raw_units = result.get("units", [])
    if not isinstance(raw_units, list):
        raise ValueError("Extraction Unit response must contain a units list.")
    raw_skipped = result.get("skipped_spans", [])
    if raw_skipped is None:
        raw_skipped = []
    if not isinstance(raw_skipped, list):
        raise ValueError("Extraction Unit response skipped_spans must be a list.")

    units = []
    for order, item in enumerate(raw_units, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Extraction Unit #{order} must be an object.")
        source_text = str(item.get("source_text", "")).strip()
        if not source_text:
            raise ValueError(f"Extraction Unit #{order} in {window['source_window_id']} must include source_text.")
        if not _source_text_matches_window(source_text, str(window.get("text", ""))):
            raise ValueError(
                f"Extraction Unit #{order} in {window['source_window_id']} source_text is not found in the source window."
            )
        unit_title = str(item.get("unit_title", "")).strip()
        unit_summary = str(item.get("unit_summary", "")).strip()
        coverage_note = str(item.get("coverage_note", "")).strip()
        if not unit_title or not unit_summary or not coverage_note:
            raise ValueError(
                f"Extraction Unit #{order} in {window['source_window_id']} must include title, summary, and coverage_note."
            )
        density = str(item.get("expected_kc_density", "medium")).strip().lower()
        if density not in EXPECTED_KC_DENSITIES:
            raise ValueError(
                f"Extraction Unit #{order} in {window['source_window_id']} has invalid expected_kc_density: {density}"
            )
        units.append(
            {
                "section_id": window.get("section_id"),
                "section_title": window.get("section_title"),
                "source_window_id": window.get("source_window_id"),
                "window_order": window.get("window_order"),
                "order_in_window": order,
                "unit_title": unit_title,
                "unit_summary": unit_summary,
                "source_text": source_text,
                "start_hint": str(item.get("start_hint") or source_text[:100]).strip(),
                "end_hint": str(item.get("end_hint") or source_text[-100:]).strip(),
                "related_categories": _string_list(item.get("related_categories", [])),
                "coverage_note": coverage_note,
                "expected_kc_density": density,
            }
        )

    skipped_spans = []
    for item in raw_skipped:
        if not isinstance(item, dict):
            raise ValueError(f"skipped_spans in {window['source_window_id']} must contain objects.")
        reason = str(item.get("reason", "")).strip()
        if not reason:
            raise ValueError(f"skipped_spans in {window['source_window_id']} must include reason.")
        skipped_spans.append(
            {
                "reason": reason,
                "text_preview": str(item.get("text_preview", "")).strip(),
            }
        )

    if not units and not skipped_spans:
        raise ValueError(f"{window['source_window_id']} returned neither units nor skipped_spans.")
    return units, skipped_spans


def _assign_unit_ids(windows: list[dict], units_by_window: dict[str, list[dict]]) -> list[dict]:
    section_counts: dict[str, int] = {}
    units: list[dict] = []
    for window in sorted(windows, key=lambda item: (item["section_index"], item["window_order"])):
        section_id = window["section_id"]
        for item in units_by_window.get(window["source_window_id"], []):
            section_counts[section_id] = section_counts.get(section_id, 0) + 1
            order_in_section = section_counts[section_id]
            units.append(
                {
                    "unit_id": f"U_{section_id}_{order_in_section:03d}",
                    "order_in_section": order_in_section,
                    **item,
                }
            )
    return units


def _validate_section_coverage(sections: list[dict], units: list[dict], skipped_spans: list[dict]) -> None:
    unit_sections = {unit.get("section_id") for unit in units}
    skipped_sections = {item.get("section_id") for item in skipped_spans}
    missing = []
    for section in sections:
        section_id = str(section.get("section_id", "")).strip()
        text = str(section.get("text", "")).strip()
        if not section_id or not text:
            continue
        if section_id not in unit_sections and section_id not in skipped_sections:
            missing.append(section_id)
    if missing:
        raise ValueError(f"Extraction Unit decomposition did not cover sections: {missing}")


def _source_text_matches_window(source_text: str, window_text: str) -> bool:
    source_norm = _normalize_ws(source_text)
    window_norm = _normalize_ws(window_text)
    return bool(source_norm and source_norm in window_norm)


def _normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _string_list(values: object) -> list[str]:
    if not isinstance(values, list):
        return []
    out = []
    for value in values:
        text = str(value).strip()
        if text:
            out.append(text)
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
