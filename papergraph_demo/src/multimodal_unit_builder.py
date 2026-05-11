from __future__ import annotations

from copy import deepcopy


def augment_extraction_units_with_multimodal_units(extraction_units: dict, kc_bank: dict) -> dict:
    """Add virtual Extraction Units for multimodal KC nodes.

    Multimodal KCs are already assigned stable unit IDs such as MMU_FIG_001.
    Edge construction needs matching unit records so candidate generation and
    verification can cite paragraph IDs instead of silently skipping them.
    """
    units = deepcopy(extraction_units.get("units", []))
    if not isinstance(units, list):
        raise ValueError("extraction_units.units must be a list.")

    existing_ids = {
        str(unit.get("unit_id", "")).strip()
        for unit in units
        if isinstance(unit, dict) and str(unit.get("unit_id", "")).strip()
    }
    multimodal_by_unit: dict[str, list[dict]] = {}
    for kc in kc_bank.get("kc_nodes", []):
        if not isinstance(kc, dict) or not bool(kc.get("modality", {}).get("is_multimodal")):
            continue
        unit_id = str(kc.get("unit_id", "")).strip()
        asset_id = str(kc.get("asset_id", "")).strip()
        if not unit_id:
            raise ValueError(f"Multimodal KC {kc.get('kc_id')} has no unit_id.")
        if not asset_id:
            raise ValueError(f"Multimodal KC {kc.get('kc_id')} has no asset_id.")
        multimodal_by_unit.setdefault(unit_id, []).append(kc)

    if not multimodal_by_unit:
        out = dict(extraction_units)
        out["units"] = units
        return out

    max_window_order = _max_int(units, "window_order")
    max_order_in_section = _max_int(units, "order_in_section")
    added = []
    for offset, (unit_id, kcs) in enumerate(sorted(multimodal_by_unit.items()), start=1):
        if unit_id in existing_ids:
            continue
        unit = _virtual_unit(unit_id, kcs, max_window_order + offset, max_order_in_section + offset)
        units.append(unit)
        added.append(unit_id)

    out = dict(extraction_units)
    out["units"] = units
    metadata = dict(out.get("metadata", {})) if isinstance(out.get("metadata"), dict) else {}
    metadata["multimodal_virtual_units_added"] = added
    metadata["multimodal_virtual_unit_count"] = len(added)
    out["metadata"] = metadata
    return out


def _virtual_unit(unit_id: str, kcs: list[dict], window_order: int, order_in_section: int) -> dict:
    asset_ids = {str(kc.get("asset_id", "")).strip() for kc in kcs if str(kc.get("asset_id", "")).strip()}
    if len(asset_ids) != 1:
        raise ValueError(f"Multimodal unit {unit_id} must map to exactly one asset_id, got {sorted(asset_ids)}.")
    asset_types = {str(kc.get("asset_type", "")).strip() for kc in kcs if str(kc.get("asset_type", "")).strip()}
    if len(asset_types) != 1:
        raise ValueError(f"Multimodal unit {unit_id} must map to exactly one asset_type, got {sorted(asset_types)}.")

    ordered_kcs = sorted(kcs, key=lambda kc: _kc_sort_key(str(kc.get("kc_id", ""))))
    first = ordered_kcs[0]
    asset_id = next(iter(asset_ids))
    asset_type = next(iter(asset_types))
    caption = str(first.get("asset_caption") or "").strip()
    summary = str(first.get("asset_summary") or "").strip()
    section_id = str(first.get("source_section_id") or first.get("section_id") or "").strip()
    section_title = str(first.get("source_section") or first.get("section") or section_id).strip()
    evidence_bases = _unique_strings(kc.get("asset_evidence_basis") for kc in ordered_kcs)
    source_bases = _unique_nested_strings(kc.get("asset_source_basis") for kc in ordered_kcs)
    possible_misreadings = _unique_nested_strings(kc.get("asset_possible_misreadings") for kc in ordered_kcs)

    paragraphs = [
        f"Asset {asset_id} ({asset_type}).",
        f"Caption: {caption}" if caption else "",
        f"Asset summary: {summary}" if summary else "",
        _format_list("Evidence bases", evidence_bases),
        _format_list("Source bases", source_bases),
        _format_list(
            "Supported KC claims",
            [
                f"{kc.get('kc_id')}: {kc.get('full_claim') or kc.get('claim')}"
                for kc in ordered_kcs
                if str(kc.get("full_claim") or kc.get("claim") or "").strip()
            ],
        ),
        _format_list("Possible misreadings", possible_misreadings),
    ]
    source_text = "\n\n".join(paragraph for paragraph in paragraphs if paragraph.strip())
    if not source_text.strip():
        raise ValueError(f"Multimodal unit {unit_id} produced empty source_text.")

    return {
        "unit_id": unit_id,
        "order_in_section": order_in_section,
        "section_id": section_id,
        "section_title": section_title,
        "source_window_id": unit_id,
        "window_order": window_order,
        "order_in_window": 1,
        "unit_title": _unit_title(asset_id, asset_type, caption),
        "unit_summary": summary or caption,
        "source_text": source_text,
        "paragraph_ids": [f"P{idx}" for idx in range(1, len([p for p in paragraphs if p.strip()]) + 1)],
        "paragraph_ids_original": [f"P{idx}" for idx in range(1, len([p for p in paragraphs if p.strip()]) + 1)],
        "paragraph_id_normalization": "as_provided",
        "start_hint": source_text.split("\n\n", 1)[0],
        "end_hint": source_text.rsplit("\n\n", 1)[-1],
        "related_categories": sorted({str(kc.get("type", "")).strip() for kc in ordered_kcs if kc.get("type")}),
        "coverage_note": f"Virtual multimodal unit generated from {asset_id} for edge construction.",
        "expected_kc_density": "high",
        "is_multimodal_unit": True,
        "asset_id": asset_id,
        "asset_type": asset_type,
        "kc_ids": [kc.get("kc_id") for kc in ordered_kcs],
    }


def _unit_title(asset_id: str, asset_type: str, caption: str) -> str:
    if caption:
        trimmed = caption[:120].rstrip()
        return f"{asset_type.title()} {asset_id}: {trimmed}"
    return f"{asset_type.title()} {asset_id}"


def _format_list(title: str, values: list[str]) -> str:
    if not values:
        return ""
    return title + ":\n" + "\n".join(f"- {value}" for value in values)


def _unique_strings(values: object) -> list[str]:
    seen = set()
    out = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out


def _unique_nested_strings(values: object) -> list[str]:
    seen = set()
    out = []
    for value in values:
        if isinstance(value, list):
            items = value
        else:
            items = [value]
        for item in items:
            text = _nested_item_text(item)
            if text and text not in seen:
                seen.add(text)
                out.append(text)
    return out


def _nested_item_text(item: object) -> str:
    if isinstance(item, dict):
        claim = str(item.get("claim") or "").strip()
        why_wrong = str(item.get("why_wrong") or item.get("reason") or "").strip()
        if claim and why_wrong:
            return f"{claim}; why wrong: {why_wrong}"
        if claim:
            return claim
        parts = [f"{key}: {value}" for key, value in item.items() if str(value).strip()]
        return "; ".join(parts).strip()
    return str(item or "").strip()


def _max_int(items: list[dict], key: str) -> int:
    values = []
    for item in items:
        try:
            values.append(int(item.get(key) or 0))
        except (TypeError, ValueError):
            continue
    return max(values, default=0)


def _kc_sort_key(kc_id: str) -> tuple[int, str]:
    digits = ""
    for char in reversed(kc_id):
        if not char.isdigit():
            break
        digits = char + digits
    return (int(digits) if digits else 10**9, kc_id)
