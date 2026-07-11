from __future__ import annotations

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.macro_extractor import macro_context_for_prompt
from src.kc_type_registry import TEXT_KC_TYPES, valid_kc_type
from src.model_client import OpenAICompatClient
from src.progress import log, span
from src.prompt_loader import load_prompt, render_prompt


VALID_IMPORTANCE = {"critical", "normal"}

VALID_CLAIM_STRENGTH = {
    "explicit",
    "experimentally_supported",
    "partially_supported",
    "author_hypothesis",
    "plausible_inference",
    "limitation_or_missing_evidence",
}


LABEL_FIELDS = ("macro_id", "type", "importance", "claim_strength")


class InvalidKCLabelError(ValueError):
    def __init__(
        self,
        unit_id: str,
        kc_index: int,
        field_name: str,
        invalid_value: str,
        allowed_values: set[str],
    ) -> None:
        self.unit_id = unit_id
        self.kc_index = kc_index
        self.field_name = field_name
        self.invalid_value = invalid_value
        self.allowed_values = set(allowed_values)
        super().__init__(
            f"KC #{kc_index} in unit {unit_id} has invalid {field_name}={invalid_value!r}; "
            f"allowed={sorted(allowed_values)!r}."
        )


def extract_kc_candidates_by_units(
    extraction_units_payload: dict,
    macro_spine: dict,
    client: OpenAICompatClient,
    return_metadata: bool = False,
) -> list[dict] | dict:
    if not client or not client.is_ready():
        raise RuntimeError("Unit-level KC extraction requires a configured online model client.")

    units = extraction_units_payload.get("units", [])
    if not isinstance(units, list) or not units:
        raise ValueError("Unit-level KC extraction requires extraction_units.json with a non-empty units list.")

    macro_ids = {
        str(macro.get("macro_id", "")).strip()
        for macro in macro_spine.get("macro_nodes", [])
        if str(macro.get("macro_id", "")).strip()
    }
    if not macro_ids:
        raise ValueError("Unit-level KC extraction requires a non-empty Macro Spine.")

    tpl = load_prompt("extract_kc_from_unit.txt")
    label_repair_tpl = load_prompt("repair_kc_labels.txt")
    macro_context_json = json.dumps(macro_context_for_prompt(macro_spine), ensure_ascii=False, indent=2)
    unit_limit = _env_nonnegative_int("KC_PER_UNIT_LIMIT", 0)
    hard_cap = _env_nonnegative_int("KC_PER_UNIT_HARD_CAP", 20)
    response_cap = _effective_response_cap(unit_limit, hard_cap)
    label_repair_attempts = _env_nonnegative_int("KC_LABEL_REPAIR_ATTEMPTS", 2)
    max_workers = min(_env_positive_int("UNIT_KC_WORKERS", 4), len(units))
    errors: list[str] = []
    candidates_by_unit: dict[str, list[dict]] = {}
    empty_units: dict[str, str] = {}

    def run_one(unit: dict) -> tuple[str, list[dict], str]:
        unit_id = str(unit.get("unit_id", "")).strip()
        if not unit_id:
            raise ValueError("Every Extraction Unit must contain unit_id before KC extraction.")
        source_text = str(unit.get("source_text", "")).strip()
        if not source_text:
            raise ValueError(f"Extraction Unit {unit_id} has empty source_text.")

        user_prompt = render_prompt(
            tpl,
            macro_context_json=macro_context_json,
            unit_json=json.dumps(_unit_prompt_payload(unit), ensure_ascii=False, indent=2),
        )
        with span("unit KC extraction", unit_id=unit_id, chars=len(source_text)):
            result = client.chat_json(
                system_prompt="You fully extract evaluable Knowledge Components from one Extraction Unit. Return JSON only.",
                user_prompt=user_prompt,
                temperature=0.1,
            )
        unit_candidates, empty_reason = _normalize_unit_result_with_label_repair(
            unit=unit,
            result=result,
            macro_ids=macro_ids,
            response_cap=response_cap,
            client=client,
            repair_template=label_repair_tpl,
            max_repair_attempts=label_repair_attempts,
        )
        return unit_id, unit_candidates, empty_reason

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(run_one, unit): unit for unit in units}
        for fut in as_completed(futures):
            unit = futures[fut]
            unit_id = str(unit.get("unit_id", "")).strip()
            try:
                out_unit_id, unit_candidates, empty_reason = fut.result()
                candidates_by_unit[out_unit_id] = unit_candidates
                if not unit_candidates:
                    empty_units[out_unit_id] = empty_reason
                log("unit KC extraction completed", unit_id=out_unit_id, candidates=len(unit_candidates))
            except Exception as exc:
                errors.append(f"{unit_id}: {type(exc).__name__}: {exc}")
                log("unit KC extraction error", unit_id=unit_id, error=f"{type(exc).__name__}: {exc}")

    if errors:
        raise RuntimeError("Unit-level KC extraction failed: " + "; ".join(errors[:5]))

    candidates: list[dict] = []
    for unit_index, unit in enumerate(units):
        unit_id = str(unit.get("unit_id", "")).strip()
        for candidate in candidates_by_unit.get(unit_id, []):
            candidates.append(
                {
                    "candidate_id": f"C{len(candidates) + 1}",
                    "unit_id": unit_id,
                    "section": unit.get("section_title", ""),
                    "section_id": unit.get("section_id", ""),
                    "source_chunk_id": unit_id,
                    "source_window_id": unit.get("source_window_id", ""),
                    "unit_title": unit.get("unit_title", ""),
                    "unit_summary": unit.get("unit_summary", ""),
                    "unit_index": unit_index,
                    **candidate,
                }
            )

    if not candidates:
        raise RuntimeError(
            "Unit-level KC extraction produced no candidates. "
            f"Empty units: {json.dumps(empty_units, ensure_ascii=False)[:1200]}"
        )

    log(
        "unit-level KC candidates extracted",
        units=len(units),
        candidates=len(candidates),
        empty_units=len(empty_units),
    )
    if return_metadata:
        return {
            "extraction_source": "unit",
            "kc_candidates": candidates,
            "empty_units": [
                {"unit_id": unit_id, "empty_reason": reason}
                for unit_id, reason in sorted(empty_units.items())
            ],
        }
    return candidates


def _unit_prompt_payload(unit: dict) -> dict:
    return {
        "unit_id": unit.get("unit_id"),
        "section_id": unit.get("section_id"),
        "section_title": unit.get("section_title"),
        "unit_title": unit.get("unit_title"),
        "unit_summary": unit.get("unit_summary"),
        "related_categories": unit.get("related_categories", []),
        "expected_kc_density": unit.get("expected_kc_density"),
        "source_paragraphs": _unit_paragraphs(str(unit.get("source_text", ""))),
    }


def _normalize_unit_result(
    unit: dict,
    result: dict,
    macro_ids: set[str],
    response_cap: int,
) -> tuple[list[dict], str]:
    raw_kcs = result.get("kcs", [])
    if not isinstance(raw_kcs, list):
        raise ValueError("Unit KC extraction response must contain kcs list.")
    if response_cap > 0 and len(raw_kcs) > response_cap:
        raise ValueError(
            f"Unit {unit.get('unit_id')} returned {len(raw_kcs)} KCs, exceeding the configured per-unit cap={response_cap}."
        )
    empty_reason = str(result.get("empty_reason", "")).strip()
    if not raw_kcs:
        if not empty_reason:
            raise ValueError(f"Unit {unit.get('unit_id')} returned no KCs but did not include empty_reason.")
        return [], empty_reason

    source_text = str(unit.get("source_text", "")).strip()
    evidence_paragraphs = _unit_paragraphs(source_text)
    evidence_paragraph_by_id = {paragraph["paragraph_id"]: paragraph for paragraph in evidence_paragraphs}
    out = []
    for idx, item in enumerate(raw_kcs, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"KC #{idx} in unit {unit.get('unit_id')} must be an object.")
        claim = str(item.get("claim", "")).strip()
        evidence_paragraph_ids = _string_list(item.get("evidence_paragraph_ids", []))
        if not claim or not evidence_paragraph_ids:
            raise ValueError(f"KC #{idx} in unit {unit.get('unit_id')} must include claim and evidence_paragraph_ids.")
        original_evidence_paragraph_ids = evidence_paragraph_ids
        evidence_paragraph_ids = _normalize_evidence_paragraph_ids_to_contiguous_range(
            str(unit.get("unit_id")),
            idx,
            evidence_paragraph_ids,
            evidence_paragraph_by_id,
        )
        evidence = "\n\n".join(evidence_paragraph_by_id[paragraph_id]["text"] for paragraph_id in evidence_paragraph_ids)
        unit_id = str(unit.get("unit_id"))
        macro_id = str(item.get("macro_id", "")).strip()
        if macro_id not in macro_ids:
            raise InvalidKCLabelError(unit_id, idx, "macro_id", macro_id, macro_ids)
        raw_kc_type = str(item.get("type", "")).strip()
        kc_type = valid_kc_type(raw_kc_type, TEXT_KC_TYPES)
        if kc_type is None:
            raise InvalidKCLabelError(unit_id, idx, "type", raw_kc_type, set(TEXT_KC_TYPES))
        importance = str(item.get("importance", "")).strip()
        if importance not in VALID_IMPORTANCE:
            raise InvalidKCLabelError(unit_id, idx, "importance", importance, VALID_IMPORTANCE)
        claim_strength = str(item.get("claim_strength", "")).strip()
        if claim_strength not in VALID_CLAIM_STRENGTH:
            raise InvalidKCLabelError(
                unit_id,
                idx,
                "claim_strength",
                claim_strength,
                VALID_CLAIM_STRENGTH,
            )
        scope = item.get("scope")
        if not isinstance(scope, dict) or not scope:
            raise ValueError(f"KC #{idx} in unit {unit.get('unit_id')} must include non-empty scope object.")
        out.append(
            {
                "claim": claim,
                "evidence": evidence,
                "evidence_paragraph_ids": evidence_paragraph_ids,
                "evidence_paragraph_ids_original": original_evidence_paragraph_ids,
                "evidence_paragraph_id_normalization": (
                    "as_provided"
                    if evidence_paragraph_ids == original_evidence_paragraph_ids
                    else "expanded_to_contiguous_range"
                ),
                "macro_id": macro_id,
                "type": kc_type,
                "importance": importance,
                "claim_strength": claim_strength,
                "scope": _normalize_scope(scope),
                "related_terms": _string_list(item.get("related_terms", [])),
            }
        )
    return out, empty_reason


def _normalize_unit_result_with_label_repair(
    unit: dict,
    result: dict,
    macro_ids: set[str],
    response_cap: int,
    client: OpenAICompatClient,
    repair_template: str,
    max_repair_attempts: int,
) -> tuple[list[dict], str]:
    current = result
    allowed_by_field = _allowed_labels(macro_ids)
    for attempt in range(max_repair_attempts + 1):
        try:
            return _normalize_unit_result(unit, current, macro_ids, response_cap)
        except InvalidKCLabelError as exc:
            if attempt >= max_repair_attempts:
                raise
            invalid_labels = _invalid_kc_labels(current, allowed_by_field)
            if not invalid_labels:
                invalid_labels = [
                    {
                        "kc_index": exc.kc_index,
                        "field": exc.field_name,
                        "invalid_value": exc.invalid_value,
                    }
                ]
            log(
                "unit KC label repair retry",
                unit_id=str(unit.get("unit_id")),
                invalid_labels=json.dumps(invalid_labels, ensure_ascii=False),
                attempt=attempt + 1,
                max_attempts=max_repair_attempts,
            )
            try:
                current = _repair_unit_kc_labels(
                    unit=unit,
                    result=current,
                    invalid_labels=invalid_labels,
                    allowed_by_field=allowed_by_field,
                    client=client,
                    repair_template=repair_template,
                )
            except ValueError as repair_error:
                log(
                    "unit KC label repair rejected",
                    unit_id=str(unit.get("unit_id")),
                    attempt=attempt + 1,
                    max_attempts=max_repair_attempts,
                    error=f"{type(repair_error).__name__}: {repair_error}",
                )
                continue
    raise RuntimeError(f"Unit {unit.get('unit_id')} label repair did not produce a result.")


def _allowed_labels(macro_ids: set[str]) -> dict[str, set[str]]:
    return {
        "macro_id": set(macro_ids),
        "type": set(TEXT_KC_TYPES),
        "importance": set(VALID_IMPORTANCE),
        "claim_strength": set(VALID_CLAIM_STRENGTH),
    }


def _repair_unit_kc_labels(
    unit: dict,
    result: dict,
    invalid_labels: list[dict],
    allowed_by_field: dict[str, set[str]],
    client: OpenAICompatClient,
    repair_template: str,
) -> dict:
    repaired = client.chat_json(
        system_prompt=(
            "You repair invalid KC enum labels by mapping each one to the closest allowed value. "
            "Return JSON only."
        ),
        user_prompt=render_prompt(
            repair_template,
            allowed_labels_json=json.dumps(
                {field: sorted(values) for field, values in allowed_by_field.items()},
                ensure_ascii=False,
                indent=2,
            ),
            invalid_labels_json=json.dumps(invalid_labels, ensure_ascii=False, indent=2),
            unit_json=json.dumps(_unit_prompt_payload(unit), ensure_ascii=False, indent=2),
            kc_result_json=json.dumps(result, ensure_ascii=False, indent=2),
        ),
        temperature=0.0,
    )
    return _validate_label_only_repair(
        original=result,
        repaired=repaired,
        unit_id=str(unit.get("unit_id")),
        allowed_by_field=allowed_by_field,
    )


def _validate_label_only_repair(
    original: dict,
    repaired: dict,
    unit_id: str,
    allowed_by_field: dict[str, set[str]],
) -> dict:
    original_kcs = original.get("kcs", [])
    repaired_kcs = repaired.get("kcs", [])
    if not isinstance(original_kcs, list) or not isinstance(repaired_kcs, list):
        raise ValueError(f"KC label repair for unit {unit_id} must preserve a kcs list.")
    if len(original_kcs) != len(repaired_kcs):
        raise ValueError(
            f"KC label repair for unit {unit_id} changed KC count from "
            f"{len(original_kcs)} to {len(repaired_kcs)}."
        )
    if repaired.get("empty_reason", "") != original.get("empty_reason", ""):
        raise ValueError(f"KC label repair for unit {unit_id} changed empty_reason.")

    checked_kcs = []
    for idx, (original_item, repaired_item) in enumerate(zip(original_kcs, repaired_kcs), start=1):
        if not isinstance(original_item, dict) or not isinstance(repaired_item, dict):
            raise ValueError(f"KC label repair for unit {unit_id} item #{idx} must be an object.")
        if set(repaired_item) != set(original_item):
            raise ValueError(f"KC label repair for unit {unit_id} changed fields on KC #{idx}.")

        checked = dict(original_item)
        for key, original_value in original_item.items():
            if key not in LABEL_FIELDS:
                if repaired_item.get(key) != original_value:
                    raise ValueError(
                        f"KC label repair for unit {unit_id} changed non-label field {key!r} on KC #{idx}."
                    )
                continue

            original_label = str(original_value).strip()
            repaired_label = str(repaired_item.get(key, "")).strip()
            allowed_values = allowed_by_field[key]
            if not _label_is_valid(key, repaired_label, allowed_values):
                raise InvalidKCLabelError(unit_id, idx, key, repaired_label, allowed_values)
            if _label_is_valid(key, original_label, allowed_values) and repaired_label != original_label:
                raise ValueError(
                    f"KC label repair for unit {unit_id} changed already-valid {key!r} on KC #{idx}."
                )
            checked[key] = repaired_label
        checked_kcs.append(checked)

    return {
        "kcs": checked_kcs,
        "empty_reason": original.get("empty_reason", ""),
    }


def _invalid_kc_labels(
    result: dict,
    allowed_by_field: dict[str, set[str]],
) -> list[dict]:
    raw_kcs = result.get("kcs", [])
    if not isinstance(raw_kcs, list):
        return []
    invalid = []
    for idx, item in enumerate(raw_kcs, start=1):
        if not isinstance(item, dict):
            continue
        for field in LABEL_FIELDS:
            value = str(item.get(field, "")).strip()
            if not _label_is_valid(field, value, allowed_by_field[field]):
                invalid.append(
                    {
                        "kc_index": idx,
                        "field": field,
                        "invalid_value": value,
                    }
                )
    return invalid


def _label_is_valid(field: str, value: str, allowed_values: set[str]) -> bool:
    if field == "type":
        return valid_kc_type(value, TEXT_KC_TYPES) is not None
    return value in allowed_values

def _normalize_scope(scope: dict) -> dict:
    normalized = {
        "task": str(scope.get("task", "not_specified")).strip() or "not_specified",
        "dataset": str(scope.get("dataset", "not_specified")).strip() or "not_specified",
        "generality": str(scope.get("generality", "limited_to_claim_context")).strip()
        or "limited_to_claim_context",
    }
    for key, value in scope.items():
        if key in normalized:
            continue
        text = str(value).strip()
        if text:
            normalized[str(key).strip()] = text
    return normalized


def _text_in_source(text: str, source_text: str) -> bool:
    needle = _normalize_ws(text)
    haystack = _normalize_ws(source_text)
    return bool(needle and needle in haystack)


def _unit_paragraphs(source_text: str) -> list[dict]:
    paragraphs = [paragraph.strip() for paragraph in re.split(r"\n\s*\n", source_text) if paragraph.strip()]
    if not paragraphs and source_text.strip():
        paragraphs = [source_text.strip()]
    return [
        {
            "paragraph_id": f"P{idx}",
            "text": paragraph,
        }
        for idx, paragraph in enumerate(paragraphs, start=1)
    ]


def _normalize_evidence_paragraph_ids_to_contiguous_range(
    unit_id: str,
    kc_index: int,
    paragraph_ids: list[str],
    paragraph_by_id: dict[str, dict],
) -> list[str]:
    unknown = [paragraph_id for paragraph_id in paragraph_ids if paragraph_id not in paragraph_by_id]
    if unknown:
        raise ValueError(
            f"KC #{kc_index} in unit {unit_id} references unknown evidence_paragraph_ids: {unknown}"
        )
    positions = sorted({int(paragraph_id[1:]) for paragraph_id in paragraph_ids})
    expected = list(range(positions[0], positions[-1] + 1))
    return [f"P{position}" for position in expected]


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


def _effective_response_cap(unit_limit: int, hard_cap: int) -> int:
    positive_caps = [value for value in (unit_limit, hard_cap) if value > 0]
    return min(positive_caps) if positive_caps else 0


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
