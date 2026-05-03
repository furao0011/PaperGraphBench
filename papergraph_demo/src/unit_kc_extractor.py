from __future__ import annotations

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.macro_extractor import macro_context_for_prompt
from src.model_client import OpenAICompatClient
from src.progress import log, span
from src.prompt_loader import load_prompt, render_prompt


VALID_KC_TYPES = {
    "problem",
    "method",
    "mechanism",
    "dataset",
    "experiment",
    "result",
    "conclusion",
    "limitation",
    "background",
    "central_claim",
    "algorithm",
    "analysis",
}

VALID_IMPORTANCE = {"critical", "normal"}

VALID_CLAIM_STRENGTH = {
    "explicit",
    "experimentally_supported",
    "partially_supported",
    "author_hypothesis",
    "plausible_inference",
    "limitation_or_missing_evidence",
}


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
    macro_context_json = json.dumps(macro_context_for_prompt(macro_spine), ensure_ascii=False, indent=2)
    unit_limit = _env_nonnegative_int("KC_PER_UNIT_LIMIT", 0)
    hard_cap = _env_nonnegative_int("KC_PER_UNIT_HARD_CAP", 20)
    response_cap = _effective_response_cap(unit_limit, hard_cap)
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
        unit_candidates, empty_reason = _normalize_unit_result(
            unit=unit,
            result=result,
            macro_ids=macro_ids,
            response_cap=response_cap,
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
        "source_text": unit.get("source_text"),
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
    out = []
    for idx, item in enumerate(raw_kcs, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"KC #{idx} in unit {unit.get('unit_id')} must be an object.")
        claim = str(item.get("claim", "")).strip()
        evidence = str(item.get("evidence", "")).strip()
        if not claim or not evidence:
            raise ValueError(f"KC #{idx} in unit {unit.get('unit_id')} must include claim and evidence.")
        if not _text_in_source(evidence, source_text):
            raise ValueError(f"KC #{idx} in unit {unit.get('unit_id')} evidence is not found in source_text.")
        macro_id = str(item.get("macro_id", "")).strip()
        if macro_id not in macro_ids:
            raise ValueError(f"KC #{idx} in unit {unit.get('unit_id')} references invalid macro_id={macro_id!r}.")
        kc_type = str(item.get("type", "")).strip()
        if kc_type not in VALID_KC_TYPES:
            raise ValueError(f"KC #{idx} in unit {unit.get('unit_id')} has invalid type={kc_type!r}.")
        importance = str(item.get("importance", "")).strip()
        if importance not in VALID_IMPORTANCE:
            raise ValueError(f"KC #{idx} in unit {unit.get('unit_id')} has invalid importance={importance!r}.")
        claim_strength = str(item.get("claim_strength", "")).strip()
        if claim_strength not in VALID_CLAIM_STRENGTH:
            raise ValueError(
                f"KC #{idx} in unit {unit.get('unit_id')} has invalid claim_strength={claim_strength!r}."
            )
        scope = item.get("scope")
        if not isinstance(scope, dict) or not scope:
            raise ValueError(f"KC #{idx} in unit {unit.get('unit_id')} must include non-empty scope object.")
        out.append(
            {
                "claim": claim,
                "evidence": evidence,
                "macro_id": macro_id,
                "type": kc_type,
                "importance": importance,
                "claim_strength": claim_strength,
                "scope": _normalize_scope(scope),
                "related_terms": _string_list(item.get("related_terms", [])),
            }
        )
    return out, empty_reason


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
