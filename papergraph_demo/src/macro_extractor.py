from __future__ import annotations

import json
import os

from src.model_client import OpenAICompatClient
from src.progress import log, span
from src.prompt_loader import load_prompt, render_prompt


ALLOWED_MACRO_ROLES = {
    "background",
    "problem_motivation",
    "prior_work_limitation",
    "central_claim",
    "method_overview",
    "method_module",
    "mechanism",
    "training_or_algorithm",
    "dataset_or_resource",
    "experiment_setup",
    "main_result",
    "ablation_analysis",
    "case_or_error_analysis",
    "conclusion",
    "limitation",
}

ALLOWED_MACRO_IMPORTANCE = {"critical", "normal"}


def extract_macro_spine(
    paper_id: str,
    sections: list[dict],
    client: OpenAICompatClient,
) -> dict:
    if not client or not client.is_ready():
        raise RuntimeError("Online Macro Spine extraction requires a configured model client.")
    if not sections:
        raise ValueError("Cannot extract Macro Spine from empty sections.")

    target_count = _bounded_env_int("MACRO_TARGET_COUNT", 8, 1, 40)
    min_count = _bounded_env_int("MACRO_MIN_COUNT", 6, 1, target_count)
    max_count = _bounded_env_int("MACRO_MAX_COUNT", 12, target_count, 40)
    if min_count > max_count:
        raise ValueError(f"Invalid Macro count bounds: min={min_count}, max={max_count}")

    section_context = _compact_sections(sections)
    tpl = load_prompt("extract_macro_spine.txt")
    user_prompt = render_prompt(
        tpl,
        paper_id=paper_id,
        sections_json=json.dumps(section_context, ensure_ascii=False, indent=2),
        macro_target_count=str(target_count),
        macro_min_count=str(min_count),
        macro_max_count=str(max_count),
        allowed_roles=", ".join(sorted(ALLOWED_MACRO_ROLES)),
    )
    with span("extract macro spine", sections=len(sections), target=target_count):
        result = client.chat_json(
            system_prompt="You construct strict paper-understanding Macro Spine graphs. Return JSON only.",
            user_prompt=user_prompt,
            temperature=0.2,
        )
    spine = _normalize_macro_spine(
        paper_id=paper_id,
        result=result,
        sections=sections,
        min_count=min_count,
        max_count=max_count,
    )
    log(
        "macro spine extracted",
        macros=len(spine["macro_nodes"]),
        edges=len(spine["macro_edges"]),
    )
    return spine


def macro_context_for_prompt(macro_spine: dict) -> list[dict]:
    context = []
    for macro in macro_spine.get("macro_nodes", []):
        context.append(
            {
                "macro_id": macro.get("macro_id"),
                "title": macro.get("title"),
                "role": macro.get("role"),
                "summary": macro.get("summary"),
                "source_sections": macro.get("source_sections", []),
            }
        )
    return context


def _compact_sections(sections: list[dict]) -> list[dict]:
    compact = []
    for sec in sections:
        text = str(sec.get("text", "")).strip()
        compact.append(
            {
                "section_id": sec.get("section_id", ""),
                "title": sec.get("title", ""),
                "text_preview": text[:1800],
            }
        )
    return compact


def _normalize_macro_spine(
    paper_id: str,
    result: dict,
    sections: list[dict],
    min_count: int,
    max_count: int,
) -> dict:
    raw_nodes = result.get("macro_nodes", [])
    if not isinstance(raw_nodes, list):
        raise ValueError("Macro Spine response must contain macro_nodes list.")
    if not (min_count <= len(raw_nodes) <= max_count):
        raise ValueError(
            f"Macro Spine node count {len(raw_nodes)} is outside allowed range [{min_count}, {max_count}]."
        )

    valid_section_titles = {str(s.get("title", "")).strip() for s in sections}
    valid_section_ids = {str(s.get("section_id", "")).strip() for s in sections}
    macro_nodes = []
    seen_ids = set()
    for idx, item in enumerate(raw_nodes, start=1):
        macro_id = str(item.get("macro_id") or f"M{idx}").strip()
        expected_id = f"M{idx}"
        if macro_id != expected_id:
            raise ValueError(f"Macro IDs must be consecutive by order; expected {expected_id}, got {macro_id}.")
        if macro_id in seen_ids:
            raise ValueError(f"Duplicate macro_id in Macro Spine: {macro_id}")
        seen_ids.add(macro_id)

        role = str(item.get("role", "")).strip()
        if role not in ALLOWED_MACRO_ROLES:
            raise ValueError(f"Invalid macro role for {macro_id}: {role}")
        title = str(item.get("title", "")).strip()
        summary = str(item.get("summary", "")).strip()
        expected_reader_question = str(item.get("expected_reader_question", "")).strip()
        if not title or not summary or not expected_reader_question:
            raise ValueError(f"Macro {macro_id} must include title, summary, and expected_reader_question.")

        source_sections = _normalize_source_sections(
            item.get("source_sections", []),
            valid_section_titles,
            valid_section_ids,
            macro_id,
        )
        importance = str(item.get("importance", "normal")).strip()
        if importance not in ALLOWED_MACRO_IMPORTANCE:
            raise ValueError(f"Invalid macro importance for {macro_id}: {importance}")

        macro_nodes.append(
            {
                "macro_id": macro_id,
                "order": idx,
                "title": title,
                "role": role,
                "summary": summary,
                "source_sections": source_sections,
                "expected_reader_question": expected_reader_question,
                "prerequisite_macro_ids": _string_list(item.get("prerequisite_macro_ids", [])),
                "next_macro_ids": _string_list(item.get("next_macro_ids", [])),
                "importance": importance,
            }
        )

    valid_macro_ids = {m["macro_id"] for m in macro_nodes}
    for macro in macro_nodes:
        for field in ("prerequisite_macro_ids", "next_macro_ids"):
            bad = [mid for mid in macro[field] if mid not in valid_macro_ids]
            if bad:
                raise ValueError(f"Macro {macro['macro_id']} has invalid {field}: {bad}")

    macro_edges = []
    for idx, edge in enumerate(result.get("macro_edges", []), start=1):
        source = str(edge.get("source", "")).strip()
        target = str(edge.get("target", "")).strip()
        if source not in valid_macro_ids or target not in valid_macro_ids or source == target:
            raise ValueError(f"Invalid macro edge: {source} -> {target}")
        relation = str(edge.get("relation", "")).strip() or "leads_to"
        description = str(edge.get("description", "")).strip()
        macro_edges.append(
            {
                "edge_id": f"ME{idx}",
                "source": source,
                "target": target,
                "relation": relation,
                "description": description,
            }
        )
    if not macro_edges:
        raise ValueError("Macro Spine must contain at least one macro edge.")

    return {"paper_id": paper_id, "macro_nodes": macro_nodes, "macro_edges": macro_edges}


def _normalize_source_sections(
    values: object,
    valid_titles: set[str],
    valid_ids: set[str],
    macro_id: str,
) -> list[str]:
    source_sections = _string_list(values)
    if not source_sections:
        raise ValueError(f"Macro {macro_id} must include source_sections.")
    invalid = [
        value
        for value in source_sections
        if value not in valid_titles and value not in valid_ids
    ]
    if invalid:
        raise ValueError(f"Macro {macro_id} references unknown source_sections: {invalid}")
    return source_sections


def _string_list(values: object) -> list[str]:
    if not isinstance(values, list):
        return []
    return [str(v).strip() for v in values if str(v).strip()]


def _bounded_env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    raw = os.getenv(name)
    try:
        value = int(raw) if raw is not None and raw.strip() else default
    except ValueError:
        value = default
    return max(minimum, min(maximum, value))

