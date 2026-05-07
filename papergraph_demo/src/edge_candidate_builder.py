from __future__ import annotations

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.edge_verifier import ALLOWED_EDGE_RELATIONS
from src.model_client import OpenAICompatClient
from src.progress import log, span
from src.prompt_loader import load_prompt, render_prompt


def build_unit_edge_candidates(
    paper_id: str,
    kc_bank: dict,
    extraction_units: dict,
    client: OpenAICompatClient,
) -> dict:
    if not client or not client.is_ready():
        raise RuntimeError("Unit edge candidate construction requires a configured online model client.")

    unit_contexts = _unit_contexts(kc_bank, extraction_units)
    if not unit_contexts:
        return {
            "paper_id": paper_id,
            "source_layer": "unit",
            "edge_candidates": [],
            "skipped_units": _skipped_units(kc_bank, extraction_units),
        }

    tpl = load_prompt("build_unit_edges.txt")
    max_workers = min(_env_positive_int("EDGE_UNIT_WORKERS", 3), len(unit_contexts))
    candidates_by_unit: dict[str, list[dict]] = {}
    errors: list[str] = []

    def run_one(context: dict) -> tuple[str, list[dict]]:
        unit_id = context["unit_id"]
        with span("build unit edge candidates", unit_id=unit_id, kcs=len(context["kc_nodes"])):
            result = client.chat_json(
                system_prompt="You construct local reasoning edge candidates for paper KCs. Return JSON only.",
                user_prompt=render_prompt(
                    tpl,
                    unit_context_json=json.dumps(context, ensure_ascii=False, indent=2),
                ),
                temperature=0.1,
            )
        return unit_id, _normalize_unit_edges(context, result)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(run_one, context): context for context in unit_contexts}
        for fut in as_completed(futures):
            context = futures[fut]
            unit_id = context["unit_id"]
            try:
                out_unit_id, edges = fut.result()
                candidates_by_unit[out_unit_id] = edges
                log("unit edge candidates built", unit_id=out_unit_id, candidates=len(edges))
            except Exception as exc:
                errors.append(f"{unit_id}: {type(exc).__name__}: {exc}")
                log("unit edge candidate error", unit_id=unit_id, error=f"{type(exc).__name__}: {exc}")

    if errors:
        raise RuntimeError("Unit edge candidate construction failed: " + "; ".join(errors[:5]))

    candidates = []
    seen = set()
    for context in unit_contexts:
        for edge in candidates_by_unit.get(context["unit_id"], []):
            key = (edge["unit_id"], edge["source"], edge["target"], edge["relation"])
            if key in seen:
                continue
            seen.add(key)
            candidates.append(
                {
                    "candidate_edge_id": f"UEC{len(candidates) + 1}",
                    **edge,
                }
            )

    return {
        "paper_id": paper_id,
        "source_layer": "unit",
        "edge_candidates": candidates,
        "skipped_units": _skipped_units(kc_bank, extraction_units),
    }


def build_macro_edge_candidates(
    paper_id: str,
    kc_bank: dict,
    macro_spine: dict,
    extraction_units: dict,
    client: OpenAICompatClient,
) -> dict:
    if not client or not client.is_ready():
        raise RuntimeError("Macro edge candidate construction requires a configured online model client.")

    macro_contexts = _macro_contexts(kc_bank, macro_spine, extraction_units)
    if not macro_contexts:
        return {
            "paper_id": paper_id,
            "source_layer": "macro",
            "edge_candidates": [],
            "skipped_macros": _skipped_macros(kc_bank, macro_spine),
        }

    tpl = load_prompt("build_macro_edges.txt")
    max_workers = min(_env_positive_int("EDGE_MACRO_WORKERS", 3), len(macro_contexts))
    candidates_by_batch: dict[str, list[dict]] = {}
    errors: list[str] = []

    def run_one(context: dict) -> tuple[str, list[dict]]:
        batch_id = context["batch_id"]
        with span(
            "build macro edge candidates",
            macro_id=context["macro_id"],
            batch_id=batch_id,
            kcs=len(context["kc_nodes"]),
        ):
            result = client.chat_json(
                system_prompt="You construct Macro-internal reasoning edge candidates for paper KCs. Return JSON only.",
                user_prompt=render_prompt(
                    tpl,
                    macro_context_json=json.dumps(context, ensure_ascii=False, indent=2),
                ),
                temperature=0.1,
            )
        return batch_id, _normalize_macro_edges(context, result)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(run_one, context): context for context in macro_contexts}
        for fut in as_completed(futures):
            context = futures[fut]
            batch_id = context["batch_id"]
            try:
                out_batch_id, edges = fut.result()
                candidates_by_batch[out_batch_id] = edges
                log("macro edge candidates built", batch_id=out_batch_id, candidates=len(edges))
            except Exception as exc:
                errors.append(f"{batch_id}: {type(exc).__name__}: {exc}")
                log("macro edge candidate error", batch_id=batch_id, error=f"{type(exc).__name__}: {exc}")

    if errors:
        raise RuntimeError("Macro edge candidate construction failed: " + "; ".join(errors[:5]))

    candidates = []
    seen = set()
    for context in macro_contexts:
        for edge in candidates_by_batch.get(context["batch_id"], []):
            key = (edge["macro_id"], edge["source"], edge["target"], edge["relation"])
            if key in seen:
                continue
            seen.add(key)
            candidates.append(
                {
                    "candidate_edge_id": f"MEC{len(candidates) + 1}",
                    **edge,
                }
            )

    return {
        "paper_id": paper_id,
        "source_layer": "macro",
        "edge_candidates": candidates,
        "skipped_macros": _skipped_macros(kc_bank, macro_spine),
    }


def build_adjacent_macro_edge_candidates(
    paper_id: str,
    kc_bank: dict,
    macro_spine: dict,
    extraction_units: dict,
    client: OpenAICompatClient,
) -> dict:
    if not client or not client.is_ready():
        raise RuntimeError("Adjacent-Macro edge candidate construction requires a configured online model client.")

    contexts = _adjacent_macro_contexts(kc_bank, macro_spine, extraction_units)
    if not contexts:
        return {
            "paper_id": paper_id,
            "source_layer": "adjacent_macro",
            "edge_candidates": [],
            "skipped_macro_edges": _skipped_macro_edges(kc_bank, macro_spine),
        }

    tpl = load_prompt("build_adjacent_macro_edges.txt")
    max_workers = min(_env_positive_int("EDGE_ADJACENT_MACRO_WORKERS", 3), len(contexts))
    candidates_by_pair: dict[str, list[dict]] = {}
    errors: list[str] = []

    def run_one(context: dict) -> tuple[str, list[dict]]:
        pair_id = context["macro_pair_id"]
        with span("build adjacent macro edge candidates", macro_pair_id=pair_id):
            result = client.chat_json(
                system_prompt="You construct adjacent-Macro reasoning edge candidates for paper KCs. Return JSON only.",
                user_prompt=render_prompt(
                    tpl,
                    adjacent_macro_context_json=json.dumps(context, ensure_ascii=False, indent=2),
                ),
                temperature=0.1,
            )
        return pair_id, _normalize_adjacent_macro_edges(context, result)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(run_one, context): context for context in contexts}
        for fut in as_completed(futures):
            context = futures[fut]
            pair_id = context["macro_pair_id"]
            try:
                out_pair_id, edges = fut.result()
                candidates_by_pair[out_pair_id] = edges
                log("adjacent macro edge candidates built", macro_pair_id=out_pair_id, candidates=len(edges))
            except Exception as exc:
                errors.append(f"{pair_id}: {type(exc).__name__}: {exc}")
                log("adjacent macro edge candidate error", macro_pair_id=pair_id, error=f"{type(exc).__name__}: {exc}")

    if errors:
        raise RuntimeError("Adjacent-Macro edge candidate construction failed: " + "; ".join(errors[:5]))

    candidates = []
    seen = set()
    for context in contexts:
        for edge in candidates_by_pair.get(context["macro_pair_id"], []):
            key = (edge["source_macro_id"], edge["target_macro_id"], edge["source"], edge["target"], edge["relation"])
            if key in seen:
                continue
            seen.add(key)
            candidates.append({"candidate_edge_id": f"AMEC{len(candidates) + 1}", **edge})

    return {
        "paper_id": paper_id,
        "source_layer": "adjacent_macro",
        "edge_candidates": candidates,
        "skipped_macro_edges": _skipped_macro_edges(kc_bank, macro_spine),
    }


def build_thread_candidate_edges(
    paper_id: str,
    kc_bank: dict,
    macro_spine: dict,
    extraction_units: dict,
    verified_edges: list[dict],
    client: OpenAICompatClient,
) -> dict:
    if not client or not client.is_ready():
        raise RuntimeError("Thread candidate edge construction requires a configured online model client.")

    context = _thread_context(kc_bank, macro_spine, extraction_units, verified_edges)
    if len(context["kc_nodes"]) < 2:
        return {
            "paper_id": paper_id,
            "source_layer": "thread",
            "edge_candidates": [],
            "skipped_reason": "fewer_than_two_thread_candidate_kcs",
        }

    tpl = load_prompt("build_thread_candidate_edges.txt")
    with span("build thread candidate edges", kcs=len(context["kc_nodes"]), verified_edges=len(verified_edges)):
        result = client.chat_json(
            system_prompt="You construct cross-Macro Thread candidate edges for paper KCs. Return JSON only.",
            user_prompt=render_prompt(
                tpl,
                thread_context_json=json.dumps(context, ensure_ascii=False, indent=2),
            ),
            temperature=0.1,
        )
    candidates = []
    seen = set()
    for edge in _normalize_thread_edges(context, result):
        key = (edge["source"], edge["target"], edge["relation"], edge.get("thread_pattern", ""))
        if key in seen:
            continue
        seen.add(key)
        candidates.append({"candidate_edge_id": f"TEC{len(candidates) + 1}", **edge})
    return {
        "paper_id": paper_id,
        "source_layer": "thread",
        "edge_candidates": candidates,
    }


def _unit_contexts(kc_bank: dict, extraction_units: dict) -> list[dict]:
    units_by_id = {
        unit["unit_id"]: unit
        for unit in extraction_units.get("units", [])
        if unit.get("unit_id")
    }
    kcs_by_unit: dict[str, list[dict]] = {}
    for kc in kc_bank.get("kc_nodes", []):
        unit_id = str(kc.get("unit_id", "")).strip()
        if unit_id:
            kcs_by_unit.setdefault(unit_id, []).append(kc)

    contexts = []
    for unit_id, kcs in sorted(kcs_by_unit.items(), key=lambda item: item[0]):
        unit = units_by_id.get(unit_id)
        if not unit or len(kcs) < 2:
            continue
        contexts.append(
            {
                "unit_id": unit_id,
                "unit_title": unit.get("unit_title", ""),
                "unit_summary": unit.get("unit_summary", ""),
                "source_text": unit.get("source_text", ""),
                "source_paragraphs": _paragraphs_from_source(unit.get("source_text", "")),
                "kc_nodes": [
                    {
                        "kc_id": kc.get("kc_id"),
                        "macro_id": kc.get("macro_id"),
                        "type": kc.get("type"),
                        "claim_strength": kc.get("claim_strength"),
                        "scope": kc.get("scope"),
                        "full_claim": kc.get("full_claim"),
                        "evidence_text": kc.get("evidence_text"),
                        "related_terms": kc.get("related_terms", []),
                    }
                    for kc in sorted(kcs, key=lambda item: _kc_sort_key(str(item.get("kc_id", ""))))
                ],
            }
        )
    return contexts


def _macro_contexts(kc_bank: dict, macro_spine: dict, extraction_units: dict) -> list[dict]:
    units_by_id = {
        unit["unit_id"]: unit
        for unit in extraction_units.get("units", [])
        if unit.get("unit_id")
    }
    macro_nodes = {
        macro["macro_id"]: macro
        for macro in macro_spine.get("macro_nodes", [])
        if macro.get("macro_id")
    }
    kcs_by_macro: dict[str, list[dict]] = {}
    for kc in kc_bank.get("kc_nodes", []):
        macro_id = str(kc.get("macro_id", "")).strip()
        unit_id = str(kc.get("unit_id", "")).strip()
        if macro_id and unit_id:
            kcs_by_macro.setdefault(macro_id, []).append(kc)

    batch_limit = _env_positive_int("EDGE_MACRO_BATCH_KCS", 30)
    contexts = []
    for macro_id, kcs in sorted(kcs_by_macro.items()):
        macro = macro_nodes.get(macro_id)
        if not macro:
            continue
        unit_ids = {str(kc.get("unit_id", "")).strip() for kc in kcs if kc.get("unit_id")}
        if len(kcs) < 2 or len(unit_ids) < 2:
            continue
        ordered_kcs = sorted(kcs, key=lambda kc: (_unit_order_key(units_by_id.get(kc.get("unit_id"), {})), _kc_sort_key(kc["kc_id"])))
        for batch_index, batch in enumerate(_sliding_batches(ordered_kcs, batch_limit), start=1):
            batch_unit_ids = {str(kc.get("unit_id", "")).strip() for kc in batch if kc.get("unit_id")}
            if len(batch) < 2 or len(batch_unit_ids) < 2:
                continue
            contexts.append(
                {
                    "batch_id": f"{macro_id}_MB{batch_index}",
                    "macro_id": macro_id,
                    "macro_title": macro.get("title", ""),
                    "macro_role": macro.get("role", ""),
                    "macro_summary": macro.get("summary", ""),
                    "unit_summaries": [
                        {
                            "unit_id": unit_id,
                            "unit_title": units_by_id.get(unit_id, {}).get("unit_title", ""),
                            "unit_summary": units_by_id.get(unit_id, {}).get("unit_summary", ""),
                            "source_text": units_by_id.get(unit_id, {}).get("source_text", ""),
                            "source_paragraphs": _paragraphs_from_source(units_by_id.get(unit_id, {}).get("source_text", "")),
                        }
                        for unit_id in sorted(batch_unit_ids, key=lambda uid: _unit_order_key(units_by_id.get(uid, {})))
                    ],
                    "kc_nodes": [
                        {
                            "kc_id": kc.get("kc_id"),
                            "unit_id": kc.get("unit_id"),
                            "macro_id": kc.get("macro_id"),
                            "type": kc.get("type"),
                            "claim_strength": kc.get("claim_strength"),
                            "scope": kc.get("scope"),
                            "full_claim": kc.get("full_claim"),
                            "evidence_text": kc.get("evidence_text"),
                            "related_terms": kc.get("related_terms", []),
                        }
                        for kc in batch
                    ],
                }
            )
    return contexts


def _adjacent_macro_contexts(kc_bank: dict, macro_spine: dict, extraction_units: dict) -> list[dict]:
    units_by_id = _units_by_id(extraction_units)
    macros_by_id = {
        macro["macro_id"]: macro
        for macro in macro_spine.get("macro_nodes", [])
        if macro.get("macro_id")
    }
    kcs_by_macro = _kcs_by_macro(kc_bank)
    top_k = _env_positive_int("EDGE_ADJACENT_MACRO_TOP_KCS", 12)
    contexts = []
    for idx, macro_edge in enumerate(macro_spine.get("macro_edges", []), start=1):
        source_macro_id = str(macro_edge.get("source", "")).strip()
        target_macro_id = str(macro_edge.get("target", "")).strip()
        source_macro = macros_by_id.get(source_macro_id)
        target_macro = macros_by_id.get(target_macro_id)
        if not source_macro or not target_macro or source_macro_id == target_macro_id:
            continue
        source_kcs = _representative_kcs(kcs_by_macro.get(source_macro_id, []), units_by_id, top_k)
        target_kcs = _representative_kcs(kcs_by_macro.get(target_macro_id, []), units_by_id, top_k)
        if not source_kcs or not target_kcs:
            continue
        selected_kcs = source_kcs + target_kcs
        unit_ids = {kc.get("unit_id") for kc in selected_kcs if kc.get("unit_id")}
        contexts.append(
            {
                "macro_pair_id": f"AMP{idx}_{source_macro_id}_{target_macro_id}",
                "macro_edge_id": macro_edge.get("edge_id") or f"ME{idx}",
                "macro_edge_relation": macro_edge.get("relation", ""),
                "macro_edge_description": macro_edge.get("description", ""),
                "source_macro": _macro_packet(source_macro),
                "target_macro": _macro_packet(target_macro),
                "source_kc_nodes": [_kc_context_packet(kc) for kc in source_kcs],
                "target_kc_nodes": [_kc_context_packet(kc) for kc in target_kcs],
                "unit_texts": _unit_text_packets(unit_ids, units_by_id),
            }
        )
    return contexts


def _thread_context(
    kc_bank: dict,
    macro_spine: dict,
    extraction_units: dict,
    verified_edges: list[dict],
) -> dict:
    units_by_id = _units_by_id(extraction_units)
    kc_limit = _env_positive_int("EDGE_THREAD_CANDIDATE_KCS", 40)
    kc_nodes = _representative_kcs(kc_bank.get("kc_nodes", []), units_by_id, kc_limit)
    unit_ids = {kc.get("unit_id") for kc in kc_nodes if kc.get("unit_id")}
    return {
        "macro_spine": {
            "macro_nodes": [
                _macro_packet(macro)
                for macro in macro_spine.get("macro_nodes", [])
            ],
            "macro_edges": macro_spine.get("macro_edges", []),
        },
        "kc_nodes": [_kc_context_packet(kc) for kc in kc_nodes],
        "verified_edge_hints": [
            {
                "edge_id": edge.get("edge_id"),
                "source_layer": edge.get("source_layer"),
                "scope": edge.get("scope"),
                "source": edge.get("source"),
                "target": edge.get("target"),
                "relation": edge.get("relation"),
                "description": edge.get("description"),
            }
            for edge in verified_edges[: _env_positive_int("EDGE_THREAD_VERIFIED_HINTS", 80)]
        ],
        "unit_texts": _unit_text_packets(unit_ids, units_by_id),
        "thread_patterns": [
            "problem_to_method",
            "method_to_mechanism",
            "method_to_result",
            "module_to_ablation",
            "claim_to_limitation",
            "baseline_to_comparison_result",
        ],
    }


def _normalize_unit_edges(context: dict, result: dict) -> list[dict]:
    raw_edges = result.get("edges", [])
    if not isinstance(raw_edges, list):
        raise ValueError(f"Unit {context['unit_id']} edge response must contain edges list.")
    valid_kc_ids = {kc["kc_id"] for kc in context["kc_nodes"]}
    paragraph_by_id = _paragraph_by_id(context.get("source_paragraphs", []))
    edges = []
    for idx, item in enumerate(raw_edges, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Unit {context['unit_id']} edge #{idx} must be an object.")
        source = str(item.get("source", "")).strip()
        target = str(item.get("target", "")).strip()
        relation = str(item.get("relation", "")).strip()
        evidence_paragraph_ids = _string_list(item.get("evidence_paragraph_ids", []))
        if source not in valid_kc_ids or target not in valid_kc_ids or source == target:
            raise ValueError(f"Unit {context['unit_id']} edge #{idx} references invalid source/target.")
        if relation not in ALLOWED_EDGE_RELATIONS:
            raise ValueError(f"Unit {context['unit_id']} edge #{idx} has invalid relation={relation!r}.")
        evidence, normalized_evidence_ids = _source_text_from_paragraph_ids(
            f"Unit {context['unit_id']} edge #{idx}",
            evidence_paragraph_ids,
            paragraph_by_id,
        )
        edges.append(
            {
                "source_layer": "unit",
                "scope": "unit",
                "unit_id": context["unit_id"],
                "source": source,
                "target": target,
                "relation": relation,
                "evidence": evidence,
                "evidence_paragraph_ids": normalized_evidence_ids,
                "evidence_paragraph_ids_original": evidence_paragraph_ids,
                "evidence_paragraph_id_normalization": (
                    "as_provided" if normalized_evidence_ids == evidence_paragraph_ids else "expanded_to_contiguous_range"
                ),
                "reason": str(item.get("reason", "")).strip(),
                "confidence": _confidence(item.get("confidence", 0.0)),
            }
        )
    return edges


def _normalize_macro_edges(context: dict, result: dict) -> list[dict]:
    raw_edges = result.get("edges", [])
    if not isinstance(raw_edges, list):
        raise ValueError(f"Macro batch {context['batch_id']} edge response must contain edges list.")
    kcs_by_id = {kc["kc_id"]: kc for kc in context["kc_nodes"]}
    unit_texts = {
        item["unit_id"]: str(item.get("source_text", "")).strip()
        for item in context.get("unit_summaries", [])
        if item.get("unit_id")
    }
    unit_paragraphs = {
        item["unit_id"]: _paragraph_by_id(item.get("source_paragraphs", []))
        for item in context.get("unit_summaries", [])
        if item.get("unit_id")
    }
    edges = []
    for idx, item in enumerate(raw_edges, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Macro batch {context['batch_id']} edge #{idx} must be an object.")
        source = str(item.get("source", "")).strip()
        target = str(item.get("target", "")).strip()
        relation = str(item.get("relation", "")).strip()
        evidence_unit_id = str(item.get("evidence_unit_id", "")).strip()
        evidence_paragraph_ids = _string_list(item.get("evidence_paragraph_ids", []))
        if source not in kcs_by_id or target not in kcs_by_id or source == target:
            raise ValueError(f"Macro batch {context['batch_id']} edge #{idx} references invalid source/target.")
        source_unit_id = str(kcs_by_id[source].get("unit_id", "")).strip()
        target_unit_id = str(kcs_by_id[target].get("unit_id", "")).strip()
        if not source_unit_id or not target_unit_id or source_unit_id == target_unit_id:
            raise ValueError(f"Macro batch {context['batch_id']} edge #{idx} must connect different Units.")
        if relation not in ALLOWED_EDGE_RELATIONS:
            raise ValueError(f"Macro batch {context['batch_id']} edge #{idx} has invalid relation={relation!r}.")
        if evidence_unit_id not in {source_unit_id, target_unit_id}:
            raise ValueError(
                f"Macro batch {context['batch_id']} edge #{idx} evidence_unit_id must be source or target Unit."
            )
        evidence, normalized_evidence_ids = _source_text_from_paragraph_ids(
            f"Macro batch {context['batch_id']} edge #{idx}",
            evidence_paragraph_ids,
            unit_paragraphs.get(evidence_unit_id, {}),
        )
        if not _text_in_source(evidence, unit_texts.get(evidence_unit_id, "")):
            raise ValueError(f"Macro batch {context['batch_id']} edge #{idx} evidence is not found in evidence Unit text.")
        edges.append(
            {
                "source_layer": "macro",
                "scope": "macro",
                "macro_id": context["macro_id"],
                "macro_title": context.get("macro_title", ""),
                "macro_role": context.get("macro_role", ""),
                "macro_summary": context.get("macro_summary", ""),
                "batch_id": context["batch_id"],
                "unit_id": None,
                "source_unit_id": source_unit_id,
                "target_unit_id": target_unit_id,
                "evidence_unit_id": evidence_unit_id,
                "source": source,
                "target": target,
                "relation": relation,
                "evidence": evidence,
                "evidence_paragraph_ids": normalized_evidence_ids,
                "evidence_paragraph_ids_original": evidence_paragraph_ids,
                "evidence_paragraph_id_normalization": (
                    "as_provided" if normalized_evidence_ids == evidence_paragraph_ids else "expanded_to_contiguous_range"
                ),
                "reason": str(item.get("reason", "")).strip(),
                "confidence": _confidence(item.get("confidence", 0.0)),
            }
        )
    return edges


def _normalize_adjacent_macro_edges(context: dict, result: dict) -> list[dict]:
    raw_edges = result.get("edges", [])
    if not isinstance(raw_edges, list):
        raise ValueError(f"Adjacent Macro {context['macro_pair_id']} response must contain edges list.")
    source_kcs = {kc["kc_id"]: kc for kc in context["source_kc_nodes"]}
    target_kcs = {kc["kc_id"]: kc for kc in context["target_kc_nodes"]}
    unit_texts = {item["unit_id"]: item.get("source_text", "") for item in context.get("unit_texts", [])}
    unit_paragraphs = {
        item["unit_id"]: _paragraph_by_id(item.get("source_paragraphs", []))
        for item in context.get("unit_texts", [])
    }
    edges = []
    for idx, item in enumerate(raw_edges, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Adjacent Macro {context['macro_pair_id']} edge #{idx} must be an object.")
        source = str(item.get("source", "")).strip()
        target = str(item.get("target", "")).strip()
        relation = str(item.get("relation", "")).strip()
        evidence_unit_id = str(item.get("evidence_unit_id", "")).strip()
        evidence_paragraph_ids = _string_list(item.get("evidence_paragraph_ids", []))
        if source not in source_kcs or target not in target_kcs:
            raise ValueError(
                f"Adjacent Macro {context['macro_pair_id']} edge #{idx} must go from source Macro KC to target Macro KC."
            )
        if relation not in ALLOWED_EDGE_RELATIONS:
            raise ValueError(f"Adjacent Macro {context['macro_pair_id']} edge #{idx} has invalid relation={relation!r}.")
        source_unit_id = str(source_kcs[source].get("unit_id", "")).strip()
        target_unit_id = str(target_kcs[target].get("unit_id", "")).strip()
        if evidence_unit_id not in {source_unit_id, target_unit_id}:
            raise ValueError(
                f"Adjacent Macro {context['macro_pair_id']} edge #{idx} evidence_unit_id must be source or target Unit."
            )
        evidence, normalized_evidence_ids = _source_text_from_paragraph_ids(
            f"Adjacent Macro {context['macro_pair_id']} edge #{idx}",
            evidence_paragraph_ids,
            unit_paragraphs.get(evidence_unit_id, {}),
        )
        if not _text_in_source(evidence, unit_texts.get(evidence_unit_id, "")):
            raise ValueError(f"Adjacent Macro {context['macro_pair_id']} edge #{idx} evidence is not found in evidence Unit text.")
        edges.append(
            {
                "source_layer": "adjacent_macro",
                "scope": "adjacent_macro",
                "macro_edge_id": context.get("macro_edge_id"),
                "source_macro_id": context["source_macro"]["macro_id"],
                "target_macro_id": context["target_macro"]["macro_id"],
                "source_unit_id": source_unit_id,
                "target_unit_id": target_unit_id,
                "evidence_unit_id": evidence_unit_id,
                "source": source,
                "target": target,
                "relation": relation,
                "evidence": evidence,
                "evidence_paragraph_ids": normalized_evidence_ids,
                "evidence_paragraph_ids_original": evidence_paragraph_ids,
                "evidence_paragraph_id_normalization": (
                    "as_provided" if normalized_evidence_ids == evidence_paragraph_ids else "expanded_to_contiguous_range"
                ),
                "reason": str(item.get("reason", "")).strip(),
                "confidence": _confidence(item.get("confidence", 0.0)),
            }
        )
    return edges


def _normalize_thread_edges(context: dict, result: dict) -> list[dict]:
    raw_edges = result.get("edges", [])
    if not isinstance(raw_edges, list):
        raise ValueError("Thread candidate edge response must contain edges list.")
    kcs_by_id = {kc["kc_id"]: kc for kc in context["kc_nodes"]}
    unit_texts = {item["unit_id"]: item.get("source_text", "") for item in context.get("unit_texts", [])}
    unit_paragraphs = {
        item["unit_id"]: _paragraph_by_id(item.get("source_paragraphs", []))
        for item in context.get("unit_texts", [])
    }
    edges = []
    for idx, item in enumerate(raw_edges, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Thread candidate edge #{idx} must be an object.")
        source = str(item.get("source", "")).strip()
        target = str(item.get("target", "")).strip()
        relation = str(item.get("relation", "")).strip()
        evidence_unit_id = str(item.get("evidence_unit_id", "")).strip()
        evidence_paragraph_ids = _string_list(item.get("evidence_paragraph_ids", []))
        if source not in kcs_by_id or target not in kcs_by_id or source == target:
            raise ValueError(f"Thread candidate edge #{idx} references invalid source/target.")
        source_macro_id = str(kcs_by_id[source].get("macro_id", "")).strip()
        target_macro_id = str(kcs_by_id[target].get("macro_id", "")).strip()
        if not source_macro_id or not target_macro_id or source_macro_id == target_macro_id:
            raise ValueError(f"Thread candidate edge #{idx} must connect different Macros.")
        if relation not in ALLOWED_EDGE_RELATIONS:
            raise ValueError(f"Thread candidate edge #{idx} has invalid relation={relation!r}.")
        source_unit_id = str(kcs_by_id[source].get("unit_id", "")).strip()
        target_unit_id = str(kcs_by_id[target].get("unit_id", "")).strip()
        if evidence_unit_id not in {source_unit_id, target_unit_id}:
            raise ValueError(
                f"Thread candidate edge #{idx} evidence_unit_id must be source or target Unit."
            )
        evidence, normalized_evidence_ids = _source_text_from_paragraph_ids(
            f"Thread candidate edge #{idx}",
            evidence_paragraph_ids,
            unit_paragraphs.get(evidence_unit_id, {}),
        )
        if not _text_in_source(evidence, unit_texts.get(evidence_unit_id, "")):
            raise ValueError(f"Thread candidate edge #{idx} evidence is not found in evidence Unit text.")
        edges.append(
            {
                "source_layer": "thread",
                "scope": "thread",
                "thread_pattern": str(item.get("thread_pattern", "")).strip(),
                "source_macro_id": source_macro_id,
                "target_macro_id": target_macro_id,
                "source_unit_id": source_unit_id,
                "target_unit_id": target_unit_id,
                "evidence_unit_id": evidence_unit_id,
                "source": source,
                "target": target,
                "relation": relation,
                "evidence": evidence,
                "evidence_paragraph_ids": normalized_evidence_ids,
                "evidence_paragraph_ids_original": evidence_paragraph_ids,
                "evidence_paragraph_id_normalization": (
                    "as_provided" if normalized_evidence_ids == evidence_paragraph_ids else "expanded_to_contiguous_range"
                ),
                "reason": str(item.get("reason", "")).strip(),
                "confidence": _confidence(item.get("confidence", 0.0)),
            }
        )
    return edges


def _skipped_units(kc_bank: dict, extraction_units: dict) -> list[dict]:
    units_by_id = {
        unit["unit_id"]: unit
        for unit in extraction_units.get("units", [])
        if unit.get("unit_id")
    }
    counts: dict[str, int] = {}
    for kc in kc_bank.get("kc_nodes", []):
        unit_id = str(kc.get("unit_id", "")).strip()
        if unit_id:
            counts[unit_id] = counts.get(unit_id, 0) + 1
    skipped = []
    for unit_id, unit in sorted(units_by_id.items()):
        count = counts.get(unit_id, 0)
        if count >= 2:
            continue
        skipped.append(
            {
                "unit_id": unit_id,
                "reason": "fewer_than_two_kcs",
                "kc_count": count,
                "unit_title": unit.get("unit_title", ""),
            }
        )
    return skipped


def _skipped_macros(kc_bank: dict, macro_spine: dict) -> list[dict]:
    counts: dict[str, int] = {}
    unit_counts: dict[str, set[str]] = {}
    for kc in kc_bank.get("kc_nodes", []):
        macro_id = str(kc.get("macro_id", "")).strip()
        unit_id = str(kc.get("unit_id", "")).strip()
        if macro_id:
            counts[macro_id] = counts.get(macro_id, 0) + 1
            if unit_id:
                unit_counts.setdefault(macro_id, set()).add(unit_id)
    skipped = []
    for macro in macro_spine.get("macro_nodes", []):
        macro_id = str(macro.get("macro_id", "")).strip()
        if not macro_id:
            continue
        kc_count = counts.get(macro_id, 0)
        unit_count = len(unit_counts.get(macro_id, set()))
        if kc_count >= 2 and unit_count >= 2:
            continue
        skipped.append(
            {
                "macro_id": macro_id,
                "reason": "fewer_than_two_kcs_or_units",
                "kc_count": kc_count,
                "unit_count": unit_count,
                "macro_title": macro.get("title", ""),
            }
        )
    return skipped


def _skipped_macro_edges(kc_bank: dict, macro_spine: dict) -> list[dict]:
    kcs_by_macro = _kcs_by_macro(kc_bank)
    skipped = []
    for idx, edge in enumerate(macro_spine.get("macro_edges", []), start=1):
        source = str(edge.get("source", "")).strip()
        target = str(edge.get("target", "")).strip()
        if kcs_by_macro.get(source) and kcs_by_macro.get(target):
            continue
        skipped.append(
            {
                "macro_edge_id": edge.get("edge_id") or f"ME{idx}",
                "source": source,
                "target": target,
                "reason": "source_or_target_macro_has_no_kcs",
                "source_kc_count": len(kcs_by_macro.get(source, [])),
                "target_kc_count": len(kcs_by_macro.get(target, [])),
            }
        )
    return skipped


def _units_by_id(extraction_units: dict) -> dict[str, dict]:
    return {
        unit["unit_id"]: unit
        for unit in extraction_units.get("units", [])
        if unit.get("unit_id")
    }


def _kcs_by_macro(kc_bank: dict) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for kc in kc_bank.get("kc_nodes", []):
        macro_id = str(kc.get("macro_id", "")).strip()
        if macro_id:
            out.setdefault(macro_id, []).append(kc)
    return out


def _representative_kcs(kcs: list[dict], units_by_id: dict[str, dict], limit: int) -> list[dict]:
    ranked = sorted(
        kcs,
        key=lambda kc: (
            0 if kc.get("importance") == "critical" else 1,
            _unit_order_key(units_by_id.get(kc.get("unit_id"), {})),
            _kc_sort_key(str(kc.get("kc_id", ""))),
        ),
    )
    return ranked[:limit]


def _macro_packet(macro: dict) -> dict:
    return {
        "macro_id": macro.get("macro_id"),
        "title": macro.get("title"),
        "role": macro.get("role"),
        "summary": macro.get("summary"),
        "importance": macro.get("importance"),
    }


def _kc_context_packet(kc: dict) -> dict:
    return {
        "kc_id": kc.get("kc_id"),
        "unit_id": kc.get("unit_id"),
        "macro_id": kc.get("macro_id"),
        "type": kc.get("type"),
        "importance": kc.get("importance"),
        "claim_strength": kc.get("claim_strength"),
        "scope": kc.get("scope"),
        "full_claim": kc.get("full_claim"),
        "evidence_text": kc.get("evidence_text"),
        "related_terms": kc.get("related_terms", []),
    }


def _unit_text_packets(unit_ids: set[str], units_by_id: dict[str, dict]) -> list[dict]:
    return [
        {
            "unit_id": unit_id,
            "unit_title": units_by_id.get(unit_id, {}).get("unit_title", ""),
            "unit_summary": units_by_id.get(unit_id, {}).get("unit_summary", ""),
            "source_text": units_by_id.get(unit_id, {}).get("source_text", ""),
            "source_paragraphs": _paragraphs_from_source(units_by_id.get(unit_id, {}).get("source_text", "")),
        }
        for unit_id in sorted(unit_ids, key=lambda uid: _unit_order_key(units_by_id.get(uid, {})))
        if unit_id in units_by_id
    ]


def _sliding_batches(items: list[dict], limit: int) -> list[list[dict]]:
    if len(items) <= limit:
        return [items]
    overlap = max(1, min(5, limit // 5))
    batches = []
    start = 0
    while start < len(items):
        end = min(len(items), start + limit)
        batches.append(items[start:end])
        if end >= len(items):
            break
        next_start = end - overlap
        start = next_start if next_start > start else end
    return batches


def _unit_order_key(unit: dict) -> tuple[int, int, str]:
    return (
        int(unit.get("window_order") or 0),
        int(unit.get("order_in_section") or 0),
        str(unit.get("unit_id", "")),
    )


def _paragraphs_from_source(source_text: object) -> list[dict]:
    paragraphs = [paragraph.strip() for paragraph in re.split(r"\n\s*\n", str(source_text or "")) if paragraph.strip()]
    if not paragraphs and str(source_text or "").strip():
        paragraphs = [str(source_text).strip()]
    return [
        {
            "paragraph_id": f"P{idx}",
            "text": paragraph,
        }
        for idx, paragraph in enumerate(paragraphs, start=1)
    ]


def _paragraph_by_id(paragraphs: object) -> dict[str, dict]:
    if not isinstance(paragraphs, list):
        return {}
    return {
        str(paragraph.get("paragraph_id", "")).strip(): paragraph
        for paragraph in paragraphs
        if isinstance(paragraph, dict) and str(paragraph.get("paragraph_id", "")).strip()
    }


def _source_text_from_paragraph_ids(
    label: str,
    paragraph_ids: list[str],
    paragraph_by_id: dict[str, dict],
) -> tuple[str, list[str]]:
    if not paragraph_ids:
        raise ValueError(f"{label} must include evidence_paragraph_ids.")
    unknown = [paragraph_id for paragraph_id in paragraph_ids if paragraph_id not in paragraph_by_id]
    if unknown:
        raise ValueError(f"{label} references unknown evidence_paragraph_ids: {unknown}")
    positions = sorted({int(paragraph_id[1:]) for paragraph_id in paragraph_ids})
    normalized_ids = [f"P{position}" for position in range(positions[0], positions[-1] + 1)]
    missing = [paragraph_id for paragraph_id in normalized_ids if paragraph_id not in paragraph_by_id]
    if missing:
        raise ValueError(f"{label} contiguous evidence range references missing paragraph IDs: {missing}")
    source_text = "\n\n".join(str(paragraph_by_id[paragraph_id].get("text", "")).strip() for paragraph_id in normalized_ids)
    if not source_text.strip():
        raise ValueError(f"{label} evidence paragraphs are empty.")
    return source_text, normalized_ids


def _text_in_source(text: str, source_text: str) -> bool:
    needle = _normalize_ws(text)
    haystack = _normalize_ws(source_text)
    return bool(needle and needle in haystack)


def _normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _string_list(values: object) -> list[str]:
    if not isinstance(values, list):
        return []
    return [str(value).strip() for value in values if str(value).strip()]


def _confidence(value: object) -> float:
    try:
        raw = float(value)
    except (TypeError, ValueError):
        raw = 0.0
    return round(max(0.0, min(1.0, raw)), 4)


def _kc_sort_key(kc_id: str) -> tuple[int, str]:
    match = re.search(r"(\d+)$", kc_id)
    return (int(match.group(1)) if match else 10**9, kc_id)


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
