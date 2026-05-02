from __future__ import annotations

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.model_client import OpenAICompatClient
from src.prompt_loader import load_prompt, render_prompt
from src.progress import log, span
from src.rubric_builder import build_kc_rubric


MACRO_META = {
    "M1": {"title": "Research Problem and Motivation", "role": "motivation", "summary": "Why the paper is needed and what gaps or problems motivate it."},
    "M2": {"title": "Core Method and Mechanisms", "role": "method", "summary": "The paper's proposed method, dataset construction, modules, and mechanisms."},
    "M3": {"title": "Experimental Design and Results", "role": "experiment", "summary": "Evaluation setup, empirical findings, analysis, and ablations."},
    "M4": {"title": "Conclusion, Limitations, and Open Evidence", "role": "conclusion", "summary": "Conclusions, contributions, limitations, and claims that remain under-validated."},
}

ALLOWED_RELATIONS = {"motivates", "solves", "mechanism_of", "tested_by", "explains_result", "contrasts_with"}
MACRO_IDS = ["M1", "M2", "M3", "M4"]


def _infer_type(claim: str, macro_role: str = "") -> str:
    c = claim.lower()
    role = macro_role.lower()
    if "problem" in role or "motivation" in role or "limitation" in role:
        return "problem"
    if "method" in role or "mechanism" in role or "module" in role:
        if "module" in c or "mechanism" in c:
            return "mechanism"
        return "method"
    if "experiment" in role or "result" in role or "ablation" in role or "analysis" in role:
        return "result"
    if "dataset" in role or "resource" in role:
        return "dataset"
    return "conclusion"


def _macro_score(claim: str, section: str = "") -> dict[str, int]:
    c = claim.lower()
    sec = section.lower()
    scores = {"M1": 0, "M2": 0, "M3": 0, "M4": 0}
    existing_dataset_context = (
        "video fact-checking datasets" in sec
        and any(name in c for name in ["checked", "fakesv", "mocheg", "vmh", "existing"])
    )
    if any(kw in sec for kw in ["abstract", "introduction", "related"]):
        scores["M1"] += 3
    if any(kw in sec for kw in ["method", "framework", "approach", "model", "dataset construction"]):
        scores["M2"] += 4
    if any(kw in sec for kw in ["experiment", "result", "analysis", "ablation", "case study", "statistics"]):
        scores["M3"] += 4
    if any(kw in sec for kw in ["conclusion", "limitation", "future"]):
        scores["M4"] += 4
    for kw in ["problem", "challenge", "motivation", "issue", "lack", "gap", "insufficient"]:
        if kw in c:
            scores["M1"] += 2
    if existing_dataset_context:
        scores["M1"] += 5
        scores["M2"] -= 3
    for kw in ["propose", "framework", "method", "module", "mechanism", "retriever", "descriptor", "manager", "reasoner", "construct"]:
        if kw in c:
            scores["M2"] += 2
    if "dataset" in c and any(kw in c for kw in ["novel", "proposed", "our", "we develop", "we construct"]):
        scores["M2"] += 2
    for kw in ["experiment", "result", "ablation", "accuracy", "table", "figure", "score", "outperform", "performance", "comparison"]:
        if kw in c:
            scores["M3"] += 2
    for kw in ["limitation", "future", "conclusion", "improvement", "remain", "further"]:
        if kw in c:
            scores["M4"] += 2
    return scores


def _assign_macro_id(claim: str, section: str = "") -> str:
    scores = _macro_score(claim, section)
    # Stable tie-breaker keeps claims from falling into M1 just because all scores are zero.
    order = ["M2", "M3", "M1", "M4"]
    return max(order, key=lambda mid: (scores[mid], -order.index(mid)))


def _short_label(claim: str) -> str:
    words = claim.strip().split()
    return " ".join(words[:10]) + ("..." if len(words) > 10 else "")


def _build_kc_nodes(
    kcs: list[dict],
    client: OpenAICompatClient | None = None,
    allow_offline_fallback: bool = False,
    macro_spine: dict | None = None,
) -> tuple[list[dict], dict[str, list[str]]]:
    macro_nodes = _macro_nodes_from_spine(macro_spine)
    macro_ids = [m["macro_id"] for m in macro_nodes] or MACRO_IDS
    macro_by_id = {m["macro_id"]: m for m in macro_nodes}
    groups = {macro_id: [] for macro_id in macro_ids}
    kc_nodes: list[dict] = []
    online_budget = _resolve_online_budget(len(kcs))
    if not allow_offline_fallback:
        online_budget = len(kcs)
    rubric_cache: dict[str, dict] = {}

    # Parallel online rubric generation for top-budget KCs.
    if client and client.is_ready():
        max_workers = int(os.getenv("RUBRIC_ONLINE_WORKERS", "4"))
        futures = {}
        log("rubric generation started", online_budget=online_budget, workers=max_workers)
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            for idx, kc in enumerate(kcs[:online_budget]):
                claim = _kc_claim(kc)
                macro_id = _resolve_kc_macro_id(kc, claim, macro_ids, allow_offline_fallback)
                kc_type = _valid_kc_type(kc.get("type")) or _infer_type(claim, macro_by_id.get(macro_id, {}).get("role", ""))
                importance = _valid_importance(kc.get("importance")) or ("critical" if macro_id in {"M1", "M2"} else "normal")
                if _has_rubric(kc):
                    continue
                fut = ex.submit(
                    build_kc_rubric,
                    kc["kc_id"],
                    claim,
                    _kc_evidence_text(kc),
                    kc_type,
                    importance,
                    client,
                    allow_offline_fallback,
                )
                futures[fut] = kc["kc_id"]
            for fut in as_completed(futures):
                kc_id = futures[fut]
                try:
                    rubric_cache[kc_id] = fut.result()
                    log("rubric generated", kc_id=kc_id, completed=len(rubric_cache), total=online_budget)
                except Exception:
                    log("rubric generation error", kc_id=kc_id)
                    pass

    for idx, kc in enumerate(kcs):
        claim = _kc_claim(kc)
        macro_id = _resolve_kc_macro_id(kc, claim, macro_ids, allow_offline_fallback)
        kc_type = _valid_kc_type(kc.get("type")) or _infer_type(claim, macro_by_id.get(macro_id, {}).get("role", ""))
        importance = _valid_importance(kc.get("importance")) or ("critical" if macro_id in {"M1", "M2"} else "normal")
        must_1 = _short_label(claim)
        node = {
            "kc_id": kc["kc_id"],
            "macro_id": macro_id,
            "type": kc_type,
            "short_label": must_1,
            "full_claim": claim,
            "importance": importance,
            "section": kc.get("section", ""),
            "section_id": kc.get("section_id", ""),
        }
        rubric = _existing_rubric(kc) or rubric_cache.get(kc["kc_id"])
        if not rubric and not allow_offline_fallback:
            raise RuntimeError(f"Online rubric generation failed for {kc['kc_id']} and offline fallback is disabled.")
        if not rubric:
            rubric = build_kc_rubric(
                kc_id=kc["kc_id"],
                full_claim=claim,
                evidence_text=_kc_evidence_text(kc),
                kc_type=kc_type,
                importance=node["importance"],
                client=client if idx < online_budget else None,
                allow_offline_fallback=allow_offline_fallback,
            )
        node.update(rubric)
        kc_nodes.append(node)
        groups[macro_id].append(kc["kc_id"])
    return kc_nodes, groups


def _valid_macro_id(value: object, macro_ids: list[str]) -> str | None:
    text = str(value or "").strip()
    return text if text in macro_ids else None


def _kc_claim(kc: dict) -> str:
    claim = str(kc.get("claim") or kc.get("full_claim") or "").strip()
    if not claim:
        raise ValueError(f"KC {kc.get('kc_id')} has no claim/full_claim.")
    return claim


def _kc_evidence_text(kc: dict) -> str:
    if kc.get("evidence_text"):
        return str(kc["evidence_text"])
    evidence = kc.get("evidence")
    if isinstance(evidence, str):
        return evidence
    if isinstance(evidence, list):
        texts = [str(item.get("text", "")).strip() for item in evidence if isinstance(item, dict)]
        return "\n".join(t for t in texts if t)
    return _kc_claim(kc)


def _has_rubric(kc: dict) -> bool:
    return bool(kc.get("must_include") and kc.get("acceptable_variants"))


def _existing_rubric(kc: dict) -> dict | None:
    if not _has_rubric(kc):
        return None
    return {
        "must_include": kc.get("must_include", []),
        "acceptable_variants": kc.get("acceptable_variants", []),
        "forbidden_claims": kc.get("forbidden_claims", []),
        "evidence": kc.get("evidence", []),
        "importance": kc.get("importance", "normal"),
        "type": kc.get("type", "method"),
    }


def _resolve_kc_macro_id(
    kc: dict,
    claim: str,
    macro_ids: list[str],
    allow_offline_fallback: bool,
) -> str:
    macro_id = _valid_macro_id(kc.get("macro_id"), macro_ids)
    if macro_id:
        return macro_id
    if not allow_offline_fallback:
        raise ValueError(
            f"KC {kc.get('kc_id')} references invalid macro_id={kc.get('macro_id')!r}. "
            f"Allowed macro IDs: {macro_ids}"
        )
    if set(MACRO_IDS).issubset(set(macro_ids)):
        return _assign_macro_id(claim, kc.get("section", ""))
    return macro_ids[0]


def _valid_kc_type(value: object) -> str | None:
    text = str(value or "").strip()
    return text if text in {"problem", "method", "mechanism", "dataset", "experiment", "result", "conclusion", "limitation", "background", "central_claim", "algorithm", "analysis"} else None


def _valid_importance(value: object) -> str | None:
    text = str(value or "").strip()
    return text if text in {"critical", "normal"} else None


def _rebalance_macro_minimum(kc_nodes: list[dict], groups: dict[str, list[str]], minimum: int = 2) -> None:
    if len(kc_nodes) < minimum * len(MACRO_IDS):
        return
    by_id = {kc["kc_id"]: kc for kc in kc_nodes}
    for macro_id in MACRO_IDS:
        while len(groups[macro_id]) < minimum:
            donor = max(MACRO_IDS, key=lambda mid: len(groups[mid]))
            if donor == macro_id or len(groups[donor]) <= minimum:
                return
            candidates = groups[donor]
            scored = [
                (kid, _macro_score(by_id[kid]["full_claim"], by_id[kid].get("section", ""))[macro_id])
                for kid in candidates
            ]
            best, best_score = max(scored, key=lambda item: item[1])
            if best_score <= 0:
                return
            groups[donor].remove(best)
            groups[macro_id].append(best)
            by_id[best]["macro_id"] = macro_id
            by_id[best]["type"] = _infer_type(by_id[best]["full_claim"], macro_id)
            by_id[best]["importance"] = "critical" if macro_id in {"M1", "M2"} else "normal"


def _resolve_online_budget(total_kc: int) -> int:
    """
    RUBRIC_ONLINE_BUDGET strategy:
    - integer N > 0: online for first N KCs
    - 0: no online
    - -1 / all / unlimited / full: all KCs online
    default: 8
    """
    raw = os.getenv("RUBRIC_ONLINE_BUDGET", "8").strip().lower()
    if raw in {"-1", "all", "unlimited", "full"}:
        return total_kc
    try:
        n = int(raw)
    except ValueError:
        return min(8, total_kc)
    if n < 0:
        return total_kc
    return min(n, total_kc)


def _build_reasoning_edges(kc_nodes: list[dict], groups: dict[str, list[str]]) -> list[dict]:
    by_id = {k["kc_id"]: k for k in kc_nodes}
    edges: list[dict] = []
    seen = set()

    def add(source: str, target: str, relation: str, desc: str) -> None:
        if relation not in ALLOWED_RELATIONS:
            return
        key = (source, target, relation)
        if source == target or key in seen:
            return
        seen.add(key)
        edges.append(
            {
                "edge_id": f"E{len(edges) + 1}",
                "source": source,
                "target": target,
                "relation": relation,
                "description": desc,
                "forbidden_claims": [],
            }
        )

    macro_order = list(groups.keys())
    if not set(MACRO_IDS).issubset(set(groups)):
        for left, right in zip(macro_order, macro_order[1:]):
            for a in groups.get(left, [])[:2]:
                for b in groups.get(right, [])[:2]:
                    add(a, b, "motivates", "Earlier Macro claim supports the later Macro claim.")
        return edges

    for a in groups["M1"][:3]:
        for b in groups["M2"][:3]:
            add(a, b, "motivates", "Problem claim motivates method design.")
            add(b, a, "solves", "Method claim addresses the stated problem.")
    for a in groups["M2"][:4]:
        for b in groups["M3"][:4]:
            add(a, b, "tested_by", "Method is validated by experiment.")
            add(a, b, "explains_result", "Mechanism explains observed results.")
    for a in groups["M2"][:4]:
        for b in groups["M2"][:4]:
            if a != b:
                ca = by_id[a]["full_claim"].lower()
                cb = by_id[b]["full_claim"].lower()
                if "module" in ca and ("framework" in cb or "method" in cb):
                    add(a, b, "mechanism_of", "Module claim is a mechanism of the method.")
    comp = [k["kc_id"] for k in kc_nodes if re.search(r"\b(compared|baseline|however|vs)\b", k["full_claim"].lower())]
    for i in range(0, len(comp) - 1):
        add(comp[i], comp[i + 1], "contrasts_with", "Comparative claims used for misleading test.")
    if not any(e["relation"] == "contrasts_with" for e in edges) and groups["M2"] and groups["M3"]:
        add(groups["M2"][0], groups["M3"][0], "contrasts_with", "Fallback contrast edge.")
    return edges


def _build_reasoning_paths(groups: dict[str, list[str]]) -> list[dict]:
    paths: list[dict] = []

    def add(pattern: str, seq: list[str], desc: str) -> None:
        if len(seq) < 3:
            return
        paths.append(
            {
                "path_id": f"P{len(paths) + 1}",
                "pattern": pattern,
                "kc_sequence": seq[:3],
                "description": desc,
                "trigger_condition": {"required_lit_kc": seq[:2], "target_kc": seq[2]},
                "forbidden_claims": [],
            }
        )

    if groups["M1"] and groups["M2"] and groups["M3"]:
        add("problem_method_result", [groups["M1"][0], groups["M2"][0], groups["M3"][0]], "Problem to method to result.")
    if len(groups["M2"]) >= 2 and groups["M3"]:
        add("module_mechanism_ablation", [groups["M2"][0], groups["M2"][1], groups["M3"][0]], "Module and mechanism validated by ablation/result.")
    if groups["M2"] and len(groups["M3"]) >= 2 and groups["M4"]:
        add("claim_evidence_conclusion", [groups["M2"][0], groups["M3"][0], groups["M4"][0]], "Claim supported by evidence to conclusion.")
    if len(groups["M2"]) >= 2 and groups["M3"]:
        add("method_baseline_contrast", [groups["M2"][0], groups["M3"][0], groups["M2"][1]], "Method baseline contrast.")
    if groups["M4"] and groups["M3"] and groups["M1"]:
        add("conclusion_missing_evidence_limitation", [groups["M4"][0], groups["M3"][0], groups["M1"][0]], "Conclusion and limitation check.")
    return paths[:5]


def _build_reasoning_edges_online(kc_nodes: list[dict], macro_nodes: list[dict], client: OpenAICompatClient | None) -> list[dict] | None:
    if not client or not client.is_ready():
        return None
    try:
        tpl = load_prompt("build_edges.txt")
        context = {"macro_nodes": macro_nodes, "kc_nodes": kc_nodes}
        user_prompt = render_prompt(tpl, graph_context_json=json.dumps(context, ensure_ascii=False))
        with span("generate reasoning edges", kcs=len(kc_nodes)):
            result = client.chat_json(
                system_prompt="You are a strict graph-construction assistant for paper evaluation.",
                user_prompt=user_prompt,
            )
        edges = result.get("reasoning_edges", [])
        ok = []
        valid_kc_ids = {k["kc_id"] for k in kc_nodes}
        for e in edges:
            rel = e.get("relation")
            if rel not in ALLOWED_RELATIONS:
                continue
            source = e.get("source")
            target = e.get("target")
            if source not in valid_kc_ids or target not in valid_kc_ids or source == target:
                continue
            ok.append(
                {
                    "edge_id": f"E{len(ok) + 1}",
                    "source": source,
                    "target": target,
                    "relation": rel,
                    "description": e.get("description", ""),
                    "forbidden_claims": _normalize_forbidden_claims(e.get("forbidden_claims", []), f"E{len(ok) + 1}"),
                }
            )
        log("reasoning edges parsed", count=len(ok))
        return ok or None
    except Exception:
        return None


def build_reasoning_edges_for_kcs(
    kc_nodes: list[dict],
    macro_nodes: list[dict],
    client: OpenAICompatClient | None,
    allow_offline_fallback: bool = False,
) -> list[dict]:
    edges = _build_reasoning_edges_online(kc_nodes, macro_nodes, client)
    if not edges and not allow_offline_fallback:
        raise RuntimeError("Online reasoning edge generation failed and offline fallback is disabled.")
    if edges:
        _ensure_edge_forbidden_claims(edges, kc_nodes)
        return edges
    groups = {
        macro["macro_id"]: [
            kc["kc_id"]
            for kc in kc_nodes
            if kc.get("macro_id") == macro.get("macro_id")
        ]
        for macro in macro_nodes
    }
    fallback_edges = _build_reasoning_edges(kc_nodes, groups)
    _ensure_edge_forbidden_claims(fallback_edges, kc_nodes)
    return fallback_edges


def _build_reasoning_paths_online(
    kc_nodes: list[dict],
    macro_nodes: list[dict],
    reasoning_edges: list[dict],
    client: OpenAICompatClient | None,
) -> list[dict] | None:
    if not client or not client.is_ready():
        return None
    try:
        tpl = load_prompt("build_paths.txt")
        context = {"macro_nodes": macro_nodes, "kc_nodes": kc_nodes, "reasoning_edges": reasoning_edges}
        user_prompt = render_prompt(tpl, graph_context_json=json.dumps(context, ensure_ascii=False))
        with span("generate reasoning paths", kcs=len(kc_nodes), edges=len(reasoning_edges)):
            result = client.chat_json(
                system_prompt="You are a strict reasoning-path construction assistant.",
                user_prompt=user_prompt,
            )
        paths = result.get("reasoning_paths", [])
        validated = _validate_reasoning_paths(paths, kc_nodes)
        log("reasoning paths parsed", raw=len(paths), valid=len(validated))
        return validated or None
    except Exception:
        return None


def build_master_graph(
    paper_id: str,
    paper_text_path: str,
    kcs: list[dict],
    client: OpenAICompatClient | None = None,
    allow_offline_fallback: bool = False,
    macro_spine: dict | None = None,
    kc_bank_path: str | None = None,
    active_kc_path: str | None = None,
    precomputed_reasoning_edges: list[dict] | None = None,
) -> dict:
    macro_source_nodes = _macro_nodes_from_spine(macro_spine)
    kc_nodes, groups = _build_kc_nodes(
        kcs,
        client=client,
        allow_offline_fallback=allow_offline_fallback,
        macro_spine=macro_spine,
    )
    log(
        "KC nodes built",
        count=len(kc_nodes),
        macro_counts=json.dumps({mid: len(ids) for mid, ids in groups.items()}, ensure_ascii=False),
    )
    macro_nodes = []
    for macro in macro_source_nodes:
        macro_id = macro["macro_id"]
        macro_nodes.append(
            {
                "macro_id": macro_id,
                "order": macro.get("order"),
                "title": macro.get("title", macro_id),
                "role": macro.get("role", ""),
                "summary": macro.get("summary", ""),
                "source_sections": macro.get("source_sections", []),
                "expected_reader_question": macro.get("expected_reader_question", ""),
                "kc_ids": groups[macro_id],
                "prerequisite_macro_ids": macro.get("prerequisite_macro_ids", []),
                "next_macro_ids": macro.get("next_macro_ids", []),
                "importance": macro.get("importance", "normal"),
            }
        )
    active_ids = {kc["kc_id"] for kc in kc_nodes}
    edges = _filter_precomputed_edges(precomputed_reasoning_edges, active_ids)
    if not edges:
        edges = build_reasoning_edges_for_kcs(
            kc_nodes,
            macro_nodes,
            client,
            allow_offline_fallback=allow_offline_fallback,
        )
    paths = _build_reasoning_paths_online(kc_nodes, macro_nodes, edges, client)
    if not paths and not allow_offline_fallback:
        raise RuntimeError("Online reasoning path generation failed and offline fallback is disabled.")
    if not paths:
        paths = _build_reasoning_paths(groups)
    return {
        "paper_id": paper_id,
        "paper_title": paper_id,
        "paper_text_path": paper_text_path,
        "macro_spine_path": "data/graphs/macro_spine.json" if macro_spine else None,
        "kc_bank_path": kc_bank_path,
        "active_kc_path": active_kc_path,
        "macro_nodes": macro_nodes,
        "macro_edges": macro_spine.get("macro_edges", []) if macro_spine else [],
        "kc_nodes": kc_nodes,
        "active_kc_ids": [kc["kc_id"] for kc in kc_nodes],
        "reasoning_edges": edges,
        "reasoning_paths": paths,
    }


def _filter_precomputed_edges(edges: list[dict] | None, active_ids: set[str]) -> list[dict]:
    if not edges:
        return []
    filtered = []
    for edge in edges:
        source = edge.get("source")
        target = edge.get("target")
        if source not in active_ids or target not in active_ids:
            continue
        item = dict(edge)
        item["edge_id"] = f"E{len(filtered) + 1}"
        filtered.append(item)
    return filtered


def _macro_nodes_from_spine(macro_spine: dict | None) -> list[dict]:
    if macro_spine and macro_spine.get("macro_nodes"):
        return sorted(macro_spine["macro_nodes"], key=lambda item: int(item.get("order") or 0))
    out = []
    for idx, macro_id in enumerate(MACRO_IDS, start=1):
        meta = MACRO_META[macro_id]
        out.append(
            {
                "macro_id": macro_id,
                "order": idx,
                "title": meta["title"],
                "role": meta["role"],
                "summary": meta["summary"],
                "source_sections": [],
                "expected_reader_question": "",
                "prerequisite_macro_ids": [] if macro_id == "M1" else [f"M{int(macro_id[1]) - 1}"],
                "next_macro_ids": [] if macro_id == "M4" else [f"M{int(macro_id[1]) + 1}"],
                "importance": "critical" if macro_id in {"M1", "M2"} else "normal",
            }
        )
    return out


def _validate_reasoning_paths(paths: list[dict], kc_nodes: list[dict]) -> list[dict]:
    valid_kc_ids = {k["kc_id"] for k in kc_nodes}
    out: list[dict] = []
    for p in paths:
        seq = [kid for kid in p.get("kc_sequence", []) if kid in valid_kc_ids]
        seq = list(dict.fromkeys(seq))
        if len(seq) < 3:
            continue
        path_id = f"P{len(out) + 1}"
        out.append(
            {
                "path_id": path_id,
                "pattern": p.get("pattern", "claim_evidence_conclusion"),
                "kc_sequence": seq[:3],
                "description": p.get("description", ""),
                "trigger_condition": {
                    "required_lit_kc": seq[:2],
                    "target_kc": seq[2],
                },
                "forbidden_claims": _normalize_forbidden_claims(p.get("forbidden_claims", []), path_id),
            }
        )
        if len(out) >= 5:
            break
    return out


def _normalize_forbidden_claims(items: list, owner_id: str) -> list[dict]:
    normalized: list[dict] = []
    for idx, item in enumerate(items[:4], start=1):
        if isinstance(item, dict):
            claim = str(item.get("claim", "")).strip()
            if not claim:
                continue
            normalized.append(
                {
                    "claim_id": str(item.get("claim_id") or f"FC_{owner_id}_{idx}"),
                    "claim": claim,
                    "type": str(item.get("type") or "logic_hallucination"),
                    "severity": str(item.get("severity") or "high"),
                    "why_wrong": str(item.get("why_wrong") or "This claim is inconsistent with the graph evidence."),
                    "followup_hint": str(item.get("followup_hint") or "Ask the model to restate the relation using the paper evidence."),
                }
            )
        elif isinstance(item, str) and item.strip():
            normalized.append(
                {
                    "claim_id": f"FC_{owner_id}_{idx}",
                    "claim": item.strip(),
                    "type": "logic_hallucination",
                    "severity": "high",
                    "why_wrong": "This claim is inconsistent with the reasoning edge/path.",
                    "followup_hint": "Ask the model to compare the claim against the paper evidence.",
                }
            )
    return normalized


def _ensure_edge_forbidden_claims(edges: list[dict], kc_nodes: list[dict]) -> None:
    by_id = {kc["kc_id"]: kc for kc in kc_nodes}
    for edge in edges:
        if edge.get("forbidden_claims"):
            edge["forbidden_claims"] = _normalize_forbidden_claims(edge["forbidden_claims"], edge["edge_id"])
            continue
        source = by_id.get(edge.get("source"), {})
        target = by_id.get(edge.get("target"), {})
        source_label = source.get("short_label") or source.get("full_claim") or edge.get("source")
        target_label = target.get("short_label") or target.get("full_claim") or edge.get("target")
        relation = edge.get("relation", "relates_to")
        edge["forbidden_claims"] = [
            {
                "claim_id": f"FC_{edge['edge_id']}_1",
                "claim": f"The relation between {source_label} and {target_label} is the reverse of {relation}.",
                "type": "wrong_relation",
                "severity": "high",
                "why_wrong": "The graph encodes a directed reasoning relation; reversing it changes the paper's argument structure.",
                "followup_hint": "Ask whether the source claim supports the target claim, or whether the answer has reversed the direction.",
            }
        ]
