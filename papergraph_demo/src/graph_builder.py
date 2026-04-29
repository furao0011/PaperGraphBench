from __future__ import annotations

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.model_client import OpenAICompatClient
from src.prompt_loader import load_prompt, render_prompt
from src.rubric_builder import build_kc_rubric


MACRO_META = {
    "M1": {"title": "研究问题与动机", "role": "motivation", "summary": "论文提出方法的动机与问题背景。"},
    "M2": {"title": "核心方法与机制", "role": "method", "summary": "核心方法、模块与机制解释。"},
    "M3": {"title": "实验设计与结果", "role": "experiment", "summary": "实验验证、结果与分析。"},
    "M4": {"title": "结论、局限与未验证点", "role": "conclusion", "summary": "结论、局限和未来改进。"},
}

ALLOWED_RELATIONS = {"motivates", "solves", "mechanism_of", "tested_by", "explains_result", "contrasts_with"}


def _infer_type(claim: str, macro_id: str) -> str:
    c = claim.lower()
    if macro_id == "M1":
        return "problem"
    if macro_id == "M2":
        if "module" in c or "mechanism" in c:
            return "mechanism"
        return "method"
    if macro_id == "M3":
        return "result"
    return "conclusion"


def _macro_score(claim: str) -> dict[str, int]:
    c = claim.lower()
    scores = {"M1": 0, "M2": 0, "M3": 0, "M4": 0}
    for kw in ["problem", "challenge", "motivation", "issue", "lack"]:
        if kw in c:
            scores["M1"] += 2
    for kw in ["propose", "framework", "method", "module", "mechanism", "retriever", "descriptor"]:
        if kw in c:
            scores["M2"] += 2
    for kw in ["experiment", "result", "ablation", "accuracy", "table", "figure", "score"]:
        if kw in c:
            scores["M3"] += 2
    for kw in ["limitation", "future", "conclusion", "improvement"]:
        if kw in c:
            scores["M4"] += 2
    return scores


def _short_label(claim: str) -> str:
    words = claim.strip().split()
    return " ".join(words[:10]) + ("..." if len(words) > 10 else "")


def _build_kc_nodes(kcs: list[dict], client: OpenAICompatClient | None = None) -> tuple[list[dict], dict[str, list[str]]]:
    groups = {"M1": [], "M2": [], "M3": [], "M4": []}
    kc_nodes: list[dict] = []
    online_budget = _resolve_online_budget(len(kcs))
    rubric_cache: dict[str, dict] = {}

    # Parallel online rubric generation for top-budget KCs.
    if client and client.is_ready():
        max_workers = int(os.getenv("RUBRIC_ONLINE_WORKERS", "4"))
        futures = {}
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            for idx, kc in enumerate(kcs[:online_budget]):
                claim = kc["claim"].strip()
                scores = _macro_score(claim)
                macro_id = max(scores, key=scores.get)
                kc_type = _infer_type(claim, macro_id)
                importance = "critical" if macro_id in {"M1", "M2"} else "normal"
                fut = ex.submit(
                    build_kc_rubric,
                    kc["kc_id"],
                    claim,
                    kc["evidence"],
                    kc_type,
                    importance,
                    client,
                )
                futures[fut] = kc["kc_id"]
            for fut in as_completed(futures):
                kc_id = futures[fut]
                try:
                    rubric_cache[kc_id] = fut.result()
                except Exception:
                    pass

    for idx, kc in enumerate(kcs):
        claim = kc["claim"].strip()
        scores = _macro_score(claim)
        macro_id = max(scores, key=scores.get)
        kc_type = _infer_type(claim, macro_id)
        must_1 = _short_label(claim)
        node = {
            "kc_id": kc["kc_id"],
            "macro_id": macro_id,
            "type": kc_type,
            "short_label": must_1,
            "full_claim": claim,
            "importance": "critical" if macro_id in {"M1", "M2"} else "normal",
        }
        rubric = rubric_cache.get(kc["kc_id"])
        if not rubric:
            rubric = build_kc_rubric(
                kc_id=kc["kc_id"],
                full_claim=claim,
                evidence_text=kc["evidence"],
                kc_type=kc_type,
                importance=node["importance"],
                client=client if idx < online_budget else None,
            )
        node.update(rubric)
        kc_nodes.append(node)
        groups[macro_id].append(kc["kc_id"])
    return kc_nodes, groups


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
        result = client.chat_json(
            system_prompt="You are a strict graph-construction assistant for paper evaluation.",
            user_prompt=user_prompt,
        )
        edges = result.get("reasoning_edges", [])
        ok = []
        for i, e in enumerate(edges, start=1):
            rel = e.get("relation")
            if rel not in ALLOWED_RELATIONS:
                continue
            ok.append(
                {
                    "edge_id": f"E{i}",
                    "source": e.get("source"),
                    "target": e.get("target"),
                    "relation": rel,
                    "description": e.get("description", ""),
                    "forbidden_claims": e.get("forbidden_claims", []),
                }
            )
        return ok or None
    except Exception:
        return None


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
        result = client.chat_json(
            system_prompt="You are a strict reasoning-path construction assistant.",
            user_prompt=user_prompt,
        )
        paths = result.get("reasoning_paths", [])
        return paths[:5] if paths else None
    except Exception:
        return None


def build_master_graph(
    paper_id: str,
    paper_text_path: str,
    kcs: list[dict],
    client: OpenAICompatClient | None = None,
) -> dict:
    kc_nodes, groups = _build_kc_nodes(kcs, client=client)
    macro_nodes = []
    for macro_id in ["M1", "M2", "M3", "M4"]:
        meta = MACRO_META[macro_id]
        macro_nodes.append(
            {
                "macro_id": macro_id,
                "title": meta["title"],
                "role": meta["role"],
                "summary": meta["summary"],
                "kc_ids": groups[macro_id][:5],
                "prerequisite_macro_ids": [] if macro_id == "M1" else [f"M{int(macro_id[1]) - 1}"],
            }
        )
    edges = _build_reasoning_edges_online(kc_nodes, macro_nodes, client)
    if not edges:
        edges = _build_reasoning_edges(kc_nodes, groups)
    paths = _build_reasoning_paths_online(kc_nodes, macro_nodes, edges, client)
    if not paths:
        paths = _build_reasoning_paths(groups)
    return {
        "paper_id": paper_id,
        "paper_title": paper_id,
        "paper_text_path": paper_text_path,
        "macro_nodes": macro_nodes,
        "kc_nodes": kc_nodes,
        "reasoning_edges": edges,
        "reasoning_paths": paths,
    }
