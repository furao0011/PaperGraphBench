from __future__ import annotations

import json
import math
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.model_client import OpenAICompatClient
from src.progress import log
from src.prompt_loader import load_prompt, render_prompt
from src.rubric_builder import build_kc_rubric


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


def build_kc_bank(
    paper_id: str,
    candidates: list[dict],
    macro_spine: dict,
    client: OpenAICompatClient,
    allow_offline_fallback: bool = False,
) -> dict:
    if not candidates:
        raise ValueError("Cannot build KC Bank from an empty candidate pool.")
    if not client or not client.is_ready():
        raise RuntimeError("KC Bank construction requires a configured online LLM client.")
    if not client.embeddings_ready():
        raise RuntimeError("KC Bank evidence scoring requires EMBED_MODEL and a configured embeddings endpoint.")

    macro_by_id = {
        m["macro_id"]: m
        for m in macro_spine.get("macro_nodes", [])
        if m.get("macro_id")
    }
    if not macro_by_id:
        raise ValueError("KC Bank construction requires a non-empty Macro Spine.")

    max_bank = _env_int("KC_BANK_MAX", 120, minimum=1)
    selected_candidates = candidates[:max_bank]
    nodes = []
    for idx, candidate in enumerate(selected_candidates, start=1):
        macro_id = str(candidate.get("macro_id", "")).strip()
        if macro_id not in macro_by_id:
            raise ValueError(f"KC candidate {candidate.get('candidate_id')} references invalid macro_id={macro_id!r}.")
        claim = str(candidate.get("claim", "")).strip()
        if not claim:
            raise ValueError(f"KC candidate {candidate.get('candidate_id')} has empty claim.")
        evidence_text = str(candidate.get("evidence", "")).strip() or claim
        kc_type = _valid_type(candidate.get("type")) or _infer_type_from_macro(macro_by_id[macro_id].get("role", ""))
        importance = _valid_importance(candidate.get("importance")) or macro_by_id[macro_id].get("importance", "normal")
        if importance not in {"critical", "normal"}:
            importance = "normal"
        kc_id = f"KC{idx}"
        nodes.append(
            {
                "kc_id": kc_id,
                "source_candidate_id": candidate.get("candidate_id", f"C{idx}"),
                "macro_id": macro_id,
                "type": kc_type,
                "source_section": candidate.get("section", ""),
                "source_section_id": candidate.get("section_id", ""),
                "source_span_ids": [candidate.get("section_id", "")] if candidate.get("section_id") else [],
                "short_label": _short_label(claim),
                "claim": claim,
                "full_claim": claim,
                "evidence_text": evidence_text,
                "evidence": [
                    {
                        "section": candidate.get("section", ""),
                        "span_id": candidate.get("section_id", "") or candidate.get("candidate_id", f"C{idx}"),
                        "text": evidence_text,
                    }
                ],
                "importance": importance,
                "importance_scores": {},
                "llm_scores_raw": {},
                "flags": {
                    "active_for_question_generation": False,
                    "active_for_core_metrics": False,
                    "usable_for_claim_verification": True,
                },
            }
        )

    _attach_evidence_quality(nodes, client)
    _attach_llm_subjective_scores(nodes, macro_spine, client)
    _attach_rubrics(nodes, client, allow_offline_fallback)
    log("KC Bank built", candidates=len(candidates), bank_kcs=len(nodes), dedupe_or_merge="disabled")
    return {"paper_id": paper_id, "kc_nodes": nodes}


def finalize_kc_bank_scores(
    kc_bank: dict,
    macro_spine: dict,
    reasoning_edges: list[dict],
) -> None:
    macro_scores = _macro_centrality_scores(macro_spine)
    graph_scores = _graph_connectivity_scores(kc_bank.get("kc_nodes", []), reasoning_edges, macro_spine)
    for node in kc_bank.get("kc_nodes", []):
        macro_centrality = macro_scores.get(node["macro_id"], 0.0)
        graph_connectivity = graph_scores.get(node["kc_id"], 0.0)
        scores = node.setdefault("importance_scores", {})
        scores["macro_centrality"] = macro_centrality
        scores["graph_connectivity"] = graph_connectivity
        _require_score(scores, "evidence_quality", node["kc_id"])
        _require_score(scores, "claim_specificity", node["kc_id"])
        _require_score(scores, "questionability", node["kc_id"])
        final_score = (
            0.30 * scores["macro_centrality"]
            + 0.25 * scores["evidence_quality"]
            + 0.20 * scores["claim_specificity"]
            + 0.15 * scores["questionability"]
            + 0.10 * scores["graph_connectivity"]
        )
        scores["final_importance_score"] = round(final_score, 4)
        node["scores"] = scores


def _attach_evidence_quality(nodes: list[dict], client: OpenAICompatClient) -> None:
    texts = []
    for node in nodes:
        texts.append(node["full_claim"])
        texts.append(node["evidence_text"])
    vectors = client.embed_texts(texts)
    for idx, node in enumerate(nodes):
        claim_vec = vectors[2 * idx]
        evidence_vec = vectors[2 * idx + 1]
        sim = _cosine_similarity(claim_vec, evidence_vec)
        quality = max(0.0, min(1.0, (sim - 0.45) / 0.40))
        node["importance_scores"]["evidence_quality"] = round(quality, 4)
        node["importance_scores"]["evidence_similarity"] = round(sim, 4)


def _attach_llm_subjective_scores(
    nodes: list[dict],
    macro_spine: dict,
    client: OpenAICompatClient,
) -> None:
    tpl = load_prompt("score_kc_subjective.txt")
    payload = [
        {
            "kc_id": node["kc_id"],
            "macro_id": node["macro_id"],
            "type": node["type"],
            "full_claim": node["full_claim"],
            "evidence": node["evidence_text"][:1200],
        }
        for node in nodes
    ]
    result = client.chat_json(
        system_prompt="You score subjective KC quality for paper evaluation. Return JSON only.",
        user_prompt=render_prompt(
            tpl,
            macro_spine_json=json.dumps(macro_spine, ensure_ascii=False, indent=2),
            kc_nodes_json=json.dumps(payload, ensure_ascii=False, indent=2),
        ),
        temperature=0.1,
    )
    raw_scores = result.get("scores", [])
    if not isinstance(raw_scores, list):
        raise ValueError("score_kc_subjective response must contain scores list.")
    by_id = {item.get("kc_id"): item for item in raw_scores if isinstance(item, dict)}
    missing = [node["kc_id"] for node in nodes if node["kc_id"] not in by_id]
    if missing:
        raise ValueError(f"Subjective KC scoring missing KC IDs: {missing}")

    for node in nodes:
        raw = by_id[node["kc_id"]]
        specificity_raw = _score_1_5(raw, "claim_specificity_score", node["kc_id"])
        questionability_raw = _score_1_5(raw, "questionability_score", node["kc_id"])
        node["importance_scores"]["claim_specificity"] = round((specificity_raw - 1) / 4, 4)
        node["importance_scores"]["questionability"] = round((questionability_raw - 1) / 4, 4)
        node["llm_scores_raw"] = {
            "claim_specificity_score": specificity_raw,
            "questionability_score": questionability_raw,
            "reason": str(raw.get("reason", "")).strip(),
        }


def _attach_rubrics(
    nodes: list[dict],
    client: OpenAICompatClient,
    allow_offline_fallback: bool,
) -> None:
    workers = _env_int("RUBRIC_ONLINE_WORKERS", 4, minimum=1)
    futures = {}
    with ThreadPoolExecutor(max_workers=min(workers, len(nodes))) as ex:
        for node in nodes:
            futures[
                ex.submit(
                    build_kc_rubric,
                    node["kc_id"],
                    node["full_claim"],
                    node["evidence_text"],
                    node["type"],
                    node["importance"],
                    client,
                    allow_offline_fallback,
                )
            ] = node
        for fut in as_completed(futures):
            node = futures[fut]
            node.update(fut.result())
            log("KC Bank rubric generated", kc_id=node["kc_id"])


def _macro_centrality_scores(macro_spine: dict) -> dict[str, float]:
    macro_ids = [m["macro_id"] for m in macro_spine.get("macro_nodes", [])]
    neighbors = {macro_id: [] for macro_id in macro_ids}
    for edge in macro_spine.get("macro_edges", []):
        source = edge.get("source")
        target = edge.get("target")
        if source in neighbors and target in neighbors and source != target:
            neighbors[source].append(target)
            neighbors[target].append(source)
    max_degree = max((len(set(v)) for v in neighbors.values()), default=0)
    scores = {}
    for macro_id, linked in neighbors.items():
        degree = len(set(linked))
        degree_score = _log_score(degree, max_degree)
        entropy = _entropy_score(linked, len(macro_ids))
        scores[macro_id] = round(0.7 * degree_score + 0.3 * entropy, 4)
    return scores


def _graph_connectivity_scores(
    kc_nodes: list[dict],
    reasoning_edges: list[dict],
    macro_spine: dict,
) -> dict[str, float]:
    macro_count = len(macro_spine.get("macro_nodes", []))
    by_kc = {node["kc_id"]: node for node in kc_nodes}
    neighbors = {node["kc_id"]: [] for node in kc_nodes}
    for edge in reasoning_edges:
        source = edge.get("source")
        target = edge.get("target")
        if source in neighbors and target in neighbors and source != target:
            neighbors[source].append(target)
            neighbors[target].append(source)
    max_degree = max((len(set(v)) for v in neighbors.values()), default=0)
    scores = {}
    for kc_id, linked_ids in neighbors.items():
        degree = len(set(linked_ids))
        degree_score = _log_score(degree, max_degree)
        neighbor_macros = [
            by_kc[nid]["macro_id"]
            for nid in set(linked_ids)
            if nid in by_kc
        ]
        entropy = _entropy_score(neighbor_macros, macro_count)
        scores[kc_id] = round(0.7 * degree_score + 0.3 * entropy, 4)
    return scores


def _log_score(degree: int, max_degree: int) -> float:
    if degree <= 0 or max_degree <= 0:
        return 0.0
    return math.log(1 + degree) / math.log(1 + max_degree)


def _entropy_score(items: list[str], total_categories: int) -> float:
    if not items or total_categories <= 1:
        return 0.0
    counts: dict[str, int] = {}
    for item in items:
        counts[item] = counts.get(item, 0) + 1
    total = sum(counts.values())
    entropy = 0.0
    for count in counts.values():
        p = count / total
        entropy -= p * math.log(p)
    return entropy / math.log(total_categories)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        raise ValueError("Embedding vector has zero norm.")
    return dot / (norm_a * norm_b)


def _require_score(scores: dict, key: str, kc_id: str) -> None:
    if key not in scores:
        raise ValueError(f"KC {kc_id} missing importance score: {key}")


def _score_1_5(raw: dict, key: str, kc_id: str) -> int:
    try:
        value = int(raw[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"KC {kc_id} missing integer {key}.") from exc
    if value < 1 or value > 5:
        raise ValueError(f"KC {kc_id} {key} must be in [1, 5], got {value}.")
    return value


def _short_label(claim: str) -> str:
    words = claim.strip().split()
    return " ".join(words[:10]) + ("..." if len(words) > 10 else "")


def _valid_type(value: object) -> str | None:
    text = str(value or "").strip()
    return text if text in VALID_KC_TYPES else None


def _valid_importance(value: object) -> str | None:
    text = str(value or "").strip()
    return text if text in {"critical", "normal"} else None


def _infer_type_from_macro(role: str) -> str:
    r = role.lower()
    if "problem" in r or "motivation" in r:
        return "problem"
    if "dataset" in r or "resource" in r:
        return "dataset"
    if "method" in r or "mechanism" in r or "module" in r:
        return "method"
    if "result" in r or "experiment" in r or "ablation" in r or "analysis" in r:
        return "result"
    if "limitation" in r:
        return "limitation"
    return "conclusion"


def _env_int(name: str, default: int, minimum: int = 0) -> int:
    raw = os.getenv(name)
    try:
        value = int(raw) if raw is not None and raw.strip() else default
    except ValueError:
        value = default
    return max(minimum, value)

