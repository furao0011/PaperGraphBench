from __future__ import annotations

import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.model_client import OpenAICompatClient
from src.progress import log
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
    macro_nodes = macro_spine.get("macro_nodes", [])
    macro_by_id = {m["macro_id"]: m for m in macro_nodes if m.get("macro_id")}
    if not macro_by_id:
        raise ValueError("KC Bank construction requires a non-empty Macro Spine.")

    max_bank = _env_int("KC_BANK_MAX", 120, minimum=1)
    merged = _dedupe_candidates(candidates, macro_by_id)[:max_bank]
    if not merged:
        raise ValueError("KC Bank construction produced no valid KC nodes after deduplication.")

    nodes = []
    for idx, item in enumerate(merged, start=1):
        macro_id = str(item.get("macro_id", "")).strip()
        if macro_id not in macro_by_id:
            raise ValueError(f"KC candidate references invalid macro_id={macro_id!r}.")
        kc_id = f"KC{idx}"
        evidence_text = str(item.get("evidence", "")).strip()
        full_claim = str(item.get("claim", "")).strip()
        if not full_claim:
            raise ValueError(f"KC candidate {item.get('candidate_id')} has empty claim.")
        if not evidence_text:
            evidence_text = full_claim
        kc_type = _valid_type(item.get("type")) or _infer_type_from_macro(macro_by_id[macro_id].get("role", ""))
        importance = _valid_importance(item.get("importance")) or macro_by_id[macro_id].get("importance", "normal")
        if importance not in {"critical", "normal"}:
            importance = "normal"
        scores = _score_candidate(item, macro_by_id[macro_id])
        nodes.append(
            {
                "kc_id": kc_id,
                "candidate_id": item.get("candidate_id", f"C{idx}"),
                "macro_id": macro_id,
                "type": kc_type,
                "source_section": item.get("section", ""),
                "source_section_id": item.get("section_id", ""),
                "source_span_ids": [item.get("section_id", "")] if item.get("section_id") else [],
                "short_label": _short_label(full_claim),
                "claim": full_claim,
                "full_claim": full_claim,
                "evidence_text": evidence_text,
                "evidence": [
                    {
                        "section": item.get("section", ""),
                        "span_id": item.get("section_id", "") or item.get("candidate_id", f"C{idx}"),
                        "text": evidence_text,
                    }
                ],
                "importance": importance,
                "scores": scores,
                "flags": {
                    "active_for_question_generation": False,
                    "active_for_core_metrics": False,
                    "usable_for_claim_verification": True,
                },
                "merged_from": item.get("merged_from", [item.get("candidate_id", f"C{idx}")]),
            }
        )

    _attach_rubrics(nodes, client, allow_offline_fallback)
    log("KC Bank built", candidates=len(candidates), bank_kcs=len(nodes))
    return {"paper_id": paper_id, "kc_nodes": nodes}


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
            rubric = fut.result()
            node.update(rubric)
            log("KC Bank rubric generated", kc_id=node["kc_id"])


def _dedupe_candidates(candidates: list[dict], macro_by_id: dict[str, dict]) -> list[dict]:
    by_norm: dict[str, dict] = {}
    for idx, candidate in enumerate(candidates, start=1):
        claim = str(candidate.get("claim", "")).strip()
        norm = _normalize_claim(claim)
        if not norm:
            continue
        item = dict(candidate)
        item.setdefault("candidate_id", f"C{idx}")
        if norm not in by_norm:
            item["merged_from"] = [item["candidate_id"]]
            by_norm[norm] = item
            continue
        existing = by_norm[norm]
        existing["merged_from"].append(item["candidate_id"])
        if _candidate_quality(item, macro_by_id) > _candidate_quality(existing, macro_by_id):
            item["merged_from"] = existing["merged_from"]
            by_norm[norm] = item
    ordered = sorted(
        by_norm.values(),
        key=lambda item: (
            int(item.get("section_index", 0)),
            -_candidate_quality(item, macro_by_id),
            str(item.get("claim", "")),
        ),
    )
    return ordered


def _score_candidate(candidate: dict, macro: dict) -> dict:
    claim = str(candidate.get("claim", ""))
    evidence = str(candidate.get("evidence", ""))
    macro_centrality = 0.9 if macro.get("importance") == "critical" else 0.65
    evidence_quality = min(1.0, max(0.35, len(evidence.split()) / 45))
    claim_words = len(claim.split())
    claim_specificity = min(1.0, max(0.35, claim_words / 28))
    if any(ch.isdigit() for ch in claim):
        claim_specificity = min(1.0, claim_specificity + 0.08)
    questionability = 0.55
    if re.search(r"\b(why|because|therefore|shows|improves|outperform|ablation|result|mechanism|limitation)\b", claim.lower()):
        questionability = 0.82
    graph_connectivity = 0.5
    final_score = (
        0.30 * macro_centrality
        + 0.25 * evidence_quality
        + 0.20 * claim_specificity
        + 0.15 * questionability
        + 0.10 * graph_connectivity
    )
    return {
        "macro_centrality": round(macro_centrality, 4),
        "evidence_quality": round(evidence_quality, 4),
        "claim_specificity": round(claim_specificity, 4),
        "questionability": round(questionability, 4),
        "graph_connectivity": round(graph_connectivity, 4),
        "final_importance_score": round(final_score, 4),
    }


def _candidate_quality(candidate: dict, macro_by_id: dict[str, dict]) -> float:
    macro = macro_by_id.get(candidate.get("macro_id"), {})
    return _score_candidate(candidate, macro).get("final_importance_score", 0.0)


def _normalize_claim(claim: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", claim.lower()).strip()


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

