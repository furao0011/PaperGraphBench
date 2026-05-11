from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.model_client import OpenAICompatClient
from src.progress import log, span
from src.prompt_loader import load_prompt, render_prompt


ALLOWED_EDGE_RELATIONS = {
    "part_of",
    "defines",
    "motivates",
    "solves",
    "explains",
    "supports",
    "contrasts_with",
    "limits",
}

VALID_EDGE_DECISIONS = {
    "valid",
    "invalid",
    "insufficient_context",
    "wrong_direction",
    "relation_should_be_other",
}


def verify_edge_candidates(
    paper_id: str,
    edge_candidates: list[dict],
    kc_bank: dict,
    extraction_units: dict,
    client: OpenAICompatClient,
) -> dict:
    if not client or not client.is_ready():
        raise RuntimeError("Edge verification requires a configured online model client.")

    if not edge_candidates:
        return {
            "paper_id": paper_id,
            "verified_edges": [],
            "verification_log": [],
            "summary": {
                "candidate_count": 0,
                "verified_count": 0,
                "rejected_count": 0,
            },
        }

    by_kc = _kc_by_id(kc_bank)
    by_unit = _unit_by_id(extraction_units)
    tpl = load_prompt("verify_edge_candidate.txt")
    max_workers = min(_env_positive_int("EDGE_VERIFY_WORKERS", 3), len(edge_candidates))
    logs_by_candidate: dict[str, dict] = {}
    errors: list[str] = []

    def run_one(candidate: dict) -> tuple[str, dict]:
        candidate_edge_id = str(candidate.get("candidate_edge_id", "")).strip()
        if not candidate_edge_id:
            raise ValueError("Every edge candidate must contain candidate_edge_id.")
        context_packet = _context_packet(candidate, by_kc, by_unit)
        with span("verify edge candidate", candidate_edge_id=candidate_edge_id):
            result = client.chat_json(
                system_prompt="You verify reasoning edges for a paper evaluation graph. Return JSON only.",
                user_prompt=render_prompt(
                    tpl,
                    context_packet_json=json.dumps(context_packet, ensure_ascii=False, indent=2),
                ),
                temperature=0.1,
            )
        return candidate_edge_id, _normalize_verification_result(candidate, result)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(run_one, candidate): candidate for candidate in edge_candidates}
        for fut in as_completed(futures):
            candidate = futures[fut]
            candidate_edge_id = str(candidate.get("candidate_edge_id", "")).strip()
            try:
                out_id, log_item = fut.result()
                logs_by_candidate[out_id] = log_item
                log(
                    "edge candidate verified",
                    candidate_edge_id=out_id,
                    decision=log_item.get("decision"),
                )
            except Exception as exc:
                errors.append(f"{candidate_edge_id}: {type(exc).__name__}: {exc}")
                log("edge verification error", candidate_edge_id=candidate_edge_id, error=f"{type(exc).__name__}: {exc}")

    if errors:
        raise RuntimeError("Edge verification failed: " + "; ".join(errors[:5]))

    verification_log = [
        logs_by_candidate[candidate["candidate_edge_id"]]
        for candidate in edge_candidates
        if candidate["candidate_edge_id"] in logs_by_candidate
    ]
    verified_edges = _verified_edges_from_log(verification_log)
    return {
        "paper_id": paper_id,
        "verified_edges": verified_edges,
        "verification_log": verification_log,
        "summary": {
            "candidate_count": len(edge_candidates),
            "verified_count": len(verified_edges),
            "rejected_count": len(edge_candidates) - len(verified_edges),
        },
    }


def _normalize_verification_result(candidate: dict, result: dict) -> dict:
    decision = str(result.get("decision", "")).strip()
    if decision not in VALID_EDGE_DECISIONS:
        raise ValueError(
            f"Verifier returned invalid decision for {candidate.get('candidate_edge_id')}: {decision!r}"
        )
    suggested_relation = result.get("suggested_relation")
    if suggested_relation is not None:
        suggested_relation = str(suggested_relation).strip() or None
    if decision == "relation_should_be_other":
        if suggested_relation not in ALLOWED_EDGE_RELATIONS:
            raise ValueError(
                f"Verifier suggested invalid relation for {candidate.get('candidate_edge_id')}: {suggested_relation!r}"
            )
    elif suggested_relation is not None:
        suggested_relation = None
    confidence = _confidence(result.get("confidence", 0.0))
    accepted = decision in {"valid", "relation_should_be_other"}
    verified_relation = suggested_relation if decision == "relation_should_be_other" else candidate.get("relation")
    return {
        "candidate_edge_id": candidate.get("candidate_edge_id"),
        "source_layer": candidate.get("source_layer"),
        "scope": candidate.get("scope"),
        "unit_id": candidate.get("unit_id"),
        "macro_id": candidate.get("macro_id"),
        "macro_edge_id": candidate.get("macro_edge_id"),
        "source_macro_id": candidate.get("source_macro_id"),
        "target_macro_id": candidate.get("target_macro_id"),
        "thread_pattern": candidate.get("thread_pattern"),
        "source": candidate.get("source"),
        "target": candidate.get("target"),
        "candidate_relation": candidate.get("relation"),
        "decision": decision,
        "accepted": accepted,
        "verified_relation": verified_relation if accepted else None,
        "suggested_relation": suggested_relation,
        "confidence": confidence,
        "reason": str(result.get("reason", "")).strip(),
        "candidate": candidate,
    }


def _verified_edges_from_log(verification_log: list[dict]) -> list[dict]:
    verified = []
    for item in verification_log:
        if not item.get("accepted"):
            continue
        candidate = item["candidate"]
        edge_id = f"E{len(verified) + 1}"
        verified.append(
            {
                "edge_id": edge_id,
                "source_candidate_edge_id": candidate.get("candidate_edge_id"),
                "source_layer": candidate.get("source_layer"),
                "scope": candidate.get("scope"),
                "unit_id": candidate.get("unit_id"),
                "macro_id": candidate.get("macro_id"),
                "macro_edge_id": candidate.get("macro_edge_id"),
                "source_macro_id": candidate.get("source_macro_id"),
                "target_macro_id": candidate.get("target_macro_id"),
                "thread_pattern": candidate.get("thread_pattern"),
        "source_unit_id": candidate.get("source_unit_id"),
        "target_unit_id": candidate.get("target_unit_id"),
        "evidence_unit_id": candidate.get("evidence_unit_id"),
        "evidence_paragraph_ids": candidate.get("evidence_paragraph_ids", []),
        "source": candidate.get("source"),
                "target": candidate.get("target"),
                "relation": item.get("verified_relation"),
                "description": candidate.get("reason", ""),
                "evidence": candidate.get("evidence", ""),
                "confidence": item.get("confidence", 0.0),
                "verifier_reason": item.get("reason", ""),
                "forbidden_claims": [],
            }
        )
    return verified


def _context_packet(candidate: dict, by_kc: dict[str, dict], by_unit: dict[str, dict]) -> dict:
    source_id = str(candidate.get("source", "")).strip()
    target_id = str(candidate.get("target", "")).strip()
    unit_id = str(candidate.get("unit_id", "")).strip()
    source_unit_id = str(candidate.get("source_unit_id") or unit_id).strip()
    target_unit_id = str(candidate.get("target_unit_id") or unit_id).strip()
    source_kc = by_kc.get(source_id)
    target_kc = by_kc.get(target_id)
    unit = by_unit.get(unit_id) if unit_id else None
    source_unit = by_unit.get(source_unit_id)
    target_unit = by_unit.get(target_unit_id)
    if not source_kc or not target_kc:
        raise ValueError(f"Candidate {candidate.get('candidate_edge_id')} references unknown KC IDs.")
    if candidate.get("scope") == "unit" and not unit:
        raise ValueError(f"Candidate {candidate.get('candidate_edge_id')} references unknown unit_id={unit_id!r}.")
    if candidate.get("scope") in {"macro", "adjacent_macro", "thread"} and (not source_unit or not target_unit):
        raise ValueError(
            f"Candidate {candidate.get('candidate_edge_id')} references unknown source/target Unit IDs."
        )
    return {
        "scope": candidate.get("scope"),
        "candidate_edge": {
            "candidate_edge_id": candidate.get("candidate_edge_id"),
            "source": source_id,
            "target": target_id,
            "relation": candidate.get("relation"),
            "evidence": candidate.get("evidence"),
            "reason": candidate.get("reason"),
            "macro_id": candidate.get("macro_id"),
            "macro_edge_id": candidate.get("macro_edge_id"),
            "source_macro_id": candidate.get("source_macro_id"),
            "target_macro_id": candidate.get("target_macro_id"),
            "thread_pattern": candidate.get("thread_pattern"),
            "macro_title": candidate.get("macro_title"),
            "macro_role": candidate.get("macro_role"),
            "macro_summary": candidate.get("macro_summary"),
            "unit_id": unit_id,
            "source_unit_id": source_unit_id,
            "target_unit_id": target_unit_id,
            "evidence_unit_id": candidate.get("evidence_unit_id"),
            "evidence_paragraph_ids": candidate.get("evidence_paragraph_ids", []),
        },
        "source_kc": _kc_packet(source_kc),
        "target_kc": _kc_packet(target_kc),
        "available_context": _available_context(
            scope=str(candidate.get("scope", "")),
            unit=unit,
            source_unit=source_unit,
            target_unit=target_unit,
        ),
    }


def _kc_packet(kc: dict) -> dict:
    return {
        "kc_id": kc.get("kc_id"),
        "claim": kc.get("full_claim") or kc.get("claim"),
        "evidence": _edge_context_evidence_text(kc),
        "unit_id": kc.get("unit_id"),
        "macro_id": kc.get("macro_id"),
        "claim_strength": kc.get("claim_strength"),
        "scope": kc.get("scope"),
        "modality": kc.get("modality", {"is_multimodal": False}),
        "asset_id": kc.get("asset_id"),
        "asset_type": kc.get("asset_type"),
        "asset_caption": kc.get("asset_caption"),
        "asset_summary": kc.get("asset_summary"),
    }


def _edge_context_evidence_text(kc: dict) -> str:
    if bool(kc.get("modality", {}).get("is_multimodal")):
        evidence_basis = str(kc.get("asset_evidence_basis") or "").strip()
        if evidence_basis:
            return evidence_basis
    return str(kc.get("evidence_text") or "").strip()


def _available_context(
    scope: str,
    unit: dict | None,
    source_unit: dict | None,
    target_unit: dict | None,
) -> dict:
    if scope in {"macro", "adjacent_macro", "thread"}:
        return {
            "source_unit": _unit_packet(source_unit),
            "target_unit": _unit_packet(target_unit),
        }
    return {
        "unit_id": unit.get("unit_id") if unit else None,
        "unit_title": unit.get("unit_title") if unit else None,
        "unit_summary": unit.get("unit_summary") if unit else None,
        "unit_source_text": unit.get("source_text") if unit else None,
    }


def _unit_packet(unit: dict | None) -> dict | None:
    if not unit:
        return None
    return {
        "unit_id": unit.get("unit_id"),
        "unit_title": unit.get("unit_title"),
        "unit_summary": unit.get("unit_summary"),
        "unit_source_text": unit.get("source_text"),
    }


def _kc_by_id(kc_bank: dict) -> dict[str, dict]:
    return {
        str(kc.get("kc_id", "")).strip(): kc
        for kc in kc_bank.get("kc_nodes", [])
        if str(kc.get("kc_id", "")).strip()
    }


def _unit_by_id(extraction_units: dict) -> dict[str, dict]:
    return {
        str(unit.get("unit_id", "")).strip(): unit
        for unit in extraction_units.get("units", [])
        if str(unit.get("unit_id", "")).strip()
    }


def _confidence(value: object) -> float:
    try:
        raw = float(value)
    except (TypeError, ValueError):
        raw = 0.0
    return round(max(0.0, min(1.0, raw)), 4)


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
