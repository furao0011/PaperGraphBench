from __future__ import annotations

import math
import os
import re


STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "into", "onto", "are", "was", "were",
    "has", "have", "had", "its", "their", "there", "which", "when", "where", "also", "than",
}


def retrieve_kc_and_evidence(
    claim: dict,
    kc_bank: dict,
    top_k_kc: int | None = None,
    top_k_evidence: int | None = None,
) -> dict:
    top_k_kc = top_k_kc if top_k_kc is not None else _env_int("CLAIM_RETRIEVE_TOP_KC", 5)
    top_k_evidence = top_k_evidence if top_k_evidence is not None else _env_int("CLAIM_RETRIEVE_TOP_EVIDENCE", 5)
    claim_tokens = _tokens(claim.get("claim_text", ""))
    scored = []
    for kc in kc_bank.get("kc_nodes", []):
        kc_text = " ".join(
            [
                str(kc.get("short_label", "")),
                str(kc.get("full_claim", "")),
                " ".join(str(e.get("text", "")) for e in kc.get("evidence", []) if isinstance(e, dict)),
            ]
        )
        score = _weighted_overlap(claim_tokens, _tokens(kc_text))
        if score > 0:
            scored.append((score, kc))
    scored.sort(key=lambda item: (-item[0], item[1].get("kc_id", "")))
    retrieved_kcs = [
        {
            "kc_id": kc.get("kc_id"),
            "macro_id": kc.get("macro_id"),
            "full_claim": kc.get("full_claim"),
            "score": round(score, 4),
            "evidence": kc.get("evidence", [])[:top_k_evidence],
        }
        for score, kc in scored[:top_k_kc]
    ]
    evidence_spans = []
    for item in retrieved_kcs:
        for evidence in item.get("evidence", []):
            if not isinstance(evidence, dict):
                continue
            evidence_spans.append(
                {
                    "kc_id": item.get("kc_id"),
                    "section": evidence.get("section"),
                    "span_id": evidence.get("span_id"),
                    "text": evidence.get("text"),
                }
            )
            if len(evidence_spans) >= top_k_evidence:
                break
        if len(evidence_spans) >= top_k_evidence:
            break
    return {
        "claim_id": claim.get("claim_id"),
        "retrieved_kcs": retrieved_kcs,
        "retrieved_evidence": evidence_spans,
    }


def _weighted_overlap(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    overlap = left & right
    if not overlap:
        return 0.0
    precision = len(overlap) / len(left)
    recall = len(overlap) / len(right)
    if precision + recall == 0:
        return 0.0
    return (2 * precision * recall / (precision + recall)) * math.log(1 + len(overlap))


def _tokens(text: str) -> set[str]:
    return {
        tok
        for tok in re.findall(r"[a-zA-Z0-9_]{3,}", text.lower())
        if tok not in STOPWORDS
    }


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    try:
        value = int(raw) if raw is not None and raw.strip() else default
    except ValueError:
        return default
    return max(1, value)
