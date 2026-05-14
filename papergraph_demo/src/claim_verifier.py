from __future__ import annotations

import json
import os
from pathlib import Path

from src.claim_extractor import extract_atomic_claims, verifiable_claims
from src.eval_artifacts import write_json
from src.kc_retriever import retrieve_kc_and_evidence
from src.model_client import OpenAICompatClient
from src.prompt_loader import load_prompt, render_prompt


SUPPORTED_LABELS = {
    "SUPPORTED",
    "CONTRADICTED",
    "OVERCLAIM",
    "NOT_ENOUGH_INFO",
    "NOT_IN_KC_BUT_SUPPORTED_BY_EVIDENCE",
    "IRRELEVANT_EXTRA",
}


def claim_verification_enabled() -> bool:
    return _env_bool("CLAIM_VERIFY_ENABLED", False)


def verify_global_claims(turn: dict, kc_bank: dict, client: OpenAICompatClient) -> list[dict]:
    if not claim_verification_enabled():
        return []
    if not client or not client.is_ready():
        raise RuntimeError("Global claim verification requires a configured online model client.")
    if not client.embeddings_ready():
        raise RuntimeError(
            "Global claim verification requires a configured embedding client: "
            "set EMBED_API_KEY, EMBED_BASE_URL, and EMBED_MODEL."
        )
    extracted = extract_atomic_claims(turn, client)
    claims = verifiable_claims(extracted)
    results = []
    for claim in claims:
        retrieval = retrieve_kc_and_evidence(claim, kc_bank, client)
        results.append(_verify_one_claim(claim, retrieval, client))
    return results


def append_claim_verification_log(path: Path, results: list[dict]) -> None:
    if not results:
        return
    existing = []
    if path.exists():
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(loaded, list):
            existing = loaded
    write_json(path, existing + results)


def summarize_claim_verifications(results: list[dict]) -> dict:
    labels: dict[str, int] = {}
    for item in results:
        label = str(item.get("label", "NOT_ENOUGH_INFO"))
        labels[label] = labels.get(label, 0) + 1
    return {
        "verified_claim_count": len(results),
        "supported": labels.get("SUPPORTED", 0) + labels.get("NOT_IN_KC_BUT_SUPPORTED_BY_EVIDENCE", 0),
        "contradicted": labels.get("CONTRADICTED", 0),
        "overclaim": labels.get("OVERCLAIM", 0),
        "not_enough_info": labels.get("NOT_ENOUGH_INFO", 0),
        "labels": labels,
    }


def _verify_one_claim(claim: dict, retrieval: dict, client: OpenAICompatClient) -> dict:
    tpl = load_prompt("verify_global_claim.txt")
    result = client.chat_json(
        system_prompt="You verify whether an answer claim is supported by the paper evidence. Return JSON only.",
        user_prompt=render_prompt(
            tpl,
            claim_json=json.dumps(claim, ensure_ascii=False),
            retrieval_json=json.dumps(retrieval, ensure_ascii=False),
        ),
        temperature=0.1,
    )
    label = str(result.get("label", "NOT_ENOUGH_INFO")).strip()
    if label not in SUPPORTED_LABELS:
        raise RuntimeError(f"Global claim verifier returned unsupported label: {label}")
    retrieved_kc_ids = [kc.get("kc_id") for kc in retrieval.get("retrieved_kcs", []) if kc.get("kc_id")]
    supporting_kc_ids = retrieved_kc_ids if label in {"SUPPORTED", "NOT_IN_KC_BUT_SUPPORTED_BY_EVIDENCE"} else []
    contradicted_kc_ids = retrieved_kc_ids if label == "CONTRADICTED" else []
    return {
        "claim_id": claim.get("claim_id"),
        "turn_id": claim.get("turn_id"),
        "claim": claim.get("claim_text"),
        "claim_text": claim.get("claim_text"),
        "attribution_type": claim.get("attribution_type"),
        "retrieved_kc_ids": retrieved_kc_ids,
        "matched_kc_ids": retrieved_kc_ids,
        "supporting_kc_ids": supporting_kc_ids,
        "contradicted_kc_ids": contradicted_kc_ids,
        "retrieved_span_ids": [
            span.get("span_id")
            for span in retrieval.get("retrieved_evidence", [])
            if span.get("span_id")
        ],
        "label": label,
        "confidence": float(result.get("confidence", 0.0) or 0.0),
        "verifier_explanation": str(result.get("verifier_explanation", "")).strip(),
    }


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}
