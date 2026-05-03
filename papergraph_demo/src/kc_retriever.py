from __future__ import annotations

import hashlib
import math
import os
from typing import Any

from src.model_client import OpenAICompatClient


RETRIEVAL_METHOD = "embedding_cosine"
_INDEX_CACHE: dict[tuple[str, str, str], dict[str, Any]] = {}


def retrieve_kc_and_evidence(
    claim: dict,
    kc_bank: dict,
    client: OpenAICompatClient,
    top_k_kc: int | None = None,
    top_k_evidence: int | None = None,
) -> dict:
    if not client or not client.embeddings_ready():
        raise RuntimeError(
            "Claim Verification retrieval requires EMBED_API_KEY, EMBED_BASE_URL, and EMBED_MODEL."
        )

    top_k_kc = top_k_kc if top_k_kc is not None else _env_positive_int("CLAIM_RETRIEVE_TOP_KC", 5)
    top_k_evidence = (
        top_k_evidence
        if top_k_evidence is not None
        else _env_positive_int("CLAIM_RETRIEVE_TOP_EVIDENCE", 5)
    )
    claim_text = str(claim.get("claim_text", "")).strip()
    if not claim_text:
        raise ValueError("Claim object must contain a non-empty claim_text field.")

    index = _get_or_build_index(kc_bank, client)
    claim_vector = client.embed_texts([claim_text])[0]

    kc_scores = [
        (_cosine_similarity(claim_vector, entry["embedding"]), entry)
        for entry in index["kc_entries"]
    ]
    kc_scores.sort(key=lambda item: (-item[0], item[1]["kc_id"]))
    selected_kc_scores = kc_scores[:top_k_kc]
    selected_kc_ids = {entry["kc_id"] for _, entry in selected_kc_scores}

    evidence_scores = [
        (_cosine_similarity(claim_vector, entry["embedding"]), entry)
        for entry in index["evidence_entries"]
        if entry["kc_id"] in selected_kc_ids
    ]
    evidence_scores = sorted(
        evidence_scores,
        key=lambda item: (-item[0], item[1]["kc_id"], str(item[1].get("span_id", ""))),
    )
    top_evidence_scores = evidence_scores[:top_k_evidence]

    retrieved_kcs = []
    for score, entry in selected_kc_scores:
        kc_evidence = [
            _public_evidence(ev_score, ev_entry)
            for ev_score, ev_entry in evidence_scores
            if ev_entry["kc_id"] == entry["kc_id"]
        ][:top_k_evidence]
        retrieved_kcs.append(
            {
                "kc_id": entry["kc_id"],
                "macro_id": entry["kc"].get("macro_id"),
                "full_claim": entry["kc"].get("full_claim"),
                "score": round(score, 4),
                "embedding_similarity": round(score, 4),
                "retrieval_method": RETRIEVAL_METHOD,
                "evidence": kc_evidence,
            }
        )

    return {
        "claim_id": claim.get("claim_id"),
        "retrieval_method": RETRIEVAL_METHOD,
        "embedding_model": client.cfg.embed_model,
        "retrieved_kcs": retrieved_kcs,
        "retrieved_evidence": [
            _public_evidence(score, entry)
            for score, entry in top_evidence_scores
        ],
    }


def _get_or_build_index(kc_bank: dict, client: OpenAICompatClient) -> dict[str, Any]:
    paper_id = str(kc_bank.get("paper_id", "unknown_paper"))
    signature = _kc_bank_signature(kc_bank)
    cache_key = (paper_id, client.cfg.embed_model, signature)
    cached = _INDEX_CACHE.get(cache_key)
    if cached is not None:
        return cached
    index = _build_index(kc_bank, client)
    _INDEX_CACHE[cache_key] = index
    return index


def _build_index(kc_bank: dict, client: OpenAICompatClient) -> dict[str, Any]:
    kc_nodes = kc_bank.get("kc_nodes")
    if not isinstance(kc_nodes, list) or not kc_nodes:
        raise ValueError("KC bank must contain a non-empty kc_nodes list for embedding retrieval.")

    kc_entries = []
    evidence_entries = []
    for kc in kc_nodes:
        if not isinstance(kc, dict):
            raise ValueError("KC bank contains a non-object KC entry.")
        kc_id = str(kc.get("kc_id", "")).strip()
        if not kc_id:
            raise ValueError("Every KC entry must contain a non-empty kc_id.")
        kc_text = _kc_embedding_text(kc)
        if not kc_text:
            raise ValueError(f"KC {kc_id} does not contain retrievable claim text.")
        kc_entries.append({"kc_id": kc_id, "kc": kc, "text": kc_text})
        evidence_entries.extend(_evidence_entries(kc))

    texts = [entry["text"] for entry in kc_entries] + [
        entry["embedding_text"] for entry in evidence_entries
    ]
    embeddings = client.embed_texts(texts)
    if len(embeddings) != len(texts):
        raise RuntimeError(f"Embedding count mismatch: expected {len(texts)}, got {len(embeddings)}.")

    split = len(kc_entries)
    for entry, embedding in zip(kc_entries, embeddings[:split]):
        entry["embedding"] = _validate_embedding(embedding, entry["kc_id"])
    for entry, embedding in zip(evidence_entries, embeddings[split:]):
        entry["embedding"] = _validate_embedding(embedding, f"{entry['kc_id']}:{entry.get('span_id')}")

    return {
        "retrieval_method": RETRIEVAL_METHOD,
        "embedding_model": client.cfg.embed_model,
        "kc_entries": kc_entries,
        "evidence_entries": evidence_entries,
    }


def _kc_embedding_text(kc: dict) -> str:
    parts = [
        str(kc.get("short_label", "")).strip(),
        str(kc.get("claim", "")).strip(),
        str(kc.get("full_claim", "")).strip(),
        str(kc.get("evidence_text", "")).strip(),
    ]
    for evidence in _raw_evidence_list(kc):
        parts.append(str(evidence.get("text", "")).strip())
    return "\n".join(_unique_nonempty(parts))


def _evidence_entries(kc: dict) -> list[dict[str, Any]]:
    kc_id = str(kc.get("kc_id", "")).strip()
    claim = str(kc.get("full_claim") or kc.get("claim") or "").strip()
    entries = []
    raw_evidence = _raw_evidence_list(kc)
    if not raw_evidence and str(kc.get("evidence_text", "")).strip():
        raw_evidence = [
            {
                "section": kc.get("source_section"),
                "span_id": f"{kc_id}:evidence_text",
                "text": str(kc.get("evidence_text", "")).strip(),
            }
        ]

    for idx, evidence in enumerate(raw_evidence, start=1):
        text = str(evidence.get("text", "")).strip()
        if not text:
            raise ValueError(f"KC {kc_id} evidence span #{idx} must contain non-empty text.")
        span_id = str(evidence.get("span_id") or f"{kc_id}:evidence:{idx}").strip()
        section = evidence.get("section") or kc.get("source_section")
        embed_text = "\n".join(
            _unique_nonempty(
                [
                    f"KC claim: {claim}" if claim else "",
                    f"Evidence: {text}",
                ]
            )
        )
        entries.append(
            {
                "kc_id": kc_id,
                "macro_id": kc.get("macro_id"),
                "section": section,
                "span_id": span_id,
                "embedding_text": embed_text,
                "text": text,
            }
        )
    return entries


def _raw_evidence_list(kc: dict) -> list[dict]:
    raw = kc.get("evidence", [])
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError(f"KC {kc.get('kc_id')} evidence field must be a list.")
    for evidence in raw:
        if not isinstance(evidence, dict):
            raise ValueError(f"KC {kc.get('kc_id')} evidence list contains a non-object item.")
    return raw


def _public_evidence(score: float, entry: dict[str, Any]) -> dict:
    return {
        "kc_id": entry.get("kc_id"),
        "macro_id": entry.get("macro_id"),
        "section": entry.get("section"),
        "span_id": entry.get("span_id"),
        "text": entry.get("text"),
        "score": round(score, 4),
        "embedding_similarity": round(score, 4),
        "retrieval_method": RETRIEVAL_METHOD,
    }


def _kc_bank_signature(kc_bank: dict) -> str:
    digest = hashlib.sha256()
    digest.update(str(kc_bank.get("paper_id", "")).encode("utf-8"))
    for kc in kc_bank.get("kc_nodes", []):
        if not isinstance(kc, dict):
            continue
        digest.update(str(kc.get("kc_id", "")).encode("utf-8"))
        digest.update(str(kc.get("macro_id", "")).encode("utf-8"))
        digest.update(str(kc.get("full_claim", "")).encode("utf-8"))
        digest.update(str(kc.get("evidence_text", "")).encode("utf-8"))
        evidence_items = kc.get("evidence", [])
        if not isinstance(evidence_items, list):
            continue
        for evidence in evidence_items:
            if isinstance(evidence, dict):
                digest.update(str(evidence.get("span_id", "")).encode("utf-8"))
                digest.update(str(evidence.get("text", "")).encode("utf-8"))
    return digest.hexdigest()


def _validate_embedding(embedding: Any, source_id: str) -> list[float]:
    if not isinstance(embedding, list) or not embedding:
        raise RuntimeError(f"Embedding for {source_id} is empty or invalid.")
    try:
        vector = [float(value) for value in embedding]
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"Embedding for {source_id} contains non-numeric values.") from exc
    if not any(value != 0.0 for value in vector):
        raise RuntimeError(f"Embedding for {source_id} is a zero vector.")
    return vector


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError(f"Embedding dimensions do not match: {len(left)} vs {len(right)}.")
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        raise ValueError("Cannot compute cosine similarity for a zero vector.")
    return dot / (left_norm * right_norm)


def _unique_nonempty(parts: list[str]) -> list[str]:
    seen = set()
    values = []
    for part in parts:
        value = part.strip()
        if not value or value in seen:
            continue
        seen.add(value)
        values.append(value)
    return values


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
