from __future__ import annotations

import json
import os

from src.model_client import OpenAICompatClient
from src.prompt_loader import load_prompt, render_prompt


VERIFY_ATTRIBUTION_TYPES = {"paper_claim", "paper_inference", "dialogue_reference"}


def extract_atomic_claims(turn: dict, client: OpenAICompatClient) -> list[dict]:
    if not client or not client.is_ready():
        raise RuntimeError("Claim extraction requires a configured online model client.")
    max_claims = _env_int("CLAIM_VERIFY_MAX_CLAIMS_PER_TURN", 8)
    tpl = load_prompt("extract_atomic_claims.txt")
    result = client.chat_json(
        system_prompt="You extract atomic claims from model answers for paper-grounded verification. Return JSON only.",
        user_prompt=render_prompt(
            tpl,
            turn_id=str(turn.get("turn_id", "")),
            question_id=str(turn.get("question_id", "")),
            question_text=str(turn.get("question_text", "")),
            answer=str(turn.get("model_answer", "")),
            max_claims=str(max_claims),
        ),
        temperature=0.1,
    )
    claims = result.get("claims", [])
    if not isinstance(claims, list):
        raise RuntimeError("Atomic claim extraction returned invalid schema: claims must be a list.")
    normalized = []
    for idx, claim in enumerate(claims[:max_claims], start=1):
        if not isinstance(claim, dict):
            continue
        text = str(claim.get("claim_text", "")).strip()
        if not text:
            continue
        attribution = str(claim.get("attribution_type", "")).strip()
        normalized.append(
            {
                "claim_id": str(claim.get("claim_id") or f"CL_{turn.get('turn_id', 'T')}_{idx}"),
                "turn_id": str(turn.get("turn_id", "")),
                "claim_text": text,
                "attribution_type": attribution,
                "related_question_id": str(turn.get("question_id", "")),
                "source_sentence": str(claim.get("source_sentence", text)).strip(),
            }
        )
    return normalized


def verifiable_claims(claims: list[dict]) -> list[dict]:
    return [c for c in claims if c.get("attribution_type") in VERIFY_ATTRIBUTION_TYPES]


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    try:
        value = int(raw) if raw is not None and raw.strip() else default
    except ValueError:
        return default
    return max(0, value)
