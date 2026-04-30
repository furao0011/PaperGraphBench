from __future__ import annotations

import re

from src.model_client import OpenAICompatClient
from src.prompt_loader import load_prompt, render_prompt


def build_kc_rubric(
    kc_id: str,
    full_claim: str,
    evidence_text: str,
    kc_type: str,
    importance: str,
    client: OpenAICompatClient | None = None,
    allow_offline_fallback: bool = False,
) -> dict:
    online = _build_kc_rubric_online(kc_id, full_claim, evidence_text, importance, client)
    if online:
        online["type"] = kc_type
        return online
    if not allow_offline_fallback:
        raise RuntimeError(f"Online rubric generation failed for {kc_id} and offline fallback is disabled.")

    key_terms = _extract_key_terms(full_claim)
    must_include = key_terms[:2] if len(key_terms) >= 2 else key_terms
    if not must_include:
        must_include = [full_claim.split(".")[0][:60]]

    forbidden = []
    for idx, term in enumerate(must_include[:2], start=1):
        forbidden.append(
            {
                "claim_id": f"FC_{kc_id}_{idx}",
                "claim": f"The paper rejects or reverses '{term}'.",
                "type": "logic_hallucination",
                "severity": "high",
                "why_wrong": "This reverses the direction of the original claim.",
                "followup_hint": "Please restate this point and align it with the paper evidence.",
            }
        )

    return {
        "must_include": must_include,
        "acceptable_variants": [full_claim],
        "forbidden_claims": forbidden,
        "evidence": [{"section": "auto", "span_id": kc_id, "text": evidence_text}],
        "importance": importance if importance in {"critical", "normal"} else "normal",
        "type": kc_type,
    }


def _build_kc_rubric_online(
    kc_id: str,
    full_claim: str,
    evidence_text: str,
    importance: str,
    client: OpenAICompatClient | None,
) -> dict | None:
    if not client or not client.is_ready():
        return None
    try:
        tpl = load_prompt("generate_rubric.txt")
        user_prompt = render_prompt(tpl, kc_claim=full_claim, kc_evidence=evidence_text[:2000])
        result = client.chat_json(
            system_prompt="You are a strict rubric constructor for paper-evaluation KCs.",
            user_prompt=user_prompt,
        )
        must = result.get("must_include", [])
        variants = result.get("acceptable_variants", [])
        forbidden = result.get("forbidden_claims", [])
        imp = result.get("importance", importance)
        if not must or not isinstance(must, list):
            return None
        return {
            "must_include": must[:4],
            "acceptable_variants": variants[:4] if isinstance(variants, list) else [full_claim],
            "forbidden_claims": forbidden[:4] if isinstance(forbidden, list) else [],
            "evidence": [{"section": "auto", "span_id": kc_id, "text": evidence_text}],
            "importance": imp if imp in {"critical", "normal"} else importance,
        }
    except Exception:
        return None


def _extract_key_terms(text: str) -> list[str]:
    toks = re.findall(r"[A-Za-z][A-Za-z\-]{3,}", text)
    stop = {"that", "with", "from", "this", "these", "those", "using", "which", "their", "paper"}
    uniq = []
    seen = set()
    for t in toks:
        l = t.lower()
        if l in stop or l in seen:
            continue
        seen.add(l)
        uniq.append(t)
    return uniq[:6]
