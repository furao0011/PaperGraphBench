import os
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.model_client import OpenAICompatClient
from src.prompt_loader import load_prompt, render_prompt


MIN_KC = 12
MAX_KC = 18

_STOPWORDS = {
    "the",
    "and",
    "for",
    "are",
    "that",
    "with",
    "this",
    "from",
    "have",
    "been",
    "into",
    "than",
    "were",
    "their",
    "they",
    "will",
    "which",
    "when",
    "what",
    "where",
    "while",
    "about",
    "paper",
    "method",
    "results",
    "model",
    "using",
    "used",
    "show",
    "shows",
    "based",
}


def _clean_text(text: str) -> str:
    text = re.sub(r"`{1,3}.*?`{1,3}", " ", text, flags=re.DOTALL)
    text = re.sub(r"!\[[^\]]*\]\([^)]+\)", " ", text)
    text = re.sub(r"\[[^\]]+\]\([^)]+\)", " ", text)
    text = re.sub(r"#+\s*", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _split_sentences(text: str) -> list[str]:
    chunks = re.split(r"(?<=[\.\!\?。！？；;])\s+", text)
    sentences = [c.strip() for c in chunks if c.strip()]
    return sentences


def _sentence_score(sentence: str) -> int:
    s = sentence.lower()
    score = 0
    signal_terms = [
        "because",
        "therefore",
        "thus",
        "improve",
        "outperform",
        "achieve",
        "results",
        "compared",
        "however",
        "limitation",
        "ablation",
        "analysis",
        "we propose",
        "we introduce",
        "we show",
        "demonstrate",
        "indicates",
    ]
    for term in signal_terms:
        if term in s:
            score += 2
    length = len(sentence.split())
    if 8 <= length <= 45:
        score += 2
    if 46 <= length <= 70:
        score += 1
    if any(ch.isdigit() for ch in sentence):
        score += 1
    return score


def _normalize_claim(sentence: str) -> str:
    sentence = sentence.strip().strip("-*")
    sentence = re.sub(r"\s+", " ", sentence)
    if not sentence:
        return sentence
    if sentence[-1] not in ".!?。！？":
        sentence += "."
    return sentence[0].upper() + sentence[1:]


def _keyword_fallback_claims(text: str, needed: int) -> list[str]:
    tokens = re.findall(r"[A-Za-z][A-Za-z\-]{2,}", text.lower())
    freq = Counter(t for t in tokens if t not in _STOPWORDS)
    keywords = [k for k, _ in freq.most_common(needed * 2 + 4)]
    claims = []
    for idx, kw in enumerate(keywords[:needed], start=1):
        claims.append(
            f"The paper treats '{kw}' as a relevant concept in its technical narrative (fallback KC {idx})."
        )
    return claims


def extract_kcs(paper_text: str) -> list[dict]:
    cleaned = _clean_text(paper_text)
    sentences = _split_sentences(cleaned)

    # Score candidate claims and keep diverse, de-duplicated statements.
    scored = sorted(
        ((s, _sentence_score(s)) for s in sentences if len(s.split()) >= 6),
        key=lambda x: x[1],
        reverse=True,
    )

    selected_claims: list[str] = []
    seen_norm = set()
    for sentence, score in scored:
        if score <= 0:
            continue
        claim = _normalize_claim(sentence)
        norm = re.sub(r"[^a-z0-9]+", " ", claim.lower()).strip()
        if not norm or norm in seen_norm:
            continue
        seen_norm.add(norm)
        selected_claims.append(claim)
        if len(selected_claims) >= MAX_KC:
            break

    if len(selected_claims) < MIN_KC:
        fallback_needed = MIN_KC - len(selected_claims)
        selected_claims.extend(_keyword_fallback_claims(cleaned, fallback_needed))

    selected_claims = selected_claims[:MAX_KC]

    kcs: list[dict] = []
    evidence_window = cleaned[:800]
    for i, claim in enumerate(selected_claims, start=1):
        evidence = claim if claim in cleaned else evidence_window
        kcs.append(
            {
                "kc_id": f"KC{i}",
                "claim": claim,
                "evidence": evidence,
            }
        )
    return kcs


def extract_kcs_with_online_fallback(paper_text: str, client: OpenAICompatClient | None) -> list[dict]:
    if client and client.is_ready():
        try:
            system_prompt = "You are an accurate information extraction assistant."
            tpl = load_prompt("extract_kc.txt")
            user_prompt = render_prompt(tpl, paper_text=paper_text[:30000])
            result = client.chat_json(system_prompt=system_prompt, user_prompt=user_prompt)
            items = result.get("kcs", [])
            parsed: list[dict] = []
            for idx, item in enumerate(items[:MAX_KC], start=1):
                claim = str(item.get("claim", "")).strip()
                evidence = str(item.get("evidence", "")).strip() or claim
                if not claim:
                    continue
                parsed.append({"kc_id": f"KC{idx}", "claim": claim, "evidence": evidence})
            if len(parsed) >= MIN_KC:
                return parsed
        except Exception:
            # v0: silent fallback to local extractor for robustness.
            pass
    return extract_kcs(paper_text)


def extract_kcs_by_sections_with_online_fallback(
    sections: list[dict],
    client: OpenAICompatClient | None,
) -> list[dict]:
    """
    Guidance Step 2/3:
    - Extract 3-5 candidate KCs per section (parallel online calls)
    - Deduplicate and keep 12-18
    """
    if client and client.is_ready() and sections:
        candidates: list[dict] = []
        tpl = load_prompt("extract_kc.txt")

        def run_one(sec: dict) -> list[dict]:
            user_prompt = render_prompt(tpl, paper_text=sec["text"][:8000])
            result = client.chat_json(
                system_prompt="You extract 3-5 evaluable KCs from one section.",
                user_prompt=user_prompt,
            )
            items = result.get("kcs", [])
            out = []
            for it in items[:5]:
                claim = str(it.get("claim", "")).strip()
                evidence = str(it.get("evidence", "")).strip() or sec["text"][:300]
                if claim:
                    out.append({"claim": claim, "evidence": evidence, "section": sec["title"]})
            return out

        section_limit = int(os.getenv("ONLINE_SECTION_LIMIT", "6"))
        picked_sections = sections[:section_limit]
        max_workers = min(int(os.getenv("ONLINE_KC_WORKERS", "4")), max(1, len(picked_sections)))
        futures = {}
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            for sec in picked_sections:
                futures[ex.submit(run_one, sec)] = sec["section_id"]
            for fut in as_completed(futures):
                try:
                    candidates.extend(fut.result())
                except Exception:
                    continue

        if candidates:
            # Deduplicate by normalized claim and keep highest diversity.
            uniq = []
            seen = set()
            for c in candidates:
                norm = re.sub(r"[^a-z0-9]+", " ", c["claim"].lower()).strip()
                if not norm or norm in seen:
                    continue
                seen.add(norm)
                uniq.append(c)
            selected = uniq[:MAX_KC]
            if len(selected) < MIN_KC:
                joined = "\n".join(sec["text"] for sec in sections)
                fill_claims = _keyword_fallback_claims(joined, MIN_KC - len(selected))
                for fc in fill_claims:
                    selected.append({"claim": fc, "evidence": joined[:300], "section": "fallback"})
            out = []
            for i, c in enumerate(selected[:MAX_KC], start=1):
                out.append({"kc_id": f"KC{i}", "claim": c["claim"], "evidence": c["evidence"]})
            if len(out) >= MIN_KC:
                return out

    # fallback: collapse sections to full text local extractor
    merged = "\n\n".join(s.get("text", "") for s in sections)
    return extract_kcs(merged)
