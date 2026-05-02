import json
import os
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.model_client import OpenAICompatClient
from src.macro_extractor import macro_context_for_prompt
from src.prompt_loader import load_prompt, render_prompt
from src.progress import log, span


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


def extract_kcs_with_online_fallback(
    paper_text: str,
    client: OpenAICompatClient | None,
    macro_spine: dict | None = None,
) -> list[dict]:
    if client and client.is_ready():
        try:
            system_prompt = "You are an accurate information extraction assistant."
            tpl = load_prompt("extract_kc.txt")
            user_prompt = render_prompt(
                tpl,
                paper_text=paper_text[:30000],
                macro_context_json=_macro_context_json(macro_spine),
            )
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
    allow_offline_fallback: bool = False,
    macro_spine: dict | None = None,
) -> list[dict]:
    """
    Guidance Step 2/3:
    - Extract 3-5 candidate KCs per section (parallel online calls)
    - Deduplicate and keep 12-18
    """
    candidates = extract_kc_candidates_by_sections(
        sections,
        client,
        allow_offline_fallback=allow_offline_fallback,
        macro_spine=macro_spine,
    )
    selected = _select_diverse_candidates(candidates, MAX_KC)
    if len(selected) < MIN_KC and not allow_offline_fallback:
        raise RuntimeError(
            f"Online KC extraction returned only {len(selected)} valid KCs; required at least {MIN_KC}."
        )
    out = []
    for i, c in enumerate(selected[:MAX_KC], start=1):
        out.append(
            {
                "kc_id": f"KC{i}",
                "claim": c["claim"],
                "evidence": c["evidence"],
                "section": c.get("section", ""),
                "section_id": c.get("section_id", ""),
                "macro_id": c.get("macro_id", ""),
                "type": c.get("type", ""),
                "importance": c.get("importance", ""),
            }
        )
    return out


def extract_kc_candidates_by_sections(
    sections: list[dict],
    client: OpenAICompatClient | None,
    allow_offline_fallback: bool = False,
    macro_spine: dict | None = None,
) -> list[dict]:
    """
    v1 candidate pool extraction:
    - Extract 3-5 candidate KCs per selected section.
    - Keep all normalized unique candidates up to KC_BANK_MAX.
    - Do not pad missing KCs in strict online mode.
    """
    online_errors: list[str] = []
    if client and client.is_ready() and sections:
        candidates: list[dict] = []
        tpl = load_prompt("extract_kc.txt")
        macro_context_json = _macro_context_json(macro_spine)

        picked_sections = _pick_sections_for_extraction(sections)
        log("KC extraction sections selected", count=len(picked_sections))

        def run_one(sec: dict) -> list[dict]:
            log("KC extraction section queued", section_id=sec.get("section_id"), title=sec.get("title"))
            user_prompt = render_prompt(
                tpl,
                paper_text=sec["text"][:8000],
                macro_context_json=macro_context_json,
            )
            with span("KC extraction section", section_id=sec.get("section_id")):
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
                    out.append(
                        {
                            "claim": claim,
                            "evidence": evidence,
                            "section": sec["title"],
                            "section_id": sec.get("section_id", ""),
                            "section_index": sec.get("_section_index", 0),
                            "macro_id": str(it.get("macro_id", "")).strip(),
                            "type": str(it.get("type", "")).strip(),
                            "importance": str(it.get("importance", "")).strip(),
                        }
                    )
            log("KC extraction section parsed", section_id=sec.get("section_id"), candidates=len(out))
            return out

        max_workers = min(int(os.getenv("ONLINE_KC_WORKERS", "4")), max(1, len(picked_sections)))
        futures = {}
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            for sec in picked_sections:
                futures[ex.submit(run_one, sec)] = sec["section_id"]
            for fut in as_completed(futures):
                try:
                    section_items = fut.result()
                    candidates.extend(section_items)
                    log("KC extraction section completed", section_id=futures[fut], total_candidates=len(candidates))
                except Exception as exc:
                    online_errors.append(f"{futures[fut]}: {type(exc).__name__}: {exc}")
                    log("KC extraction section error", section_id=futures[fut], error=f"{type(exc).__name__}: {exc}")
                    continue

        if candidates:
            candidates.sort(key=lambda c: (c.get("section_index", 0), c.get("claim", "")))
            uniq = []
            seen = set()
            for c in candidates:
                norm = re.sub(r"[^a-z0-9]+", " ", c["claim"].lower()).strip()
                if not norm or norm in seen:
                    continue
                seen.add(norm)
                uniq.append(c)
            bank_limit = int(os.getenv("KC_BANK_MAX", "120") or "120")
            selected = _select_diverse_candidates(uniq, bank_limit)
            log("KC extraction candidates selected", unique=len(uniq), selected=len(selected))
            out = []
            for i, c in enumerate(selected, start=1):
                out.append(
                    {
                        "candidate_id": f"C{i}",
                        "claim": c["claim"],
                        "evidence": c["evidence"],
                        "section": c.get("section", ""),
                        "section_id": c.get("section_id", ""),
                        "macro_id": c.get("macro_id", ""),
                        "type": c.get("type", ""),
                        "importance": c.get("importance", ""),
                    }
                )
            if out:
                return out

    if not allow_offline_fallback:
        detail = "; ".join(online_errors[:3]) if online_errors else "online KC extraction returned too few valid KCs"
        raise RuntimeError(f"Online KC extraction failed and offline fallback is disabled: {detail}")

    # fallback: collapse sections to full text local extractor
    merged = "\n\n".join(s.get("text", "") for s in sections)
    return [
        {
            "candidate_id": f"C{i}",
            "claim": kc["claim"],
            "evidence": kc["evidence"],
            "section": "fallback",
            "section_id": "",
            "section_index": i,
            "macro_id": kc.get("macro_id", ""),
            "type": kc.get("type", ""),
            "importance": kc.get("importance", ""),
        }
        for i, kc in enumerate(extract_kcs(merged), start=1)
    ]


def _select_diverse_candidates(candidates: list[dict], limit: int) -> list[dict]:
    buckets: dict[int, list[dict]] = {}
    for c in candidates:
        buckets.setdefault(int(c.get("section_index", 0)), []).append(c)
    selected: list[dict] = []
    while len(selected) < limit:
        progressed = False
        for section_index in sorted(buckets):
            bucket = buckets[section_index]
            if bucket:
                selected.append(bucket.pop(0))
                progressed = True
                if len(selected) >= limit:
                    break
        if not progressed:
            break
    return selected


def _pick_sections_for_extraction(sections: list[dict]) -> list[dict]:
    section_limit = int(os.getenv("ONLINE_SECTION_LIMIT", "12"))
    indexed = [
        {**sec, "_section_index": idx}
        for idx, sec in enumerate(sections)
        if sec.get("title") != "Preamble" and len(sec.get("text", "").strip()) >= 200
    ]
    if not indexed:
        indexed = [{**sec, "_section_index": idx} for idx, sec in enumerate(sections)]
    if section_limit <= 0 or len(indexed) <= section_limit:
        return indexed

    priority_terms = [
        "abstract",
        "introduction",
        "related",
        "method",
        "framework",
        "dataset",
        "experiment",
        "result",
        "analysis",
        "ablation",
        "case",
        "conclusion",
        "limitation",
    ]
    picked: list[dict] = []
    seen = set()

    def add(sec: dict) -> None:
        idx = sec["_section_index"]
        if idx not in seen and len(picked) < section_limit:
            picked.append(sec)
            seen.add(idx)

    for term in priority_terms:
        for sec in indexed:
            if term in sec.get("title", "").lower():
                add(sec)
                break

    if len(picked) < section_limit:
        step = max(1, len(indexed) // section_limit)
        for idx in range(0, len(indexed), step):
            add(indexed[idx])
            if len(picked) >= section_limit:
                break

    return sorted(picked, key=lambda s: s["_section_index"])


def _macro_context_json(macro_spine: dict | None) -> str:
    if not macro_spine:
        return "[]"
    return json.dumps(macro_context_for_prompt(macro_spine), ensure_ascii=False, indent=2)
