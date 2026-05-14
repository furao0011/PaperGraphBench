from __future__ import annotations

import json
import re

from src.model_client import OpenAICompatClient
from src.prompt_loader import load_prompt, render_prompt


JUDGE_PROMPTS = {
    "main": "judge_main_answer.txt",
    "macro_main_question": "judge_main_answer.txt",
    "detail_followup": "judge_main_answer.txt",
    "review_followup": "judge_review_answer.txt",
    "hallucination_followup": "judge_hallucination_answer.txt",
    "multi_hop_reasoning": "judge_multi_hop_answer.txt",
    "thread_premise_question": "judge_thread_answer.txt",
    "thread_evidence_question": "judge_thread_answer.txt",
    "thread_bridge_question": "judge_thread_answer.txt",
    "thread_review_question": "judge_thread_answer.txt",
    "thread_question": "judge_thread_answer.txt",
    "challenge_question": "judge_challenge_eval_answer.txt",
    "thread_challenge_question": "judge_thread_challenge_answer.txt",
}

def judge_answer(question_text: str, answer: str, target_kcs: list[dict]) -> dict:
    covered = []
    missing = []
    matched_forbidden = []
    hallucinated = []

    answer_l = answer.lower()
    for kc in target_kcs:
        must = kc.get("must_include", [])
        claim = kc.get("full_claim", "")
        variants = kc.get("acceptable_variants", [])
        covered_by_must = all(_contains_semantic(answer_l, m) for m in must[:2]) if must else False
        covered_by_claim = _overlap_ratio(answer_l, claim) >= 0.35
        covered_by_variant = any(_overlap_ratio(answer_l, v) >= 0.35 for v in variants[:2])
        if covered_by_must or covered_by_claim or covered_by_variant:
            covered.append(kc["kc_id"])
        else:
            missing.append(kc["kc_id"])
        for fc in kc.get("forbidden_claims", []):
            if _forbidden_hit(answer_l, fc.get("claim", "").lower()):
                matched_forbidden.append(fc.get("claim_id"))
                hallucinated.append(fc.get("claim"))

    if hallucinated:
        state = "HALLUCINATION"
        next_action = "hallucination_followup"
    elif missing:
        state = "INCOMPLETE"
        next_action = "detail_followup"
    else:
        state = "MAIN_PROGRESS"
        next_action = "next_main_question"

    return {
        "state": state,
        "covered_kc_ids": covered,
        "missing_kc_ids": missing,
        "hallucinated_claims": hallucinated,
        "matched_forbidden_claims": matched_forbidden,
        "mentioned_unexplained_kc_ids": [],
        "hallucination_type": "logic_hallucination" if hallucinated else None,
        "reasoning_path_result": None,
        "next_action": next_action,
        "confidence": 0.8 if not hallucinated else 0.9,
        "judge_explanation": f"covered={len(covered)} missing={len(missing)} hallucination={len(hallucinated)}",
    }


def judge_answer_with_online_fallback(
    question_text: str,
    answer: str,
    target_kcs: list[dict],
    client: OpenAICompatClient | None,
    use_online_judge: bool = True,
    dialogue_summary: str = "",
    related_forbidden_claims: list[dict] | None = None,
    question_type: str = "main",
    thread_context: dict | None = None,
) -> dict:
    if use_online_judge and client and client.is_ready():
        try:
            tpl = load_prompt(_judge_prompt_name(question_type))
            user_prompt = render_prompt(
                tpl,
                question=question_text,
                question_type=question_type,
                answer=answer,
                target_kcs_json=json.dumps(target_kcs, ensure_ascii=False),
                dialogue_summary=dialogue_summary,
                thread_context_json=json.dumps(thread_context or {}, ensure_ascii=False),
                related_forbidden_claims_json=json.dumps(related_forbidden_claims or [], ensure_ascii=False),
            )
            result = client.chat_json(
                system_prompt="You are an accurate paper-evaluation judge.",
                user_prompt=user_prompt,
            )
            required = {"state", "coverage", "hallucination_events", "recommended_tasks", "next_action"}
            if required.issubset(result.keys()):
                return result
            missing = sorted(required - set(result.keys()))
            raise RuntimeError(
                f"Online judge returned invalid normalized JSON schema for question_type={question_type}; "
                f"missing={missing}."
            )
        except Exception as exc:
            raise RuntimeError(f"Online judge failed for question_type={question_type}: {type(exc).__name__}: {exc}") from exc
    return judge_answer(question_text, answer, target_kcs)


def _judge_prompt_name(question_type: str) -> str:
    return JUDGE_PROMPTS.get(question_type, "judge_main_answer.txt")


def _contains_semantic(answer_lower: str, phrase: str) -> bool:
    toks = [t for t in re.findall(r"[a-zA-Z]{3,}", phrase.lower()) if t not in {"the", "and", "for", "with"}]
    if not toks:
        return False
    hit = sum(1 for t in toks if t in answer_lower)
    return hit >= max(1, len(toks) // 2)


def _overlap_ratio(answer_lower: str, phrase: str) -> float:
    toks = [t for t in re.findall(r"[a-zA-Z]{3,}", phrase.lower()) if t not in {"the", "and", "for", "with"}]
    if not toks:
        return 0.0
    hit = sum(1 for t in toks if t in answer_lower)
    return hit / len(toks)


def _forbidden_hit(answer_lower: str, forbidden_claim: str) -> bool:
    # Avoid over-triggering: require explicit contradiction cues plus overlap.
    contradiction_cues = ["not", "never", "reject", "reverse", "wrong", "cannot", "no "]
    if not any(c in answer_lower for c in contradiction_cues):
        return False
    return _overlap_ratio(answer_lower, forbidden_claim) >= 0.45
