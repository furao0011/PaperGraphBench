from __future__ import annotations

import json

from src.model_client import OpenAICompatClient
from src.prompt_loader import load_prompt, render_prompt


def generate_followup_question(
    action: str,
    last_turn: dict,
    target_kcs: list[dict],
    client: OpenAICompatClient | None = None,
    allow_offline_fallback: bool = False,
) -> dict | None:
    if not target_kcs:
        return None
    primary = target_kcs[0]
    kc_id = primary["kc_id"]
    claim = primary.get("full_claim", kc_id)

    prompt_name = None
    prompt_kwargs = {}
    qtype = ""
    fallback_text = ""

    if action == "detail_followup":
        qtype = "detail_followup"
        prompt_name = "generate_detail_followup.txt"
        missing_claims = [k.get("full_claim", k.get("kc_id")) for k in target_kcs]
        prompt_kwargs = {
            "question": last_turn.get("question_text", ""),
            "missing_kcs": json.dumps(missing_claims, ensure_ascii=False),
        }
        claims_text = "; ".join(str(c) for c in missing_claims[:2])
        fallback_text = (
            "Your previous answer missed target paper claim(s). "
            f"Please answer only these missing claims using paper evidence, without adding unsupported details: {claims_text}"
        )
    elif action == "hallucination_followup":
        qtype = "hallucination_followup"
        hallucinated = last_turn.get("judge_result", {}).get("hallucinated_claims", [])
        hint = primary.get("forbidden_claims", [{}])[0].get("followup_hint", "Re-check the statement against the paper evidence and correct it.")
        prompt_name = "generate_hallucination_followup.txt"
        prompt_kwargs = {
            "question": last_turn.get("question_text", ""),
            "hallucinated_claims": json.dumps(hallucinated, ensure_ascii=False),
            "followup_hints": json.dumps([hint], ensure_ascii=False),
        }
        claims_text = "; ".join(str(c) for c in hallucinated[:3]) if hallucinated else "the suspicious claim(s) in the previous answer"
        fallback_text = (
            "Some statements in your previous answer may be unsupported or need correction: "
            f"{claims_text}. Please check each one against the paper, retract or revise unsupported parts, and give a corrected answer."
        )
    elif action == "misleading_followup":
        qtype = "misleading_followup"
        prompt_name = "generate_misleading_followup.txt"
        prompt_kwargs = {
            "target_kc": claim,
            "forbidden_claims": json.dumps(primary.get("forbidden_claims", []), ensure_ascii=False),
        }
        fallback_text = f"Could this target claim be treated as a minor background point with no effect on the paper's main conclusions? Judge carefully and explain using paper evidence: {claim}"
    elif action == "review_followup":
        qtype = "review_followup"
        prompt_name = "generate_review_followup.txt"
        prompt_kwargs = {
            "target_kc": claim,
            "context": last_turn.get("question_text", ""),
        }
        fallback_text = f"Review this target claim accurately and explain its role in the paper: {claim}"
    else:
        return None

    question_text = ""
    if client and client.is_ready() and prompt_name:
        try:
            tpl = load_prompt(prompt_name)
            user_prompt = render_prompt(tpl, **prompt_kwargs)
            out = client.chat_json(
                system_prompt="You generate one concise follow-up question for paper evaluation.",
                user_prompt=user_prompt,
            )
            qtxt = str(out.get("question_text", "")).strip()
            if qtxt:
                question_text = qtxt
        except Exception as exc:
            if not allow_offline_fallback:
                raise RuntimeError(f"Online follow-up generation failed for {action}: {type(exc).__name__}: {exc}") from exc

    if not question_text:
        if not allow_offline_fallback:
            raise RuntimeError(f"Online follow-up generation failed for {action} and offline fallback is disabled.")
        question_text = fallback_text

    return {
        "question_id": f"{last_turn['question_id']}_F",
        "question_type": qtype,
        "macro_id": last_turn.get("macro_id"),
        "target_kc_ids": [k["kc_id"] for k in target_kcs],
        "target_path_id": last_turn.get("target_path_id"),
        "question_text": question_text,
    }
