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


def generate_thread_question(
    thread_turn: dict,
    target_kcs: list[dict],
    related_turns: list[dict],
    dialogue_summary: str,
    client: OpenAICompatClient | None = None,
    allow_offline_fallback: bool = False,
) -> dict:
    if not target_kcs:
        raise ValueError("Thread question requires at least one target KC.")

    question_text = str(thread_turn.get("question_text", "")).strip()
    if client and client.is_ready() and not question_text:
        try:
            tpl = load_prompt("generate_thread_question.txt")
            out = client.chat_json(
                system_prompt="You generate one concise cross-turn reasoning question for paper evaluation.",
                user_prompt=render_prompt(
                    tpl,
                    thread_turn_json=json.dumps(thread_turn, ensure_ascii=False),
                    target_kcs_json=json.dumps(target_kcs, ensure_ascii=False),
                    related_turns_json=json.dumps(_compact_related_turns(related_turns), ensure_ascii=False),
                    dialogue_summary=dialogue_summary,
                ),
                temperature=0.2,
            )
            question_text = str(out.get("question_text", "")).strip()
        except Exception as exc:
            if not allow_offline_fallback:
                raise RuntimeError(
                    f"Online thread question generation failed for {thread_turn.get('thread_turn_id')}: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc

    if not question_text:
        if not allow_offline_fallback:
            raise RuntimeError(
                f"Online thread question generation failed for {thread_turn.get('thread_turn_id')} "
                "and offline fallback is disabled."
            )
        claims = "; ".join(k.get("full_claim", k.get("kc_id", "")) for k in target_kcs[:3])
        question_text = f"{thread_turn.get('question_goal', 'Connect the relevant paper claims')}: {claims}"

    return {
        "question_id": thread_turn.get("question_id") or f"Q_{thread_turn.get('thread_turn_id')}",
        "question_type": thread_turn.get("question_type", "thread_question"),
        "thread_id": thread_turn.get("thread_id"),
        "thread_turn_id": thread_turn.get("thread_turn_id"),
        "thread_role": thread_turn.get("thread_role"),
        "macro_id": thread_turn.get("preferred_macro_id") or thread_turn.get("macro_id"),
        "target_kc_ids": [k["kc_id"] for k in target_kcs],
        "question_goal": thread_turn.get("question_goal", ""),
        "trigger_condition": thread_turn.get("trigger_condition", {}),
        "question_text": question_text,
    }


def _compact_related_turns(turns: list[dict]) -> list[dict]:
    return [
        {
            "turn_id": t.get("turn_id"),
            "question_type": t.get("question_type"),
            "question_text": t.get("question_text"),
            "model_answer": t.get("model_answer"),
            "covered_kc_ids": t.get("judge_result", {}).get("covered_kc_ids", []),
        }
        for t in turns[-6:]
    ]
