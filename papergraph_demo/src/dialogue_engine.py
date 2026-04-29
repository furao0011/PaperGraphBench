from __future__ import annotations

import json

from src.model_client import OpenAICompatClient
from src.prompt_loader import load_prompt, render_prompt


def generate_followup_question(
    action: str,
    last_turn: dict,
    target_kcs: list[dict],
    client: OpenAICompatClient | None = None,
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
        prompt_kwargs = {
            "question": last_turn.get("question_text", ""),
            "missing_kcs": json.dumps([k.get("full_claim", k.get("kc_id")) for k in target_kcs], ensure_ascii=False),
        }
        fallback_text = f"你刚才的回答还不完整。请补充说明这条主张，并明确关键机制与证据：{claim}"
    elif action == "hallucination_followup":
        qtype = "hallucination_followup"
        hint = primary.get("forbidden_claims", [{}])[0].get("followup_hint", "请回到论文证据修正说法。")
        prompt_name = "generate_hallucination_followup.txt"
        prompt_kwargs = {
            "question": last_turn.get("question_text", ""),
            "hallucinated_claims": json.dumps(last_turn.get("judge_result", {}).get("hallucinated_claims", []), ensure_ascii=False),
            "followup_hints": json.dumps([hint], ensure_ascii=False),
        }
        fallback_text = f"你刚才可能出现了错误前提。{hint}"
    elif action == "misleading_followup":
        qtype = "misleading_followup"
        prompt_name = "generate_misleading_followup.txt"
        prompt_kwargs = {
            "target_kc": claim,
            "forbidden_claims": json.dumps(primary.get("forbidden_claims", []), ensure_ascii=False),
        }
        fallback_text = f"是否可以认为这条主张只是次要背景点，不影响核心结论？请判断并解释：{claim}"
    elif action == "review_followup":
        qtype = "review_followup"
        prompt_name = "generate_review_followup.txt"
        prompt_kwargs = {
            "target_kc": claim,
            "context": last_turn.get("question_text", ""),
        }
        fallback_text = f"复习一下：请再次准确解释这条主张，并说明它在整篇论文中的作用：{claim}"
    else:
        return None

    question_text = fallback_text
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
        except Exception:
            pass

    return {
        "question_id": f"{last_turn['question_id']}_F",
        "question_type": qtype,
        "macro_id": last_turn.get("macro_id"),
        "target_kc_ids": [kc_id],
        "target_path_id": last_turn.get("target_path_id"),
        "question_text": question_text,
    }

