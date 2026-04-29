from __future__ import annotations

import json

from src.model_client import OpenAICompatClient
from src.prompt_loader import load_prompt, render_prompt

MAIN_PROMPTS = {
    "M1": "这篇论文为什么要提出该方法？它试图解决已有研究中的什么问题？",
    "M2": "论文提出的核心方法是什么？它由哪些关键机制支撑？",
    "M3": "作者如何通过实验验证方法有效性？哪些结果最能支撑核心主张？",
    "M4": "论文最终得出了什么结论？还有哪些未充分验证的地方？",
}


def generate_main_questions(graph: dict) -> list[dict]:
    by_macro = {m["macro_id"]: m for m in graph.get("macro_nodes", [])}
    items = []
    for macro_id in ["M1", "M2", "M3", "M4"]:
        m = by_macro.get(macro_id, {"kc_ids": []})
        kcs = m.get("kc_ids", [])
        if not kcs:
            continue
        items.append(
            {
                "question_id": f"Q_{macro_id}",
                "question_type": "main",
                "macro_id": macro_id,
                "target_kc_ids": kcs[:3],
                "question_text": MAIN_PROMPTS[macro_id],
                "expected_coverage": {"must_cover": kcs[:2], "optional_cover": kcs[2:3]},
                "allowed_next_actions": [
                    "detail_followup",
                    "hallucination_followup",
                    "multi_hop_question",
                    "next_main_question",
                ],
            }
        )
    return items


def generate_multi_hop_questions(graph: dict) -> list[dict]:
    by_path = graph.get("reasoning_paths", [])
    by_kc = {k["kc_id"]: k for k in graph.get("kc_nodes", [])}
    questions = []
    for path in by_path:
        seq = path.get("kc_sequence", [])
        if len(seq) < 3:
            continue
        c1 = by_kc.get(seq[0], {}).get("full_claim", seq[0])
        c2 = by_kc.get(seq[1], {}).get("full_claim", seq[1])
        c3 = by_kc.get(seq[2], {}).get("full_claim", seq[2])
        questions.append(
            {
                "question_id": f"Q_{path['path_id']}",
                "question_type": "multi_hop_reasoning",
                "path_id": path["path_id"],
                "target_kc_ids": seq[:3],
                "question_text": (
                    "请把下面三个主张串成一个完整论证链，并说明它们如何共同支撑论文结论："
                    f"\n1) {c1}\n2) {c2}\n3) {c3}"
                ),
                "expected_reasoning": {"must_connect": seq[:3]},
            }
        )
    return questions


def generate_questions_with_online_fallback(graph: dict, client: OpenAICompatClient | None) -> dict:
    if client and client.is_ready():
        try:
            tpl = load_prompt("generate_questions.txt")
            user_prompt = render_prompt(tpl, graph_json=json.dumps(graph, ensure_ascii=False))
            result = client.chat_json(
                system_prompt="You are a paper evaluation question generation assistant.",
                user_prompt=user_prompt,
            )
            main_q = result.get("main_questions", [])
            hop_q = result.get("multi_hop_questions", [])
            if main_q and hop_q:
                return {
                    "main_questions": main_q,
                    "multi_hop_questions": hop_q,
                    "reserved_followup_templates": result.get("reserved_followup_templates", []),
                }
        except Exception:
            pass
    return {
        "main_questions": generate_main_questions(graph),
        "multi_hop_questions": generate_multi_hop_questions(graph),
        "reserved_followup_templates": [],
    }
