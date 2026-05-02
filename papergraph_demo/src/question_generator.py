from __future__ import annotations

import json
from pathlib import Path

from src.model_client import OpenAICompatClient
from src.progress import log, span
from src.prompt_loader import load_prompt, render_prompt


def generate_questions_with_online_fallback(
    graph: dict,
    client: OpenAICompatClient | None,
    allow_offline_fallback: bool = False,
) -> dict:
    if client and client.is_ready():
        try:
            tpl = load_prompt("generate_questions.txt")
            user_prompt = render_prompt(tpl, graph_json=json.dumps(graph, ensure_ascii=False))
            result = client.chat_json(
                system_prompt="You generate graph-grounded paper evaluation questions. Return JSON only.",
                user_prompt=user_prompt,
                temperature=0.2,
            )
            bundle = normalize_question_bundle(graph, result)
            if bundle["macro_main_questions"]:
                return bundle
        except Exception as exc:
            if not allow_offline_fallback:
                raise RuntimeError(f"Online question generation failed: {type(exc).__name__}: {exc}") from exc

    if not allow_offline_fallback:
        raise RuntimeError("Online question generation failed and offline fallback is disabled.")
    return _debug_question_bundle(graph)


def generate_questions_cached(
    graph: dict,
    client: OpenAICompatClient | None,
    cache_path: Path,
    resume: bool = False,
    restart: bool = False,
    allow_offline_fallback: bool = False,
) -> dict:
    if not client or not client.is_ready():
        if allow_offline_fallback:
            return _debug_question_bundle(graph)
        raise RuntimeError("Online question generation requires a configured model client.")

    signature = _graph_signature(graph)
    cache = _load_question_cache(cache_path) if resume and not restart else {}
    if cache.get("graph_signature") != signature:
        cache = {
            "paper_id": graph.get("paper_id", "unknown"),
            "graph_signature": signature,
            "macro_main_questions": {},
            "thread_question_seeds": {},
            "review_question_seeds": [],
        }
    cache.setdefault("macro_main_questions", cache.pop("main_questions", {}))
    cache.setdefault("thread_question_seeds", {})
    cache.setdefault("review_question_seeds", [])
    _write_json(cache_path, cache)

    macro_main_questions = []
    for macro in graph.get("macro_nodes", []):
        macro_id = macro.get("macro_id")
        if not macro_id or not macro.get("kc_ids"):
            continue
        cached = cache["macro_main_questions"].get(macro_id)
        if cached:
            try:
                macro_main_questions.append(_normalize_macro_main_question(graph, macro_id, cached))
                log("question cache hit", kind="macro_main", id=macro_id)
                continue
            except Exception as exc:
                cache["macro_main_questions"].pop(macro_id, None)
                _write_json(cache_path, cache)
                log("question cache invalidated", kind="macro_main", id=macro_id, error=f"{type(exc).__name__}: {exc}")
        with span("generate macro main question", macro_id=macro_id):
            question = _generate_one_macro_main_question(graph, macro, client)
        cache["macro_main_questions"][macro_id] = question
        _write_json(cache_path, cache)
        macro_main_questions.append(question)
        log("question cached", kind="macro_main", id=macro_id, path=cache_path)

    thread_question_seeds = _thread_question_seeds(graph)
    cache["thread_question_seeds"] = {seed["thread_turn_id"]: seed for seed in thread_question_seeds}
    _write_json(cache_path, cache)

    return {
        "macro_main_questions": macro_main_questions,
        "thread_question_seeds": thread_question_seeds,
        "review_question_seeds": cache.get("review_question_seeds", []),
        "main_questions": [_legacy_main_question(q) for q in macro_main_questions],
        "multi_hop_questions": [],
        "reserved_followup_templates": [],
    }


def normalize_question_bundle(graph: dict, result: dict) -> dict:
    valid_kc_ids = {k["kc_id"] for k in graph.get("kc_nodes", [])}
    macro_targets = {
        m["macro_id"]: [kid for kid in m.get("kc_ids", []) if kid in valid_kc_ids]
        for m in graph.get("macro_nodes", [])
    }
    raw_macro_questions = result.get("macro_main_questions", result.get("main_questions", []))
    main_by_macro = {
        q.get("macro_id"): q
        for q in raw_macro_questions
        if q.get("question_type") in {"macro_main_question", "main"} and q.get("macro_id") in macro_targets
    }
    macro_main_questions = []
    for macro_id, target_ids in macro_targets.items():
        if not target_ids:
            continue
        source = main_by_macro.get(macro_id)
        if not source:
            raise ValueError(f"Missing main question for {macro_id}")
        macro_main_questions.append(_normalize_macro_main_question(graph, macro_id, source))

    return {
        "macro_main_questions": macro_main_questions,
        "thread_question_seeds": _thread_question_seeds(graph),
        "review_question_seeds": result.get("review_question_seeds", []),
        "main_questions": [_legacy_main_question(q) for q in macro_main_questions],
        "multi_hop_questions": [],
        "reserved_followup_templates": result.get("reserved_followup_templates", []),
    }

def _generate_one_macro_main_question(graph: dict, macro: dict, client: OpenAICompatClient) -> dict:
    macro_id = macro["macro_id"]
    target_ids = set(macro.get("kc_ids", []))
    subgraph = {
        "paper_id": graph.get("paper_id", "unknown"),
        "macro_nodes": [macro],
        "kc_nodes": [k for k in graph.get("kc_nodes", []) if k.get("kc_id") in target_ids],
        "reasoning_paths": [],
    }
    tpl = load_prompt("generate_questions.txt")
    result = client.chat_json(
        system_prompt="You generate graph-grounded paper evaluation questions. Return JSON only.",
        user_prompt=render_prompt(tpl, graph_json=json.dumps(subgraph, ensure_ascii=False)),
        temperature=0.2,
    )
    source = next(
        (
            q
            for q in result.get("macro_main_questions", result.get("main_questions", []))
            if q.get("question_type") in {"macro_main_question", "main"} and q.get("macro_id") == macro_id
        ),
        None,
    )
    if not source:
        raise RuntimeError(f"Online question generation returned no macro main question for {macro_id}.")
    return _normalize_macro_main_question(graph, macro_id, source)


def _generate_one_multi_hop_question(graph: dict, path: dict, client: OpenAICompatClient) -> dict:
    path_id = path["path_id"]
    seq = set(path.get("kc_sequence", [])[:3])
    macro_ids = {k.get("macro_id") for k in graph.get("kc_nodes", []) if k.get("kc_id") in seq}
    subgraph = {
        "paper_id": graph.get("paper_id", "unknown"),
        "macro_nodes": [m for m in graph.get("macro_nodes", []) if m.get("macro_id") in macro_ids],
        "kc_nodes": [k for k in graph.get("kc_nodes", []) if k.get("kc_id") in seq],
        "reasoning_paths": [path],
    }
    tpl = load_prompt("generate_questions.txt")
    result = client.chat_json(
        system_prompt="You generate graph-grounded paper evaluation questions. Return JSON only.",
        user_prompt=render_prompt(tpl, graph_json=json.dumps(subgraph, ensure_ascii=False)),
        temperature=0.2,
    )
    source = next(
        (
            q
            for q in result.get("multi_hop_questions", [])
            if q.get("question_type") == "multi_hop_reasoning" and q.get("path_id") == path_id
        ),
        None,
    )
    if not source:
        raise RuntimeError(f"Online question generation returned no multi-hop question for {path_id}.")
    return _normalize_multi_hop_question(graph, path_id, source)


def _normalize_macro_main_question(graph: dict, macro_id: str, source: dict) -> dict:
    valid_kc_ids = {k["kc_id"] for k in graph.get("kc_nodes", [])}
    macro = next((m for m in graph.get("macro_nodes", []) if m.get("macro_id") == macro_id), None)
    if not macro:
        raise ValueError(f"Unknown macro id in question cache: {macro_id}")
    target_ids = [kid for kid in macro.get("kc_ids", []) if kid in valid_kc_ids]
    targets = [kid for kid in source.get("target_kc_ids", []) if kid in target_ids] or target_ids[: min(3, len(target_ids))]
    question_text = str(source.get("question_text", "")).strip()
    if not question_text:
        raise ValueError(f"Empty main question text for {macro_id}")
    return {
        "question_id": f"Q_{macro_id}",
        "question_type": "macro_main_question",
        "macro_id": macro_id,
        "target_kc_ids": targets,
        "question_text": question_text,
        "expected_coverage": {
            "must_cover": targets[: min(2, len(targets))],
            "optional_cover": targets[2:],
        },
        "allowed_next_actions": [
            "detail_followup",
            "hallucination_followup",
            "thread_question",
            "next_macro_question",
        ],
    }


def _normalize_multi_hop_question(graph: dict, path_id: str, source: dict) -> dict:
    path = next((p for p in graph.get("reasoning_paths", []) if p.get("path_id") == path_id), None)
    if not path:
        raise ValueError(f"Unknown path id in question cache: {path_id}")
    valid_kc_ids = {k["kc_id"] for k in graph.get("kc_nodes", [])}
    seq = [kid for kid in path.get("kc_sequence", [])[:3] if kid in valid_kc_ids]
    if len(seq) < 3:
        raise ValueError(f"Path {path_id} has fewer than three valid KCs.")
    question_text = str(source.get("question_text", "")).strip()
    if not question_text:
        raise ValueError(f"Empty multi-hop question text for {path_id}")
    return {
        "question_id": f"Q_{path_id}",
        "question_type": "multi_hop_reasoning",
        "path_id": path_id,
        "target_kc_ids": seq,
        "question_text": question_text,
        "expected_reasoning": {"must_connect": seq},
    }


def _graph_signature(graph: dict) -> dict:
    return {
        "paper_id": graph.get("paper_id", "unknown"),
        "kc_ids": [k.get("kc_id") for k in graph.get("kc_nodes", [])],
        "macro_ids": [m.get("macro_id") for m in graph.get("macro_nodes", [])],
        "path_ids": [p.get("path_id") for p in graph.get("reasoning_paths", [])],
        "thread_ids": [t.get("thread_id") for t in graph.get("reasoning_threads", [])],
        "thread_step_ids": [
            step.get("thread_turn_id")
            for thread in graph.get("reasoning_threads", [])
            for step in thread.get("planned_turns", [])
        ],
    }


def _thread_question_seeds(graph: dict) -> list[dict]:
    seeds = []
    for thread in graph.get("reasoning_threads", []):
        thread_id = thread.get("thread_id")
        for step in thread.get("planned_turns", []):
            role = step.get("role")
            qtype = {
                "establish_premise": "thread_premise_question",
                "establish_evidence": "thread_evidence_question",
                "bridge_reasoning": "thread_bridge_question",
                "review_consistency": "thread_review_question",
            }.get(role, "thread_question")
            seeds.append(
                {
                    "question_id": f"Q_{step.get('thread_turn_id')}",
                    "question_type": qtype,
                    "thread_id": thread_id,
                    "thread_turn_id": step.get("thread_turn_id"),
                    "thread_role": role,
                    "preferred_macro_id": step.get("preferred_macro_id"),
                    "target_kc_ids": step.get("target_kc_ids", []),
                    "question_goal": step.get("question_goal", ""),
                    "trigger_condition": step.get("trigger_condition", {}),
                    "requires_runtime_generation": role in {"bridge_reasoning", "review_consistency"},
                }
            )
    return seeds


def _legacy_main_question(question: dict) -> dict:
    legacy = dict(question)
    legacy["question_type"] = "main"
    legacy["allowed_next_actions"] = [
        "detail_followup",
        "hallucination_followup",
        "multi_hop_question",
        "next_main_question",
    ]
    return legacy


def _load_question_cache(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _debug_question_bundle(graph: dict) -> dict:
    by_kc = {k["kc_id"]: k for k in graph.get("kc_nodes", [])}
    macro_main_questions = []
    for macro in graph.get("macro_nodes", []):
        target_ids = macro.get("kc_ids", [])[: min(3, len(macro.get("kc_ids", [])))]
        if not target_ids:
            continue
        claims = "; ".join(by_kc.get(kid, {}).get("full_claim", kid) for kid in target_ids)
        macro_main_questions.append(
            {
                "question_id": f"Q_{macro['macro_id']}",
                "question_type": "macro_main_question",
                "macro_id": macro["macro_id"],
                "target_kc_ids": target_ids,
                "question_text": f"Explain the paper content covered by these target claims, without adding unsupported details: {claims}",
                "expected_coverage": {"must_cover": target_ids[:2], "optional_cover": target_ids[2:]},
                "allowed_next_actions": [
                    "detail_followup",
                    "hallucination_followup",
                    "thread_question",
                    "next_macro_question",
                ],
            }
        )

    return {
        "macro_main_questions": macro_main_questions,
        "thread_question_seeds": _thread_question_seeds(graph),
        "review_question_seeds": [],
        "main_questions": [_legacy_main_question(q) for q in macro_main_questions],
        "multi_hop_questions": [],
        "reserved_followup_templates": [],
    }
