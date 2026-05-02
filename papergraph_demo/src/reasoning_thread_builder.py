from __future__ import annotations

import json
import os

from src.model_client import OpenAICompatClient
from src.progress import log, span
from src.prompt_loader import load_prompt, render_prompt


ALLOWED_THREAD_TYPES = {
    "problem_to_method",
    "method_to_result",
    "module_to_ablation",
    "claim_to_limitation",
}

ALLOWED_STEP_ROLES = {
    "establish_premise",
    "establish_evidence",
    "bridge_reasoning",
    "review_consistency",
}


def build_reasoning_threads(
    paper_id: str,
    macro_spine: dict,
    active_kc: dict,
    reasoning_edges: list[dict],
    reasoning_paths: list[dict],
    client: OpenAICompatClient,
) -> dict:
    if not client or not client.is_ready():
        raise RuntimeError("Reasoning Thread generation requires a configured online model client.")

    thread_target = _bounded_env_int("REASONING_THREAD_TARGET", 4, 1, 20)
    thread_min = _bounded_env_int("REASONING_THREAD_MIN", 2, 1, thread_target)
    thread_max = _bounded_env_int("REASONING_THREAD_MAX", 5, thread_target, 20)
    if thread_min > thread_max:
        raise ValueError(f"Invalid reasoning thread bounds: min={thread_min}, max={thread_max}")

    graph_context = {
        "paper_id": paper_id,
        "macro_spine": macro_spine,
        "active_kc_nodes": active_kc.get("kc_nodes", []),
        "reasoning_edges": reasoning_edges,
        "reasoning_paths": reasoning_paths,
        "thread_types": sorted(ALLOWED_THREAD_TYPES),
    }
    tpl = load_prompt("build_reasoning_threads.txt")
    with span("generate reasoning threads", target=thread_target):
        result = client.chat_json(
            system_prompt="You construct cross-turn reasoning threads for dynamic paper evaluation. Return JSON only.",
            user_prompt=render_prompt(
                tpl,
                graph_context_json=json.dumps(graph_context, ensure_ascii=False, indent=2),
                thread_target=str(thread_target),
                thread_min=str(thread_min),
                thread_max=str(thread_max),
            ),
            temperature=0.2,
        )
    threads = _validate_threads(
        raw_threads=result.get("threads", []),
        paper_id=paper_id,
        macro_spine=macro_spine,
        active_kc=active_kc,
        reasoning_edges=reasoning_edges,
        min_count=thread_min,
        max_count=thread_max,
    )
    log("reasoning threads generated", count=len(threads["threads"]))
    return threads


def _validate_threads(
    raw_threads: object,
    paper_id: str,
    macro_spine: dict,
    active_kc: dict,
    reasoning_edges: list[dict],
    min_count: int,
    max_count: int,
) -> dict:
    if not isinstance(raw_threads, list):
        raise ValueError("Reasoning thread response must contain a threads list.")
    if not (min_count <= len(raw_threads) <= max_count):
        raise ValueError(
            f"Reasoning thread count {len(raw_threads)} is outside allowed range [{min_count}, {max_count}]."
        )

    valid_macro_ids = {m["macro_id"] for m in macro_spine.get("macro_nodes", [])}
    valid_kc_ids = {kc["kc_id"] for kc in active_kc.get("kc_nodes", [])}
    valid_edge_ids = {edge.get("edge_id") for edge in reasoning_edges if edge.get("edge_id")}
    threads = []
    seen_thread_ids = set()
    for idx, raw in enumerate(raw_threads, start=1):
        if not isinstance(raw, dict):
            raise ValueError(f"Thread #{idx} must be an object.")
        thread_id = str(raw.get("thread_id") or f"RT{idx}").strip()
        expected_id = f"RT{idx}"
        if thread_id != expected_id:
            raise ValueError(f"Thread IDs must be consecutive; expected {expected_id}, got {thread_id}.")
        if thread_id in seen_thread_ids:
            raise ValueError(f"Duplicate thread_id: {thread_id}")
        seen_thread_ids.add(thread_id)
        thread_type = str(raw.get("thread_type", "")).strip()
        if thread_type not in ALLOWED_THREAD_TYPES:
            raise ValueError(f"{thread_id} has invalid thread_type={thread_type!r}.")
        macro_sequence = _valid_id_sequence(raw.get("macro_sequence", []), valid_macro_ids, f"{thread_id}.macro_sequence")
        if len(set(macro_sequence)) < 2:
            raise ValueError(f"{thread_id} must span at least two Macros.")
        kc_sequence = _valid_id_sequence(raw.get("kc_sequence", []), valid_kc_ids, f"{thread_id}.kc_sequence")
        if len(kc_sequence) < 2:
            raise ValueError(f"{thread_id} must contain at least two KCs.")
        edge_sequence = _valid_id_sequence(raw.get("edge_sequence", []), valid_edge_ids, f"{thread_id}.edge_sequence", allow_empty=True)
        planned_turns = _validate_planned_turns(thread_id, raw.get("planned_turns", []), valid_macro_ids, valid_kc_ids)
        roles = {turn["role"] for turn in planned_turns}
        if "bridge_reasoning" not in roles:
            raise ValueError(f"{thread_id} must include a bridge_reasoning planned turn.")
        if "review_consistency" not in roles:
            raise ValueError(f"{thread_id} must include a review_consistency planned turn.")
        threads.append(
            {
                "thread_id": thread_id,
                "thread_type": thread_type,
                "description": str(raw.get("description", "")).strip(),
                "macro_sequence": macro_sequence,
                "kc_sequence": kc_sequence,
                "edge_sequence": edge_sequence,
                "status": "not_started",
                "planned_turns": planned_turns,
                "success_criteria": _string_list(raw.get("success_criteria", [])),
            }
        )
    return {"paper_id": paper_id, "threads": threads}


def _validate_planned_turns(
    thread_id: str,
    raw_turns: object,
    valid_macro_ids: set[str],
    valid_kc_ids: set[str],
) -> list[dict]:
    if not isinstance(raw_turns, list) or len(raw_turns) < 3 or len(raw_turns) > 4:
        raise ValueError(f"{thread_id}.planned_turns must contain 3-4 turns.")
    turns = []
    seen_ids = set()
    for idx, raw in enumerate(raw_turns, start=1):
        if not isinstance(raw, dict):
            raise ValueError(f"{thread_id}.planned_turns[{idx}] must be an object.")
        step_id = str(raw.get("thread_turn_id") or f"{thread_id}_STEP{idx}").strip()
        expected_id = f"{thread_id}_STEP{idx}"
        if step_id != expected_id:
            raise ValueError(f"Thread step IDs must be consecutive; expected {expected_id}, got {step_id}.")
        if step_id in seen_ids:
            raise ValueError(f"Duplicate thread step id: {step_id}")
        seen_ids.add(step_id)
        role = str(raw.get("role", "")).strip()
        if role not in ALLOWED_STEP_ROLES:
            raise ValueError(f"{step_id} has invalid role={role!r}.")
        preferred_macro = raw.get("preferred_macro_id")
        if preferred_macro is not None:
            preferred_macro = str(preferred_macro).strip() or None
        if preferred_macro is not None and preferred_macro not in valid_macro_ids:
            raise ValueError(f"{step_id} references invalid preferred_macro_id={preferred_macro!r}.")
        target_kc_ids = _valid_id_sequence(raw.get("target_kc_ids", []), valid_kc_ids, f"{step_id}.target_kc_ids")
        trigger_condition = raw.get("trigger_condition", {})
        if not isinstance(trigger_condition, dict):
            raise ValueError(f"{step_id}.trigger_condition must be an object.")
        _validate_trigger_condition(step_id, role, trigger_condition, valid_macro_ids, valid_kc_ids)
        question_goal = str(raw.get("question_goal", "")).strip()
        if not question_goal:
            raise ValueError(f"{step_id} must include question_goal.")
        turns.append(
            {
                "thread_turn_id": step_id,
                "role": role,
                "preferred_macro_id": preferred_macro,
                "target_kc_ids": target_kc_ids,
                "question_goal": question_goal,
                "trigger_condition": trigger_condition,
            }
        )
    return turns


def _validate_trigger_condition(
    step_id: str,
    role: str,
    trigger: dict,
    valid_macro_ids: set[str],
    valid_kc_ids: set[str],
) -> None:
    macro_reached = trigger.get("macro_reached")
    if macro_reached is not None and macro_reached not in valid_macro_ids:
        raise ValueError(f"{step_id}.trigger_condition.macro_reached is invalid: {macro_reached}")
    required_lit = trigger.get("required_lit_kc_ids", [])
    if required_lit:
        _valid_id_sequence(required_lit, valid_kc_ids, f"{step_id}.trigger_condition.required_lit_kc_ids")
    if role == "bridge_reasoning" and not required_lit:
        raise ValueError(f"{step_id} bridge_reasoning must include required_lit_kc_ids.")
    if role == "review_consistency" and not trigger.get("at_review_stage"):
        raise ValueError(f"{step_id} review_consistency must include at_review_stage=true.")


def _valid_id_sequence(
    values: object,
    valid_ids: set[str],
    field_name: str,
    allow_empty: bool = False,
) -> list[str]:
    ids = _string_list(values)
    if not ids and not allow_empty:
        raise ValueError(f"{field_name} must not be empty.")
    bad = [item for item in ids if item not in valid_ids]
    if bad:
        raise ValueError(f"{field_name} contains invalid IDs: {bad}")
    return ids


def _string_list(values: object) -> list[str]:
    if not isinstance(values, list):
        return []
    return [str(value).strip() for value in values if str(value).strip()]


def _bounded_env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    raw = os.getenv(name)
    try:
        value = int(raw) if raw is not None and raw.strip() else default
    except ValueError:
        value = default
    return max(minimum, min(maximum, value))

