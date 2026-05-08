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

THREAD_BUILDER_VERSION = "v2_verified_edges"


def build_reasoning_threads(
    paper_id: str,
    macro_spine: dict,
    active_kc: dict,
    reasoning_edges: list[dict],
    reasoning_paths: list[dict],
    client: OpenAICompatClient,
    edge_coverage_report: dict | None = None,
) -> dict:
    if not client or not client.is_ready():
        raise RuntimeError("Reasoning Thread generation requires a configured online model client.")

    thread_target = _bounded_env_int("REASONING_THREAD_TARGET", 4, 1, 20)
    thread_min = _bounded_env_int("REASONING_THREAD_MIN", 2, 1, thread_target)
    thread_max = _bounded_env_int("REASONING_THREAD_MAX", 5, thread_target, 20)
    if thread_min > thread_max:
        raise ValueError(f"Invalid reasoning thread bounds: min={thread_min}, max={thread_max}")

    active_ids = {kc["kc_id"] for kc in active_kc.get("kc_nodes", []) if kc.get("kc_id")}
    active_edges = _active_reasoning_edges(reasoning_edges, active_ids)
    active_paths = _active_reasoning_paths(reasoning_paths, active_ids)
    if not active_edges:
        raise RuntimeError("Reasoning Thread v2 requires at least one verified edge inside the Active KC subgraph.")

    graph_context = {
        "paper_id": paper_id,
        "macro_spine": macro_spine,
        "active_kc_nodes": active_kc.get("kc_nodes", []),
        "verified_active_edges": active_edges,
        "verified_active_paths": active_paths,
        "edge_coverage_report": edge_coverage_report or {},
        "thread_types": sorted(ALLOWED_THREAD_TYPES),
        "thread_builder_version": THREAD_BUILDER_VERSION,
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
        reasoning_edges=active_edges,
        min_count=thread_min,
        max_count=thread_max,
    )
    log("reasoning threads generated", count=len(threads["threads"]))
    threads["thread_builder_version"] = THREAD_BUILDER_VERSION
    threads["source_edge_count"] = len(active_edges)
    threads["source_path_count"] = len(active_paths)
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
    edge_by_id = {
        edge["edge_id"]: edge
        for edge in reasoning_edges
        if edge.get("edge_id")
    }
    valid_edge_ids = set(edge_by_id)
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
        edge_sequence = _valid_id_sequence(raw.get("edge_sequence", []), valid_edge_ids, f"{thread_id}.edge_sequence")
        planned_turns = _validate_planned_turns(
            thread_id,
            raw.get("planned_turns", []),
            valid_macro_ids,
            valid_kc_ids,
            valid_edge_ids,
            edge_by_id,
        )
        roles = {turn["role"] for turn in planned_turns}
        if "bridge_reasoning" not in roles:
            raise ValueError(f"{thread_id} must include a bridge_reasoning planned turn.")
        if "review_consistency" not in roles:
            raise ValueError(f"{thread_id} must include a review_consistency planned turn.")
        _validate_thread_edge_grounding(thread_id, kc_sequence, edge_sequence, planned_turns, edge_by_id)
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
    valid_edge_ids: set[str],
    edge_by_id: dict[str, dict],
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
        supporting_edge_ids = _valid_id_sequence(
            raw.get("supporting_edge_ids", []),
            valid_edge_ids,
            f"{step_id}.supporting_edge_ids",
            allow_empty=True,
        )
        if role == "bridge_reasoning" and not supporting_edge_ids:
            raise ValueError(f"{step_id} bridge_reasoning must include supporting_edge_ids.")
        _validate_step_edge_targets(step_id, role, target_kc_ids, supporting_edge_ids, edge_by_id)
        trigger_condition = raw.get("trigger_condition", {})
        if not isinstance(trigger_condition, dict):
            raise ValueError(f"{step_id}.trigger_condition must be an object.")
        _validate_trigger_condition(step_id, role, trigger_condition, valid_macro_ids, valid_kc_ids)
        question_goal = str(raw.get("question_goal", "")).strip()
        if not question_goal:
            raise ValueError(f"{step_id} must include question_goal.")
        expected_reasoning = str(raw.get("expected_reasoning", "")).strip()
        if not expected_reasoning:
            raise ValueError(f"{step_id} must include expected_reasoning.")
        turns.append(
            {
                "thread_turn_id": step_id,
                "role": role,
                "preferred_macro_id": preferred_macro,
                "target_kc_ids": target_kc_ids,
                "supporting_edge_ids": supporting_edge_ids,
                "question_goal": question_goal,
                "expected_reasoning": expected_reasoning,
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


def _active_reasoning_edges(reasoning_edges: list[dict], active_ids: set[str]) -> list[dict]:
    return [
        edge
        for edge in reasoning_edges
        if edge.get("source") in active_ids and edge.get("target") in active_ids and edge.get("edge_id")
    ]


def _active_reasoning_paths(reasoning_paths: list[dict], active_ids: set[str]) -> list[dict]:
    out = []
    for path in reasoning_paths:
        seq = _string_list(path.get("kc_sequence", []))
        if seq and all(kc_id in active_ids for kc_id in seq):
            out.append(path)
    return out


def _validate_step_edge_targets(
    step_id: str,
    role: str,
    target_kc_ids: list[str],
    supporting_edge_ids: list[str],
    edge_by_id: dict[str, dict],
) -> None:
    if not supporting_edge_ids:
        return
    target_set = set(target_kc_ids)
    touched = set()
    for edge_id in supporting_edge_ids:
        edge = edge_by_id[edge_id]
        touched.add(edge.get("source"))
        touched.add(edge.get("target"))
    if role == "bridge_reasoning":
        missing = target_set - touched
        if missing:
            raise ValueError(f"{step_id} bridge target_kc_ids are not all grounded by supporting_edge_ids: {sorted(missing)}")
    elif not target_set.intersection(touched):
        raise ValueError(f"{step_id} supporting_edge_ids do not touch any target_kc_ids.")


def _validate_thread_edge_grounding(
    thread_id: str,
    kc_sequence: list[str],
    edge_sequence: list[str],
    planned_turns: list[dict],
    edge_by_id: dict[str, dict],
) -> None:
    kc_set = set(kc_sequence)
    for edge_id in edge_sequence:
        edge = edge_by_id[edge_id]
        endpoints = {edge.get("source"), edge.get("target")}
        if not endpoints.issubset(kc_set):
            raise ValueError(f"{thread_id}.edge_sequence contains edge {edge_id} outside kc_sequence.")
    bridge_edges = {
        edge_id
        for turn in planned_turns
        if turn.get("role") == "bridge_reasoning"
        for edge_id in turn.get("supporting_edge_ids", [])
    }
    if not bridge_edges:
        raise ValueError(f"{thread_id} has no verified edge supporting bridge_reasoning.")
    if not bridge_edges.issubset(set(edge_sequence)):
        raise ValueError(f"{thread_id} bridge supporting_edge_ids must be included in edge_sequence.")


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
