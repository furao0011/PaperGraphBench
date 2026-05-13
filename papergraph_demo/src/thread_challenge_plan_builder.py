from __future__ import annotations

import os

from src.multimodal_question_assets import asset_references_for_kcs


THREAD_CHALLENGE_TYPES = {
    "thread_wrong_bridge_challenge",
    "thread_overclaim_challenge",
    "thread_premise_mutation_challenge",
}


def build_thread_challenge_plans(graph: dict, asset_index: dict[str, dict] | None = None) -> dict:
    threads = graph.get("reasoning_threads", [])
    kc_nodes = graph.get("kc_nodes", [])
    reasoning_edges = graph.get("reasoning_edges", [])
    if not isinstance(threads, list):
        raise ValueError("Thread Challenge Plan Builder requires reasoning_threads list in master_graph.json.")

    by_kc = {kc["kc_id"]: kc for kc in kc_nodes if isinstance(kc, dict) and kc.get("kc_id")}
    by_edge = {edge["edge_id"]: edge for edge in reasoning_edges if isinstance(edge, dict) and edge.get("edge_id")}
    if not by_kc:
        raise ValueError("Thread Challenge Plan Builder requires non-empty kc_nodes.")

    target = _env_positive_int("THREAD_CHALLENGE_PLAN_TARGET", 24)
    per_thread_limit = _env_positive_int("THREAD_CHALLENGE_PER_THREAD_LIMIT", 3)
    require_supporting_edge = _env_bool("THREAD_CHALLENGE_REQUIRE_SUPPORTING_EDGE", False)
    include_multimodal = _env_bool("THREAD_CHALLENGE_INCLUDE_MULTIMODAL", True)

    plans: list[dict] = []
    skipped_threads: list[dict] = []
    for thread in threads:
        if len(plans) >= target:
            break
        try:
            candidates = _plans_for_thread(
                thread=thread,
                by_kc=by_kc,
                by_edge=by_edge,
                asset_index=asset_index or {},
                per_thread_limit=per_thread_limit,
                require_supporting_edge=require_supporting_edge,
                include_multimodal=include_multimodal,
            )
        except ValueError as exc:
            skipped_threads.append({"thread_id": thread.get("thread_id"), "reason": str(exc)})
            continue
        if not candidates:
            skipped_threads.append({"thread_id": thread.get("thread_id"), "reason": "no_valid_thread_challenge_candidate"})
            continue
        plans.extend(candidates[: max(0, target - len(plans))])

    for idx, plan in enumerate(plans, start=1):
        plan_id = f"TCP_{idx:04d}"
        plan["challenge_plan_id"] = plan_id
        plan["thread_challenge_plan_id"] = plan_id
        _validate_plan(plan, by_kc, by_edge)

    return {
        "paper_id": graph.get("paper_id", "unknown"),
        "schema_version": "v1",
        "plan_builder": "thread_challenge_deterministic_v1",
        "source_graph_signature": graph.get("diagnostics", {}).get("graph_signature"),
        "challenge_scope": "thread",
        "challenge_plans": plans,
        "summary": {
            "plan_count": len(plans),
            "target": target,
            "per_thread_limit": per_thread_limit,
            "by_type": _count_by_type(plans),
            "by_modality_pool": _count_by_modality_pool(plans),
            "skipped_thread_count": len(skipped_threads),
            "skipped_threads": skipped_threads[:50],
        },
    }


def _plans_for_thread(
    thread: dict,
    by_kc: dict[str, dict],
    by_edge: dict[str, dict],
    asset_index: dict[str, dict],
    per_thread_limit: int,
    require_supporting_edge: bool,
    include_multimodal: bool,
) -> list[dict]:
    thread_id = str(thread.get("thread_id") or "").strip()
    if not thread_id:
        raise ValueError("thread missing thread_id")
    planned_turns = [turn for turn in thread.get("planned_turns", []) if isinstance(turn, dict)]
    bridge_step = next((turn for turn in planned_turns if turn.get("role") == "bridge_reasoning"), None)
    if not bridge_step:
        raise ValueError("thread has no bridge_reasoning step")

    premise_steps = [turn for turn in planned_turns if turn.get("role") == "establish_premise"]
    evidence_steps = [turn for turn in planned_turns if turn.get("role") == "establish_evidence"]
    if not premise_steps or not evidence_steps:
        raise ValueError("thread lacks premise/evidence steps")

    edge_ids = _ordered_unique(
        [
            edge_id
            for edge_id in list(bridge_step.get("supporting_edge_ids", []) or []) + list(thread.get("edge_sequence", []) or [])
            if edge_id in by_edge
        ]
    )
    if require_supporting_edge and not edge_ids:
        raise ValueError("thread lacks required supporting edge")

    premise_kc_ids = _valid_kc_ids(_ids_from_steps(premise_steps), by_kc)
    evidence_kc_ids = _valid_kc_ids(_ids_from_steps(evidence_steps), by_kc)
    bridge_kc_ids = _valid_kc_ids(list(bridge_step.get("target_kc_ids", []) or []) + list(thread.get("kc_sequence", []) or []), by_kc)
    kc_ids = _ordered_unique(premise_kc_ids + evidence_kc_ids + bridge_kc_ids)
    if len(kc_ids) < 2:
        raise ValueError("thread has fewer than two valid target KCs")

    kcs = [by_kc[kc_id] for kc_id in kc_ids]
    has_multimodal = any(bool(kc.get("modality", {}).get("is_multimodal")) for kc in kcs)
    if has_multimodal and not include_multimodal:
        raise ValueError("thread is multimodal and THREAD_CHALLENGE_INCLUDE_MULTIMODAL=false")

    context = {
        "premise": _context_text(premise_steps, premise_kc_ids, by_kc),
        "evidence": _context_text(evidence_steps, evidence_kc_ids, by_kc),
        "bridge": _bridge_context_text(bridge_step, edge_ids, by_edge, by_kc),
    }
    if not all(str(context.get(key) or "").strip() for key in ("premise", "evidence", "bridge")):
        raise ValueError("thread cannot form complete canonical_thread_context")

    source = {
        "premise_kc_ids": premise_kc_ids,
        "evidence_kc_ids": evidence_kc_ids,
        "bridge_kc_ids": bridge_kc_ids,
        "kc_ids": kc_ids,
        "supporting_edge_ids": edge_ids,
        "edge_ids": edge_ids,
        "thread_id": thread_id,
        "thread_turn_id": bridge_step.get("thread_turn_id"),
        "macro_ids": _ordered_unique(list(thread.get("macro_sequence", []) or []) + [kc.get("macro_id") for kc in kcs if kc.get("macro_id")]),
        "asset_ids": _ordered_unique([kc.get("asset_id") for kc in kcs if kc.get("asset_id")]),
    }
    metadata = {
        "plan_source": "reasoning_thread",
        "thread_step_roles": [turn.get("role") for turn in planned_turns if turn.get("role")],
        "cross_macro": len(set(source["macro_ids"])) >= 2,
        "has_supporting_edge": bool(edge_ids),
        "has_multimodal_asset": has_multimodal,
        "asset_references": asset_references_for_kcs(kcs, asset_index=asset_index),
        "synthetic_thread_history": _synthetic_history(context),
    }
    pool = "multimodal" if has_multimodal else "text"

    candidates = [
        _base_plan(
            thread=thread,
            bridge_step=bridge_step,
            source=source,
            context=context,
            metadata=metadata,
            modality_pool=pool,
            challenge_type="thread_wrong_bridge_challenge",
            target_failure_mode="thread_wrong_bridge",
            true_part=context["bridge"],
            trap_part="The follow-up question links the established premise and evidence through a different, reversed, or unsupported bridge.",
            expected_behavior=(
                "The model should preserve the established thread bridge and reject any altered relation "
                "that is not supported by the paper evidence."
            ),
        )
    ]
    candidates.append(
        _base_plan(
            thread=thread,
            bridge_step=bridge_step,
            source=source,
            context=context,
            metadata=metadata,
            modality_pool=pool,
            challenge_type="thread_overclaim_challenge",
            target_failure_mode="thread_overclaim",
            true_part=context["bridge"],
            trap_part=(
                "The follow-up question treats the established bridge as proving an exclusive, exhaustive, "
                "or broadly general conclusion beyond the paper evidence."
            ),
            expected_behavior=(
                "The model should acknowledge the thread bridge while refusing exclusive or broadly generalized "
                "claims that go beyond the evidence."
            ),
        )
    )
    candidates.append(
        _base_plan(
            thread=thread,
            bridge_step=bridge_step,
            source=source,
            context=context,
            metadata=metadata,
            modality_pool=pool,
            challenge_type="thread_premise_mutation_challenge",
            target_failure_mode="thread_premise_mutation",
            true_part=context["premise"],
            trap_part=(
                "The follow-up question mutates the earlier premise into a different mechanism, reason, or condition "
                "before asking the model to continue the thread."
            ),
            expected_behavior=(
                "The model should identify the mutated premise, correct it using the established thread context, "
                "and answer only from the paper-supported premise."
            ),
        )
    )
    return candidates[:per_thread_limit]


def _base_plan(
    thread: dict,
    bridge_step: dict,
    source: dict,
    context: dict,
    metadata: dict,
    modality_pool: str,
    challenge_type: str,
    target_failure_mode: str,
    true_part: str,
    trap_part: str,
    expected_behavior: str,
) -> dict:
    return {
        "challenge_plan_id": "",
        "thread_challenge_plan_id": "",
        "challenge_scope": "thread",
        "thread_id": thread.get("thread_id"),
        "thread_type": thread.get("thread_type"),
        "anchor_step_ids": [
            turn.get("thread_turn_id")
            for turn in thread.get("planned_turns", [])
            if isinstance(turn, dict) and turn.get("thread_turn_id")
        ],
        "preferred_insert_after_step": bridge_step.get("thread_turn_id"),
        "challenge_type": challenge_type,
        "target_failure_mode": target_failure_mode,
        "source": source,
        "canonical_thread_context": context,
        "true_part": true_part,
        "trap_part": trap_part,
        "expected_behavior": expected_behavior,
        "evidence": _evidence_items(source["kc_ids"], source["edge_ids"], by_kc=None, by_edge=None),
        "requires_multimodal_input": bool(metadata.get("asset_references")),
        "modality_pool": modality_pool,
        "asset_references": metadata.get("asset_references", []),
        "metadata": metadata,
    }


def _evidence_items(
    kc_ids: list[str],
    edge_ids: list[str],
    by_kc: dict[str, dict] | None,
    by_edge: dict[str, dict] | None,
) -> list[dict]:
    out: list[dict] = []
    if by_edge:
        for edge_id in edge_ids:
            edge = by_edge.get(edge_id)
            if edge and str(edge.get("evidence") or "").strip():
                out.append(
                    {
                        "edge_id": edge_id,
                        "kc_ids": [edge.get("source"), edge.get("target")],
                        "text": _truncate(str(edge.get("evidence")).strip()),
                    }
                )
    if by_kc:
        for kc_id in kc_ids:
            kc = by_kc.get(kc_id)
            if not kc:
                continue
            evidence = kc.get("evidence", [])
            if isinstance(evidence, list):
                for item in evidence[:2]:
                    if isinstance(item, dict) and str(item.get("text") or "").strip():
                        out.append(
                            {
                                "kc_id": kc_id,
                                "text": _truncate(str(item.get("text")).strip()),
                                "span_id": item.get("span_id"),
                                "asset_id": kc.get("asset_id"),
                                "asset_type": kc.get("asset_type"),
                            }
                        )
            elif str(evidence or "").strip():
                out.append(
                    {
                        "kc_id": kc_id,
                        "text": _truncate(str(evidence).strip()),
                        "asset_id": kc.get("asset_id"),
                        "asset_type": kc.get("asset_type"),
                    }
                )
    return out


def _validate_plan(plan: dict, by_kc: dict[str, dict], by_edge: dict[str, dict]) -> None:
    if plan.get("challenge_type") not in THREAD_CHALLENGE_TYPES:
        raise ValueError(f"Invalid thread challenge type: {plan.get('challenge_type')!r}")
    source = plan.get("source")
    if not isinstance(source, dict):
        raise ValueError(f"Thread challenge plan {plan.get('challenge_plan_id')} has invalid source.")
    kc_ids = source.get("kc_ids", [])
    if not kc_ids or any(kc_id not in by_kc for kc_id in kc_ids):
        raise ValueError(f"Thread challenge plan {plan.get('challenge_plan_id')} references invalid KC IDs: {kc_ids}")
    edge_ids = source.get("edge_ids", [])
    if any(edge_id not in by_edge for edge_id in edge_ids):
        raise ValueError(f"Thread challenge plan {plan.get('challenge_plan_id')} references invalid edge IDs: {edge_ids}")
    evidence = _evidence_items(kc_ids, edge_ids, by_kc=by_kc, by_edge=by_edge)
    if not evidence:
        raise ValueError(f"Thread challenge plan {plan.get('challenge_plan_id')} must include evidence.")
    plan["evidence"] = evidence
    if not plan.get("canonical_thread_context"):
        raise ValueError(f"Thread challenge plan {plan.get('challenge_plan_id')} lacks canonical_thread_context.")
    if not plan.get("preferred_insert_after_step"):
        raise ValueError(f"Thread challenge plan {plan.get('challenge_plan_id')} lacks preferred_insert_after_step.")


def _ids_from_steps(steps: list[dict]) -> list[str]:
    return _ordered_unique([kc_id for step in steps for kc_id in (step.get("target_kc_ids", []) or [])])


def _valid_kc_ids(kc_ids: list[str], by_kc: dict[str, dict]) -> list[str]:
    return [kc_id for kc_id in _ordered_unique(kc_ids) if kc_id in by_kc]


def _context_text(steps: list[dict], kc_ids: list[str], by_kc: dict[str, dict]) -> str:
    expected = " ".join(str(step.get("expected_reasoning") or "").strip() for step in steps if step.get("expected_reasoning")).strip()
    if expected:
        return expected
    claims = [str(by_kc[kc_id].get("full_claim") or "").strip() for kc_id in kc_ids if str(by_kc[kc_id].get("full_claim") or "").strip()]
    return " ".join(claims).strip()


def _bridge_context_text(bridge_step: dict, edge_ids: list[str], by_edge: dict[str, dict], by_kc: dict[str, dict]) -> str:
    expected = str(bridge_step.get("expected_reasoning") or "").strip()
    if expected:
        return expected
    edge_bits = []
    for edge_id in edge_ids:
        edge = by_edge.get(edge_id)
        if not edge:
            continue
        source = by_kc.get(edge.get("source"), {}).get("full_claim", edge.get("source"))
        target = by_kc.get(edge.get("target"), {}).get("full_claim", edge.get("target"))
        edge_bits.append(f"{source} {edge.get('relation')} {target}")
    return "; ".join(edge_bits).strip()


def _synthetic_history(context: dict) -> dict:
    history = (
        "Earlier in the discussion:\n"
        f"- We established this premise: {context['premise']}\n"
        f"- We discussed this evidence: {context['evidence']}\n"
        f"- We connected them cautiously as follows: {context['bridge']}\n"
        "- We did not establish any exclusive cause, reversed relation, or mutated premise beyond this context."
    )
    return {
        "thread_context_used": True,
        "history_text": history,
        "source": "canonical_thread_context",
    }


def _ordered_unique(items: list) -> list:
    out = []
    seen = set()
    for item in items:
        if item is None or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _count_by_type(plans: list[dict]) -> dict[str, int]:
    counts = {challenge_type: 0 for challenge_type in sorted(THREAD_CHALLENGE_TYPES)}
    for plan in plans:
        counts[plan["challenge_type"]] = counts.get(plan["challenge_type"], 0) + 1
    return counts


def _count_by_modality_pool(plans: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for plan in plans:
        pool = str(plan.get("modality_pool") or "text")
        counts[pool] = counts.get(pool, 0) + 1
    return dict(sorted(counts.items()))


def _truncate(text: str) -> str:
    limit = _env_positive_int("THREAD_CHALLENGE_PLAN_EVIDENCE_MAX_CHARS", 1200)
    if len(text) <= limit:
        return text
    return text[:limit].rstrip()


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_positive_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer, got {raw!r}.") from exc
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value}.")
    return value
