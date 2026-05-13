from __future__ import annotations

import os

from src.multimodal_question_assets import asset_references_for_kcs


CHALLENGE_TYPES = {
    "overclaim_challenge",
    "wrong_relation_challenge",
    "false_premise_challenge",
}


def build_challenge_plans(graph: dict) -> dict:
    kc_nodes = graph.get("kc_nodes", [])
    reasoning_edges = graph.get("reasoning_edges", [])
    reasoning_threads = graph.get("reasoning_threads", [])
    active_ids = {
        kc["kc_id"]
        for kc in kc_nodes
        if kc.get("kc_id") and kc.get("flags", {}).get("active_for_question_generation")
    }
    if not active_ids:
        active_ids = set(graph.get("active_kc_ids", []))
    if not active_ids:
        raise ValueError("Challenge Plan Builder requires active_kc_ids or active KC flags in master_graph.json.")

    by_kc = {
        kc["kc_id"]: kc
        for kc in kc_nodes
        if kc.get("kc_id")
    }
    by_edge = {
        edge["edge_id"]: edge
        for edge in reasoning_edges
        if edge.get("edge_id")
    }
    multimodal_ids = {
        kc["kc_id"]
        for kc in kc_nodes
        if kc.get("kc_id") and bool(kc.get("modality", {}).get("is_multimodal"))
    }
    text_active_ids = {kc_id for kc_id in active_ids if kc_id not in multimodal_ids}
    per_type_limit = _env_positive_int(
        "CHALLENGE_PLAN_POOL_PER_TYPE_LIMIT",
        _env_positive_int("CHALLENGE_PLAN_PER_TYPE_LIMIT", 12),
    )
    total_target = _env_positive_int(
        "CHALLENGE_PLAN_POOL_TARGET",
        _env_positive_int("CHALLENGE_PLAN_TARGET", 30),
    )

    by_type = {
        "overclaim_challenge": _dedupe_plans(_overclaim_plans(by_kc, text_active_ids, per_type_limit))[:per_type_limit],
        "wrong_relation_challenge": _dedupe_plans(
            _wrong_relation_plans_from_threads(reasoning_threads, by_edge, by_kc, per_type_limit)
            + _wrong_relation_plans_from_edges(by_edge, by_kc, text_active_ids, per_type_limit)
        )[:per_type_limit],
        "false_premise_challenge": _dedupe_plans(_false_premise_plans(by_kc, text_active_ids, per_type_limit))[:per_type_limit],
    }
    text_plans = _with_modality_pool(_balanced_plan_order(by_type, total_target), "text")
    multimodal_limit = _env_nonnegative_int("MULTIMODAL_CHALLENGE_PLAN_LIMIT", 30)
    multimodal_plans = _with_modality_pool(
        _multimodal_challenge_plans(by_kc, by_edge, multimodal_ids, multimodal_limit),
        "multimodal",
    )
    if multimodal_ids and not multimodal_plans:
        raise RuntimeError("Multimodal KCs exist, but multimodal challenge plan generation produced no plans.")
    unique = text_plans + multimodal_plans
    for idx, plan in enumerate(unique, start=1):
        plan["challenge_plan_id"] = f"CHP_{idx:04d}"
        _validate_plan(plan, by_kc, by_edge)

    return {
        "paper_id": graph.get("paper_id", "unknown"),
        "schema_version": "v2",
        "plan_builder": "deterministic_v1",
        "source_graph_signature": graph.get("diagnostics", {}).get("graph_signature"),
        "challenge_plans": unique,
        "summary": {
            "plan_count": len(unique),
            "by_type": _count_by_type(unique),
            "by_modality_pool": _count_by_modality_pool(unique),
            "text_plan_count": len(text_plans),
            "multimodal_plan_count": len(multimodal_plans),
            "multimodal_plan_limit": multimodal_limit,
        },
    }


def _overclaim_plans(by_kc: dict[str, dict], active_ids: set[str], limit: int) -> list[dict]:
    candidates = []
    for kc_id in sorted(active_ids, key=_kc_sort_key):
        kc = by_kc.get(kc_id)
        if not kc:
            continue
        scope = kc.get("scope") if isinstance(kc.get("scope"), dict) else {}
        strength = str(kc.get("claim_strength", "")).strip()
        generality = str(scope.get("generality", "")).strip()
        if not strength and not scope:
            continue
        if strength == "explicit" and generality in {"broad_paper_claim", "not_specified"}:
            continue
        candidates.append(kc)
    candidates = sorted(candidates, key=_kc_rank_key, reverse=True)

    plans = []
    for kc in candidates[:limit]:
        claim = kc.get("full_claim", "")
        scope = kc.get("scope") if isinstance(kc.get("scope"), dict) else {}
        strength = str(kc.get("claim_strength", "")).strip() or "not_specified"
        generality = str(scope.get("generality", "")).strip() or "the stated Storybench context"
        dataset = str(scope.get("dataset", "")).strip() or "the stated dataset or setting"
        plans.append(
            {
                "challenge_plan_id": "",
                "challenge_type": "overclaim_challenge",
                "source": {
                    "kc_ids": [kc["kc_id"]],
                    "edge_ids": [],
                    "thread_id": None,
                    "thread_turn_id": None,
                    "macro_ids": [kc.get("macro_id")],
                },
                "true_part": claim,
                "trap_part": (
                    f"This claim can be treated as a broadly proven result beyond {generality} "
                    f"and beyond {dataset}."
                ),
                "expected_behavior": (
                    f"The model should state the Storybench's claim while respecting claim_strength={strength} "
                    f"and the recorded scope={scope}."
                ),
                "target_failure_mode": "overclaim",
                "evidence": _kc_evidence_items([kc]),
                "metadata": {
                    "claim_strength": strength,
                    "scope": scope,
                },
            }
        )
    return plans


def _wrong_relation_plans_from_threads(
    threads: list[dict],
    by_edge: dict[str, dict],
    by_kc: dict[str, dict],
    limit: int,
) -> list[dict]:
    plans = []
    for thread in threads:
        thread_id = thread.get("thread_id")
        for turn in thread.get("planned_turns", []):
            if turn.get("role") not in {"bridge_reasoning", "review_consistency"}:
                continue
            edge_ids = [
                edge_id
                for edge_id in turn.get("supporting_edge_ids", [])
                if edge_id in by_edge
            ]
            if not edge_ids:
                continue
            edges = [by_edge[edge_id] for edge_id in edge_ids]
            kc_ids = _ordered_unique(
                [
                    kc_id
                    for edge in edges
                    for kc_id in (edge.get("source"), edge.get("target"))
                    if kc_id in by_kc
                ]
            )
            if len(kc_ids) < 2:
                continue
            plans.append(
                _wrong_relation_plan(
                    edges=edges,
                    by_kc=by_kc,
                    thread_id=thread_id,
                    thread_turn_id=turn.get("thread_turn_id"),
                    expected_behavior=turn.get("expected_reasoning", ""),
                )
            )
            if len(plans) >= limit:
                return plans
    return plans


def _wrong_relation_plans_from_edges(
    by_edge: dict[str, dict],
    by_kc: dict[str, dict],
    active_ids: set[str],
    limit: int,
) -> list[dict]:
    plans = []
    used_edge_ids = set()
    for edge in sorted(by_edge.values(), key=lambda item: _edge_sort_key(str(item.get("edge_id", "")))):
        edge_id = edge.get("edge_id")
        if edge_id in used_edge_ids:
            continue
        if edge.get("source") not in active_ids or edge.get("target") not in active_ids:
            continue
        if edge.get("source") not in by_kc or edge.get("target") not in by_kc:
            continue
        plans.append(
            _wrong_relation_plan(
                edges=[edge],
                by_kc=by_kc,
                thread_id=None,
                thread_turn_id=None,
                expected_behavior="The model should preserve the verified edge direction and relation.",
            )
        )
        used_edge_ids.add(edge_id)
        if len(plans) >= limit:
            break
    return plans


def _wrong_relation_plan(
    edges: list[dict],
    by_kc: dict[str, dict],
    thread_id: str | None,
    thread_turn_id: str | None,
    expected_behavior: str,
) -> dict:
    edge_ids = [edge["edge_id"] for edge in edges]
    kc_ids = _ordered_unique(
        [
            kc_id
            for edge in edges
            for kc_id in (edge.get("source"), edge.get("target"))
            if kc_id in by_kc
        ]
    )
    macro_ids = _ordered_unique([by_kc[kc_id].get("macro_id") for kc_id in kc_ids if by_kc[kc_id].get("macro_id")])
    asset_ids = _ordered_unique([by_kc[kc_id].get("asset_id") for kc_id in kc_ids if by_kc[kc_id].get("asset_id")])
    asset_refs = asset_references_for_kcs([by_kc[kc_id] for kc_id in kc_ids if kc_id in by_kc])
    edge_descriptions = [
        f"{edge.get('source')} {edge.get('relation')} {edge.get('target')}"
        for edge in edges
    ]
    return {
        "challenge_plan_id": "",
        "challenge_type": "wrong_relation_challenge",
        "source": {
            "kc_ids": kc_ids,
            "edge_ids": edge_ids,
            "thread_id": thread_id,
            "thread_turn_id": thread_turn_id,
            "macro_ids": macro_ids,
            "asset_ids": asset_ids,
        },
        "true_part": "Verified relation chain: " + "; ".join(edge_descriptions) + ".",
        "trap_part": "The answer reverses, weakens, or replaces the verified relation chain.",
        "expected_behavior": expected_behavior or "The model should keep the verified relation direction and relation type.",
        "target_failure_mode": "wrong_relation",
        "evidence": _edge_evidence_items(edges, by_kc),
        "metadata": {
            "relations": [edge.get("relation") for edge in edges],
            "edge_source_layers": [edge.get("source_layer") for edge in edges],
            "asset_references": asset_refs,
        },
    }


def _false_premise_plans(by_kc: dict[str, dict], active_ids: set[str], limit: int) -> list[dict]:
    plans = []
    for kc in sorted((by_kc[kc_id] for kc_id in active_ids if kc_id in by_kc), key=_kc_rank_key, reverse=True):
        forbidden = kc.get("forbidden_claims", [])
        if not isinstance(forbidden, list):
            continue
        for item in forbidden:
            if not isinstance(item, dict):
                continue
            false_claim = str(item.get("claim", "")).strip()
            if not false_claim:
                continue
            plans.append(
                {
                    "challenge_plan_id": "",
                    "challenge_type": "false_premise_challenge",
                    "source": {
                        "kc_ids": [kc["kc_id"]],
                        "edge_ids": [],
                        "thread_id": None,
                        "thread_turn_id": None,
                        "macro_ids": [kc.get("macro_id")],
                    },
                    "true_part": kc.get("full_claim", ""),
                    "trap_part": false_claim,
                    "expected_behavior": item.get("why_wrong")
                    or "The model should reject the false premise and restate the Storybench-supported claim.",
                    "target_failure_mode": "false_premise",
                    "evidence": _kc_evidence_items([kc]),
                    "metadata": {
                        "forbidden_claim_id": item.get("claim_id"),
                        "false_premise_type": item.get("type"),
                        "severity": item.get("severity"),
                    },
                }
            )
            if len(plans) >= limit:
                return plans
    return plans


def _multimodal_challenge_plans(
    by_kc: dict[str, dict],
    by_edge: dict[str, dict],
    multimodal_ids: set[str],
    limit: int,
) -> list[dict]:
    plans = []
    multimodal_kcs = sorted(
        (by_kc[kc_id] for kc_id in multimodal_ids if kc_id in by_kc),
        key=_kc_rank_key,
        reverse=True,
    )
    figure_plan_min = _env_nonnegative_int("MULTIMODAL_CHALLENGE_FIGURE_PLAN_MIN", 0)
    if figure_plan_min > limit:
        raise ValueError(
            f"MULTIMODAL_CHALLENGE_FIGURE_PLAN_MIN={figure_plan_min} exceeds MULTIMODAL_CHALLENGE_PLAN_LIMIT={limit}."
        )
    figure_kcs = [kc for kc in multimodal_kcs if str(kc.get("asset_type") or "").strip().lower() == "figure"]
    if figure_plan_min and len(figure_kcs) < figure_plan_min:
        raise RuntimeError(
            f"Multimodal challenge plan quota requires {figure_plan_min} figure KCs, but only {len(figure_kcs)} are available."
        )
    for kc in figure_kcs[:figure_plan_min]:
        if _limit_reached(plans, limit):
            return plans
        plan = _multimodal_false_premise_plan(kc) or _multimodal_overclaim_plan(kc)
        plans.append(plan)
    used_kc_ids = {plan.get("source", {}).get("kc_ids", [None])[0] for plan in plans}
    for kc in multimodal_kcs:
        if kc.get("kc_id") in used_kc_ids:
            continue
        if _limit_reached(plans, limit):
            return plans
        plan = _multimodal_false_premise_plan(kc) or _multimodal_overclaim_plan(kc)
        plans.append(plan)
    for edge in sorted(by_edge.values(), key=lambda item: _edge_sort_key(str(item.get("edge_id", "")))):
        if _limit_reached(plans, limit):
            return plans
        if edge.get("source") not in multimodal_ids and edge.get("target") not in multimodal_ids:
            continue
        source = by_kc.get(edge.get("source"))
        target = by_kc.get(edge.get("target"))
        if not source or not target:
            continue
        plans.append(
            _wrong_relation_plan(
                edges=[edge],
                by_kc=by_kc,
                thread_id=None,
                thread_turn_id=None,
                expected_behavior="The model should preserve the verified relation while grounding any figure/table claim in the provided asset.",
            )
        )
    return plans


def _multimodal_false_premise_plan(kc: dict) -> dict | None:
    false_claim = ""
    why_wrong = ""
    for item in kc.get("asset_possible_misreadings", []):
        if isinstance(item, dict):
            false_claim = str(item.get("claim", "")).strip()
            why_wrong = str(item.get("why_wrong", "")).strip()
        else:
            false_claim = str(item or "").strip()
            why_wrong = "The claim is listed as a possible misreading of the asset."
        if false_claim:
            break
    if not false_claim:
        for item in kc.get("forbidden_claims", []):
            if not isinstance(item, dict):
                continue
            false_claim = str(item.get("claim", "")).strip()
            why_wrong = str(item.get("why_wrong", "")).strip()
            if false_claim:
                break
    if not false_claim:
        return None
    return {
        "challenge_plan_id": "",
        "challenge_type": "false_premise_challenge",
        "source": _kc_source(kc),
        "true_part": kc.get("full_claim", ""),
        "trap_part": false_claim,
        "expected_behavior": why_wrong or "The model should reject the visual/table misreading and restate the asset-supported claim.",
        "target_failure_mode": "false_premise",
        "evidence": _kc_evidence_items([kc]),
        "metadata": _multimodal_metadata(kc, {"false_premise_source": "asset_possible_misreading"}),
    }


def _multimodal_overclaim_plan(kc: dict) -> dict:
    asset_type = str(kc.get("asset_type") or "asset").strip()
    scope = kc.get("scope") if isinstance(kc.get("scope"), dict) else {}
    return {
        "challenge_plan_id": "",
        "challenge_type": "overclaim_challenge",
        "source": _kc_source(kc),
        "true_part": kc.get("full_claim", ""),
        "trap_part": (
            f"The {asset_type} should be treated as proving a broader claim beyond the recorded asset scope "
            f"and beyond the Storybench context."
        ),
        "expected_behavior": (
            "The model should answer using the attached figure/table and the prepared asset description, "
            f"while respecting the recorded scope={scope}."
        ),
        "target_failure_mode": "overclaim",
        "evidence": _kc_evidence_items([kc]),
        "metadata": _multimodal_metadata(kc, {"claim_strength": kc.get("claim_strength"), "scope": scope}),
    }


def _validate_plan(plan: dict, by_kc: dict[str, dict], by_edge: dict[str, dict]) -> None:
    challenge_type = plan.get("challenge_type")
    if challenge_type not in CHALLENGE_TYPES:
        raise ValueError(f"Invalid challenge_type: {challenge_type!r}")
    if not plan.get("true_part") or not plan.get("trap_part") or not plan.get("expected_behavior"):
        raise ValueError(f"Challenge plan {plan.get('challenge_plan_id')} is missing true/trap/expected behavior.")
    source = plan.get("source")
    if not isinstance(source, dict):
        raise ValueError(f"Challenge plan {plan.get('challenge_plan_id')} has invalid source.")
    kc_ids = source.get("kc_ids", [])
    edge_ids = source.get("edge_ids", [])
    if not kc_ids or any(kc_id not in by_kc for kc_id in kc_ids):
        raise ValueError(f"Challenge plan {plan.get('challenge_plan_id')} references invalid KC IDs: {kc_ids}")
    if any(edge_id not in by_edge for edge_id in edge_ids):
        raise ValueError(f"Challenge plan {plan.get('challenge_plan_id')} references invalid edge IDs: {edge_ids}")
    if not plan.get("evidence"):
        raise ValueError(f"Challenge plan {plan.get('challenge_plan_id')} must include evidence.")
    if challenge_type == "overclaim_challenge":
        metadata = plan.get("metadata", {})
        if not metadata.get("claim_strength") and not metadata.get("scope"):
            raise ValueError(f"Overclaim plan {plan.get('challenge_plan_id')} must bind claim_strength or scope.")
    if challenge_type == "wrong_relation_challenge" and not edge_ids:
        raise ValueError(f"Wrong-relation plan {plan.get('challenge_plan_id')} must bind edge_ids.")
    if challenge_type == "false_premise_challenge" and not plan.get("trap_part"):
        raise ValueError(f"False-premise plan {plan.get('challenge_plan_id')} must locate the false premise.")


def _kc_source(kc: dict) -> dict:
    return {
        "kc_ids": [kc["kc_id"]],
        "edge_ids": [],
        "thread_id": None,
        "thread_turn_id": None,
        "macro_ids": [kc.get("macro_id")],
        "asset_ids": [kc.get("asset_id")] if kc.get("asset_id") else [],
    }


def _multimodal_metadata(kc: dict, extra: dict | None = None) -> dict:
    metadata = {
        "modality_pool": "multimodal",
        "asset_references": asset_references_for_kcs([kc]),
    }
    if extra:
        metadata.update(extra)
    return metadata


def _kc_evidence_items(kcs: list[dict]) -> list[dict]:
    out = []
    for kc in kcs:
        evidence = kc.get("evidence", [])
        if isinstance(evidence, list):
            for item in evidence[:2]:
                if isinstance(item, dict) and str(item.get("text", "")).strip():
                    out.append(
                        {
                            "kc_id": kc.get("kc_id"),
                            "text": _truncate(str(item.get("text", "")).strip()),
                            "span_id": item.get("span_id"),
                            "asset_id": kc.get("asset_id"),
                            "asset_type": kc.get("asset_type"),
                        }
                    )
        elif str(evidence).strip():
            out.append(
                {
                    "kc_id": kc.get("kc_id"),
                    "text": _truncate(str(evidence).strip()),
                    "asset_id": kc.get("asset_id"),
                    "asset_type": kc.get("asset_type"),
                }
            )
        if bool(kc.get("modality", {}).get("is_multimodal")):
            basis = str(kc.get("asset_evidence_basis") or "").strip()
            if basis:
                out.append(
                    {
                        "kc_id": kc.get("kc_id"),
                        "text": _truncate(f"Asset evidence basis: {basis}"),
                        "asset_id": kc.get("asset_id"),
                        "asset_type": kc.get("asset_type"),
                    }
                )
    return out


def _edge_evidence_items(edges: list[dict], by_kc: dict[str, dict]) -> list[dict]:
    out = []
    for edge in edges:
        text = str(edge.get("evidence", "")).strip()
        if text:
            out.append(
                {
                    "edge_id": edge.get("edge_id"),
                    "kc_ids": [edge.get("source"), edge.get("target")],
                    "text": _truncate(text),
                }
            )
    for kc_id in _ordered_unique([kc_id for edge in edges for kc_id in (edge.get("source"), edge.get("target"))]):
        kc = by_kc.get(kc_id)
        if kc:
            out.extend(_kc_evidence_items([kc])[:1])
    return out


def _dedupe_plans(plans: list[dict]) -> list[dict]:
    out = []
    seen = set()
    for plan in plans:
        source = plan.get("source", {})
        key = (
            plan.get("challenge_type"),
            tuple(source.get("kc_ids", [])),
            tuple(source.get("edge_ids", [])),
            plan.get("trap_part"),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(plan)
    return out


def _with_modality_pool(plans: list[dict], pool: str) -> list[dict]:
    for plan in plans:
        plan["modality_pool"] = pool
        metadata = plan.setdefault("metadata", {})
        if isinstance(metadata, dict):
            metadata.setdefault("modality_pool", pool)
    return plans


def _balanced_plan_order(by_type: dict[str, list[dict]], total_target: int) -> list[dict]:
    order = [
        "overclaim_challenge",
        "wrong_relation_challenge",
        "false_premise_challenge",
    ]
    out = []
    index = 0
    while len(out) < total_target:
        added = False
        for challenge_type in order:
            plans = by_type.get(challenge_type, [])
            if index < len(plans):
                out.append(plans[index])
                added = True
                if len(out) >= total_target:
                    break
        if not added:
            break
        index += 1
    return out


def _count_by_type(plans: list[dict]) -> dict[str, int]:
    counts = {challenge_type: 0 for challenge_type in sorted(CHALLENGE_TYPES)}
    for plan in plans:
        counts[plan["challenge_type"]] = counts.get(plan["challenge_type"], 0) + 1
    return counts


def _count_by_modality_pool(plans: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for plan in plans:
        pool = str(plan.get("modality_pool") or plan.get("metadata", {}).get("modality_pool") or "text")
        counts[pool] = counts.get(pool, 0) + 1
    return dict(sorted(counts.items()))


def _limit_reached(plans: list[dict], limit: int) -> bool:
    return limit > 0 and len(plans) >= limit


def _ordered_unique(items: list) -> list:
    out = []
    seen = set()
    for item in items:
        if item is None or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _kc_rank_key(kc: dict) -> tuple[float, int]:
    scores = kc.get("importance_scores") or kc.get("scores") or {}
    return (float(scores.get("final_importance_score", 0.0)), -_kc_sort_key(kc.get("kc_id", ""))[0])


def _kc_sort_key(kc_id: str) -> tuple[int, str]:
    digits = "".join(ch for ch in str(kc_id) if ch.isdigit())
    return (int(digits) if digits else 10**9, str(kc_id))


def _edge_sort_key(edge_id: str) -> tuple[int, str]:
    digits = "".join(ch for ch in str(edge_id) if ch.isdigit())
    return (int(digits) if digits else 10**9, str(edge_id))


def _truncate(text: str) -> str:
    limit = _env_positive_int("CHALLENGE_PLAN_EVIDENCE_MAX_CHARS", 1200)
    if len(text) <= limit:
        return text
    return text[:limit].rstrip()


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


def _env_nonnegative_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a non-negative integer, got {raw!r}.") from exc
    if value < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {value}.")
    return value
