from __future__ import annotations

import copy
import os

from src.progress import log


def select_active_kcs(kc_bank: dict, macro_spine: dict) -> dict:
    nodes = kc_bank.get("kc_nodes", [])
    if not nodes:
        raise ValueError("Cannot select Active KCs from an empty KC Bank.")

    active_target = _bounded_env_int("ACTIVE_KC_TARGET", 30, 1, 500)
    per_macro_min = _bounded_env_int("ACTIVE_KC_MIN_PER_MACRO", 2, 1, active_target)
    critical_macro_min = _bounded_env_int("ACTIVE_KC_CRITICAL_MIN_PER_MACRO", 3, per_macro_min, active_target)
    threshold = _env_float("ACTIVE_KC_THRESHOLD", 0.65, 0.0, 1.0)

    macro_nodes = macro_spine.get("macro_nodes", [])
    macro_by_id = {m["macro_id"]: m for m in macro_nodes if m.get("macro_id")}
    by_macro: dict[str, list[dict]] = {macro_id: [] for macro_id in macro_by_id}
    for node in nodes:
        macro_id = node.get("macro_id")
        if macro_id not in by_macro:
            raise ValueError(f"KC {node.get('kc_id')} references macro not in Macro Spine: {macro_id}")
        by_macro[macro_id].append(node)

    selected_ids: list[str] = []
    selected_set: set[str] = set()
    macro_active: dict[str, list[str]] = {macro_id: [] for macro_id in macro_by_id}

    threshold_candidates = [node for node in sorted(nodes, key=_score, reverse=True) if _score(node) >= threshold]
    for node in threshold_candidates:
        _add(node, selected_ids, selected_set, macro_active)

    for macro_id, macro in macro_by_id.items():
        candidates = sorted(by_macro.get(macro_id, []), key=_score, reverse=True)
        if not candidates:
            raise ValueError(f"Macro {macro_id} has no KC candidates in KC Bank.")
        required = critical_macro_min if macro.get("importance") == "critical" else per_macro_min
        if len(candidates) < required:
            raise ValueError(
                f"Macro {macro_id} has only {len(candidates)} KC candidates; required at least {required}."
            )
        current = len(macro_active.get(macro_id, []))
        for node in candidates:
            if current >= required:
                break
            _add(node, selected_ids, selected_set, macro_active)
            current = len(macro_active.get(macro_id, []))

    if len(selected_ids) > active_target:
        selected_ids = _truncate_preserving_macro_minimum(
            selected_ids,
            {node["kc_id"]: node for node in nodes},
            macro_by_id,
            per_macro_min,
            critical_macro_min,
            active_target,
        )
        selected_set = set(selected_ids)
        macro_active = {macro_id: [] for macro_id in macro_by_id}
        for kc_id in selected_ids:
            node = next(n for n in nodes if n["kc_id"] == kc_id)
            macro_active[node["macro_id"]].append(kc_id)

    active_nodes = []
    for node in nodes:
        active = node["kc_id"] in selected_set
        node["flags"]["active_for_question_generation"] = active
        node["flags"]["active_for_core_metrics"] = active
        if active:
            active_nodes.append(copy.deepcopy(node))

    payload = {
        "paper_id": kc_bank.get("paper_id"),
        "active_kc_ids": selected_ids,
        "selection_policy": {
            "active_kc_target": active_target,
            "active_kc_threshold": threshold,
            "min_per_macro": per_macro_min,
            "critical_min_per_macro": critical_macro_min,
        },
        "macro_active_kcs": macro_active,
        "kc_nodes": active_nodes,
    }
    log("Active KCs selected", active_kcs=len(selected_ids), macros=len(macro_active))
    return payload


def _add(
    node: dict,
    selected_ids: list[str],
    selected_set: set[str],
    macro_active: dict[str, list[str]],
) -> None:
    kc_id = node["kc_id"]
    if kc_id in selected_set:
        return
    selected_ids.append(kc_id)
    selected_set.add(kc_id)
    macro_active.setdefault(node["macro_id"], []).append(kc_id)


def _score(node: dict) -> float:
    return float(node.get("importance_scores", node.get("scores", {})).get("final_importance_score", 0.0))


def _truncate_preserving_macro_minimum(
    selected_ids: list[str],
    by_id: dict[str, dict],
    macro_by_id: dict[str, dict],
    per_macro_min: int,
    critical_macro_min: int,
    active_target: int,
) -> list[str]:
    minimum_total = sum(
        critical_macro_min if macro.get("importance") == "critical" else per_macro_min
        for macro in macro_by_id.values()
    )
    if active_target < minimum_total:
        raise ValueError(
            f"ACTIVE_KC_TARGET={active_target} is smaller than required macro coverage minimum {minimum_total}."
        )
    locked: list[str] = []
    locked_set: set[str] = set()
    for macro_id, macro in macro_by_id.items():
        required = critical_macro_min if macro.get("importance") == "critical" else per_macro_min
        macro_candidates = sorted(
            [by_id[kc_id] for kc_id in selected_ids if by_id[kc_id].get("macro_id") == macro_id],
            key=_score,
            reverse=True,
        )
        for node in macro_candidates[:required]:
            locked.append(node["kc_id"])
            locked_set.add(node["kc_id"])
    rest = [
        kc_id
        for kc_id in sorted(selected_ids, key=lambda item: _score(by_id[item]), reverse=True)
        if kc_id not in locked_set
    ]
    return locked + rest[: max(0, active_target - len(locked))]


def _env_float(name: str, default: float, minimum: float, maximum: float) -> float:
    raw = os.getenv(name)
    try:
        value = float(raw) if raw is not None and raw.strip() else default
    except ValueError:
        value = default
    return max(minimum, min(maximum, value))


def _bounded_env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    raw = os.getenv(name)
    try:
        value = int(raw) if raw is not None and raw.strip() else default
    except ValueError:
        value = default
    return max(minimum, min(maximum, value))
