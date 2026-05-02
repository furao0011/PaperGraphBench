from __future__ import annotations

import copy
import os

from src.progress import log


def select_active_kcs(kc_bank: dict, macro_spine: dict) -> dict:
    nodes = kc_bank.get("kc_nodes", [])
    if not nodes:
        raise ValueError("Cannot select Active KCs from an empty KC Bank.")

    active_target = _bounded_env_int("ACTIVE_KC_TARGET", 30, 1, 500)
    active_min = _bounded_env_int("ACTIVE_KC_MIN", 18, 1, active_target)
    active_max = _bounded_env_int("ACTIVE_KC_MAX", 40, active_target, 500)
    per_macro_min = _bounded_env_int("ACTIVE_KC_PER_MACRO_MIN", 2, 1, active_max)
    critical_macro_min = _bounded_env_int("ACTIVE_KC_CRITICAL_MACRO_MIN", 3, per_macro_min, active_max)

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

    for macro_id, macro in macro_by_id.items():
        candidates = sorted(by_macro.get(macro_id, []), key=_score, reverse=True)
        if not candidates:
            raise ValueError(f"Macro {macro_id} has no KC candidates in KC Bank.")
        required = critical_macro_min if macro.get("importance") == "critical" else per_macro_min
        if len(candidates) < required:
            raise ValueError(
                f"Macro {macro_id} has only {len(candidates)} KC candidates; required at least {required}."
            )
        for node in candidates[:required]:
            _add(node, selected_ids, selected_set, macro_active)

    fill_limit = min(active_target, active_max)
    for node in sorted(nodes, key=_score, reverse=True):
        if len(selected_ids) >= fill_limit:
            break
        _add(node, selected_ids, selected_set, macro_active)

    if len(selected_ids) < active_min:
        raise ValueError(f"Active KC selection produced {len(selected_ids)} KCs; required at least {active_min}.")

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
            "active_kc_min": active_min,
            "active_kc_max": active_max,
            "per_macro_min": per_macro_min,
            "critical_macro_min": critical_macro_min,
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
    return float(node.get("scores", {}).get("final_importance_score", 0.0))


def _bounded_env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    raw = os.getenv(name)
    try:
        value = int(raw) if raw is not None and raw.strip() else default
    except ValueError:
        value = default
    return max(minimum, min(maximum, value))

