from __future__ import annotations

import os


def build_edge_coverage_report(
    paper_id: str,
    macro_spine: dict,
    kc_bank: dict,
    verified_edges: list[dict],
) -> dict:
    by_kc = {
        kc["kc_id"]: kc
        for kc in kc_bank.get("kc_nodes", [])
        if kc.get("kc_id")
    }
    macro_ids = [
        macro["macro_id"]
        for macro in macro_spine.get("macro_nodes", [])
        if macro.get("macro_id")
    ]
    if not macro_ids:
        raise ValueError("Edge coverage report requires non-empty macro_spine.macro_nodes.")

    macro_pairs = _macro_pairs(macro_spine)
    macro_pair_counts = {f"{source}->{target}": 0 for source, target, _ in macro_pairs}
    incident_counts = {macro_id: 0 for macro_id in macro_ids}
    kc_incident_counts = {kc_id: 0 for kc_id in by_kc}
    source_layer_counts: dict[str, int] = {}
    relation_counts: dict[str, int] = {}
    thread_pattern_counts: dict[str, int] = {}

    normalized_edges = []
    for edge in verified_edges:
        source = str(edge.get("source", "")).strip()
        target = str(edge.get("target", "")).strip()
        if source not in by_kc or target not in by_kc:
            raise ValueError(
                f"Verified edge {edge.get('edge_id')} references KC outside KC Bank: "
                f"{source!r}->{target!r}"
            )
        source_macro = str(edge.get("source_macro_id") or by_kc[source].get("macro_id") or "").strip()
        target_macro = str(edge.get("target_macro_id") or by_kc[target].get("macro_id") or "").strip()
        if source_macro not in incident_counts or target_macro not in incident_counts:
            raise ValueError(
                f"Verified edge {edge.get('edge_id')} references Macro outside Macro Spine: "
                f"{source_macro!r}->{target_macro!r}"
            )

        pair_key = f"{source_macro}->{target_macro}"
        if pair_key in macro_pair_counts:
            macro_pair_counts[pair_key] += 1
        incident_counts[source_macro] += 1
        incident_counts[target_macro] += 1
        kc_incident_counts[source] += 1
        kc_incident_counts[target] += 1

        source_layer = str(edge.get("source_layer") or "unknown").strip()
        relation = str(edge.get("relation") or "unknown").strip()
        source_layer_counts[source_layer] = source_layer_counts.get(source_layer, 0) + 1
        relation_counts[relation] = relation_counts.get(relation, 0) + 1
        thread_pattern = str(edge.get("thread_pattern") or "").strip()
        if thread_pattern:
            thread_pattern_counts[thread_pattern] = thread_pattern_counts.get(thread_pattern, 0) + 1
        normalized_edges.append((source, target))

    empty_macro_pairs = [
        {
            "macro_pair": f"{source}->{target}",
            "source": source,
            "target": target,
            "macro_edge_id": edge.get("edge_id"),
            "relation": edge.get("relation", ""),
            "description": edge.get("description", ""),
        }
        for source, target, edge in macro_pairs
        if macro_pair_counts.get(f"{source}->{target}", 0) == 0
    ]

    min_incident_edges = _env_non_negative_int("EDGE_COVERAGE_MIN_MACRO_INCIDENT_EDGES", 1)
    low_coverage_macros = [
        {
            "macro_id": macro_id,
            "incident_edge_count": incident_counts.get(macro_id, 0),
            "required_min_incident_edges": min_incident_edges,
        }
        for macro_id in macro_ids
        if incident_counts.get(macro_id, 0) < min_incident_edges
    ]

    isolated_kcs = [
        kc_id
        for kc_id, count in sorted(kc_incident_counts.items(), key=lambda item: _kc_sort_key(item[0]))
        if count == 0
    ]

    return {
        "paper_id": paper_id,
        "verified_edge_count": len(verified_edges),
        "macro_pair_edge_coverage": macro_pair_counts,
        "thread_pattern_edge_coverage": dict(sorted(thread_pattern_counts.items())),
        "edge_source_layer_counts": dict(sorted(source_layer_counts.items())),
        "edge_relation_counts": dict(sorted(relation_counts.items())),
        "empty_macro_pairs": empty_macro_pairs,
        "low_coverage_macros": low_coverage_macros,
        "kc_coverage": {
            "kc_count": len(by_kc),
            "connected_kc_count": len(by_kc) - len(isolated_kcs),
            "isolated_kc_count": len(isolated_kcs),
            "isolated_kc_ids": isolated_kcs,
        },
    }


def attach_reasoning_path_coverage(report: dict, reasoning_paths: list[dict]) -> dict:
    path_pattern_counts: dict[str, int] = {}
    supporting_edge_counts: dict[str, int] = {}
    paths_without_supporting_edges = []
    for path in reasoning_paths:
        path_id = str(path.get("path_id", "")).strip()
        pattern = str(path.get("pattern") or "unknown").strip()
        path_pattern_counts[pattern] = path_pattern_counts.get(pattern, 0) + 1
        supporting_edge_ids = [
            str(edge_id).strip()
            for edge_id in path.get("supporting_edge_ids", [])
            if str(edge_id).strip()
        ]
        if not supporting_edge_ids:
            paths_without_supporting_edges.append(path_id)
        for edge_id in supporting_edge_ids:
            supporting_edge_counts[edge_id] = supporting_edge_counts.get(edge_id, 0) + 1
    item = dict(report)
    item["reasoning_path_pattern_coverage"] = dict(sorted(path_pattern_counts.items()))
    item["reasoning_path_count"] = len(reasoning_paths)
    item["reasoning_path_supporting_edge_reuse"] = dict(sorted(supporting_edge_counts.items()))
    item["paths_without_supporting_edges"] = paths_without_supporting_edges
    item["thread_pattern_edge_coverage_note"] = (
        "This field counts only verified edges produced by the Thread candidate edge layer. "
        "Reasoning paths and Reasoning Threads are reported separately because they can be derived from lower-layer verified edges."
    )
    return item


def _macro_pairs(macro_spine: dict) -> list[tuple[str, str, dict]]:
    pairs = []
    seen = set()
    for idx, edge in enumerate(macro_spine.get("macro_edges", []), start=1):
        source = str(edge.get("source", "")).strip()
        target = str(edge.get("target", "")).strip()
        if not source or not target or source == target:
            continue
        key = (source, target)
        if key in seen:
            continue
        seen.add(key)
        item = dict(edge)
        item.setdefault("edge_id", f"ME{idx}")
        pairs.append((source, target, item))
    return pairs


def _env_non_negative_int(name: str, default: int) -> int:
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


def _kc_sort_key(kc_id: str) -> tuple[int, str]:
    digits = "".join(ch for ch in kc_id if ch.isdigit())
    return (int(digits) if digits else 10**9, kc_id)
