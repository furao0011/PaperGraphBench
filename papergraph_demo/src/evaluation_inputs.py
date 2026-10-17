from __future__ import annotations

import json
from pathlib import Path

from src.artifact_paths import resolve_graph_path
from src.multimodal_question_assets import (
    attach_asset_references_to_questions,
    load_multimodal_asset_index,
    normalize_question_asset_paths,
)
from src.question_generator import normalize_question_bundle


def load_kc_bank(graph: dict, base_dir: Path) -> dict:
    diagnostics = graph.get("diagnostics") or {}
    signature = diagnostics.get("graph_signature") or {}
    source = diagnostics.get("master_graph_kc_source") or signature.get("master_graph_kc_source")
    if source == "bank":
        # Released master graphs explicitly embed the whole bank. Intermediate
        # cache files are not a runtime dependency for these self-contained graphs.
        _validate_embedded_bank(graph)
        return {"paper_id": graph.get("paper_id"), "kc_nodes": graph["kc_nodes"], "source": "master_graph.kc_nodes"}
    raw_path = graph.get("kc_bank_path")
    if not raw_path:
        if source == "active":
            raise ValueError("An active-only graph requires kc_bank_path for full-bank verification.")
        return {"paper_id": graph.get("paper_id"), "kc_nodes": graph.get("kc_nodes", [])}
    path = resolve_graph_path(raw_path, base_dir, graph)
    if not path.exists():
        raise FileNotFoundError(f"KC Bank not found for claim verification: {path}")
    bank = json.loads(path.read_text(encoding="utf-8"))
    if bank.get("paper_id") != graph.get("paper_id"):
        raise ValueError(f"KC Bank paper id does not match graph: {path}")
    return bank


def _validate_embedded_bank(graph: dict) -> None:
    nodes = graph.get("kc_nodes")
    if not isinstance(nodes, list) or not nodes:
        raise ValueError("A full-bank graph must embed a nonempty kc_nodes list.")
    ids = [node.get("kc_id") for node in nodes if isinstance(node, dict)]
    if len(ids) != len(nodes) or not all(ids) or len(set(ids)) != len(ids):
        raise ValueError("Embedded KC Bank contains invalid or duplicate kc_id values.")
    macro_ids = {kc_id for macro in graph.get("macro_nodes", []) for kc_id in macro.get("kc_ids", [])}
    if macro_ids and macro_ids != set(ids):
        raise ValueError("Embedded KC Bank does not match macro KC membership.")
    macros = graph.get("macro_nodes", [])
    if macros and all("bank_kc_count" in macro for macro in macros):
        if sum(int(macro["bank_kc_count"]) for macro in macros) != len(nodes):
            raise ValueError("Embedded KC Bank size differs from declared macro bank_kc_count.")
    signature = (graph.get("diagnostics") or {}).get("graph_signature") or {}
    if "graph_kc_ids" in signature and set(signature["graph_kc_ids"]) != set(ids):
        raise ValueError("Embedded KC Bank differs from its graph signature.")
    for node in nodes:
        if not str(node.get("full_claim") or node.get("claim") or "").strip() or not node.get("evidence"):
            raise ValueError(f"Embedded KC {node.get('kc_id')} lacks claim/evidence required for verification.")


def repair_questions_for_graph(graph: dict, questions: dict, base_dir: Path | None = None) -> dict:
    base_dir = Path(base_dir) if base_dir is not None else Path(__file__).resolve().parent.parent
    normalized = normalize_question_bundle(graph, questions)
    for key in (
        "challenge_questions",
        "thread_challenge_questions",
        "challenge_scheduler_config",
        "challenge_questions_filtered_path",
        "challenge_solver_trials_path",
        "challenge_filter_summary",
        "thread_challenge_plans_path",
        "thread_challenge_plan_summary",
        "thread_challenge_questions_raw_path",
        "thread_challenge_questions_filtered_path",
        "thread_challenge_solver_trials_path",
        "thread_challenge_filter_summary",
    ):
        if key in questions:
            normalized[key] = questions[key]
    asset_index = load_multimodal_asset_index(graph, base_dir)
    by_kc = {kc["kc_id"]: kc for kc in graph.get("kc_nodes", [])}
    for key, value in list(normalized.items()):
        if not isinstance(value, list) or not key.endswith(("questions", "question_seeds")):
            continue
        if key not in {"challenge_questions", "thread_challenge_questions"}:
            value = attach_asset_references_to_questions(value, by_kc, asset_index)
        normalized[key] = [normalize_question_asset_paths(question, graph, base_dir) for question in value]
    return {
        "paper_id": graph.get("paper_id"),
        **normalized,
    }
