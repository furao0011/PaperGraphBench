"""Validate the public runtime artifacts before treating a stage as complete."""

from __future__ import annotations

from pathlib import Path

from src.evaluation_inputs import load_kc_bank, repair_questions_for_graph
from src.multimodal_question_assets import load_multimodal_asset_index
from src.paper_context import load_full_paper_text
from src.pipeline_plans import PROJECT_ROOT, paper_layout, require_json


def validate_graph_inputs(paper_id: str) -> dict:
    graph = require_json(paper_layout(paper_id).final("master_graph"), paper_id)
    if not isinstance(graph.get("kc_nodes"), list) or not graph["kc_nodes"]:
        raise ValueError(f"Graph for {paper_id} has no KC nodes.")
    if not isinstance(graph.get("macro_nodes"), list) or not graph["macro_nodes"]:
        raise ValueError(f"Graph for {paper_id} has no macro nodes.")
    base_dir = PROJECT_ROOT / "papergraph_demo"
    load_kc_bank(graph, base_dir)
    if not load_full_paper_text(graph, base_dir).strip():
        raise ValueError(f"Paper context for {paper_id} is empty.")
    load_multimodal_asset_index(graph, base_dir)
    return graph


def validate_question_inputs(paper_id: str, graph: dict | None = None) -> dict:
    graph = graph or validate_graph_inputs(paper_id)
    raw = require_json(paper_layout(paper_id).final("question_templates"), paper_id)
    questions = repair_questions_for_graph(graph, raw)
    main = questions.get("macro_main_questions") or questions.get("main_questions")
    if not isinstance(main, list) or not main:
        raise ValueError(f"Question package for {paper_id} has no macro questions.")
    ids = [item.get("question_id") for item in main if isinstance(item, dict)]
    if len(ids) != len(main) or not all(ids) or len(set(ids)) != len(ids):
        raise ValueError(f"Question package for {paper_id} has invalid macro question IDs.")
    return questions


def completed_evaluation(path: Path, paper_id: str, model: str) -> bool:
    if not path.is_file() or not (path.parent / "dialogue_trajectory.json").is_file():
        return False
    report = require_json(path, paper_id)
    trajectory = require_json(path.parent / "dialogue_trajectory.json", paper_id)
    if report.get("target_model") != model or trajectory.get("target_model") != model:
        raise ValueError(f"Model identity mismatch in {path.parent}")
    summary = report.get("summary") or {}
    if summary.get("evaluation_status") != "completed":
        return False
    turns = trajectory.get("turns")
    if not isinstance(turns, list) or not turns:
        raise ValueError(f"Completed evaluation has no dialogue turns: {path.parent}")
    if summary.get("total_turns") != len(turns):
        raise ValueError(f"Report/trajectory turn counts differ: {path.parent}")
    if summary.get("failed") is not False:
        raise ValueError(f"Completed report does not declare failed=false: {path}")
    ids = [turn.get("turn_id") for turn in turns if isinstance(turn, dict)]
    if len(ids) != len(turns) or not all(ids) or len(set(ids)) != len(ids):
        raise ValueError(f"Invalid or duplicate dialogue turn IDs: {path.parent}")
    return True
