from __future__ import annotations

import json
from pathlib import Path

from src.question_generator import normalize_question_bundle


def load_kc_bank(graph: dict, base_dir: Path) -> dict:
    raw_path = graph.get("kc_bank_path")
    if not raw_path:
        return {"paper_id": graph.get("paper_id"), "kc_nodes": graph.get("kc_nodes", [])}
    path = Path(raw_path)
    if not path.is_absolute():
        path = base_dir / path
    if not path.exists():
        raise FileNotFoundError(f"KC Bank not found for claim verification: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def repair_questions_for_graph(graph: dict, questions: dict) -> dict:
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
    return {
        "paper_id": graph.get("paper_id"),
        **normalized,
    }
