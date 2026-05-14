from __future__ import annotations

import json
import re
from pathlib import Path

from src.mermaid_exporter import export_final_state_mermaid
from src.reporter import build_report
from src.thread_scheduler import completed_thread_step_ids


def write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def append_turn(trajectory: dict, turn: dict) -> None:
    turns = trajectory.setdefault("turns", [])
    if turns:
        previous = turns[-1]
        previous["actual_next_turn_id"] = turn.get("turn_id")
        previous["actual_next_question_id"] = turn.get("question_id")
        previous["actual_next_question_type"] = turn.get("question_type")
        previous["actual_next_action"] = action_for_question_type(turn.get("question_type", ""))
    turns.append(turn)


def reconcile_actual_transitions(trajectory: dict) -> None:
    turns = trajectory.get("turns", [])
    for idx, turn in enumerate(turns):
        if idx + 1 >= len(turns):
            turn.pop("actual_next_turn_id", None)
            turn.pop("actual_next_question_id", None)
            turn.pop("actual_next_question_type", None)
            turn.pop("actual_next_action", None)
            continue
        nxt = turns[idx + 1]
        turn["actual_next_turn_id"] = nxt.get("turn_id")
        turn["actual_next_question_id"] = nxt.get("question_id")
        turn["actual_next_question_type"] = nxt.get("question_type")
        turn["actual_next_action"] = action_for_question_type(nxt.get("question_type", ""))


def load_eval_checkpoint(
    checkpoint_path: Path,
    graph: dict,
    target_model: str,
    ensure_defaults,
    rebuild_turn_counts,
) -> tuple[dict, dict, int, set[str], set[str]] | None:
    if not checkpoint_path.exists():
        return None
    data = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    if data.get("paper_id") != graph.get("paper_id"):
        return None
    trajectory = data.get("trajectory")
    eval_state = data.get("eval_state")
    if not isinstance(trajectory, dict) or not isinstance(eval_state, dict):
        return None
    trajectory.setdefault("paper_id", graph.get("paper_id", "unknown"))
    trajectory.setdefault("target_model", target_model)
    trajectory.setdefault("turns", [])
    turn_no = int(data.get("turn_no") or max((_turn_number(t.get("turn_id", "")) for t in trajectory["turns"]), default=0))
    completed = {qid for qid in data.get("completed_question_ids", []) if qid}
    if not completed:
        completed = {
            t.get("question_id")
            for t in trajectory.get("turns", [])
            if t.get("question_type") in {"main", "macro_main_question", "multi_hop_reasoning"} and t.get("question_id")
        }
    ensure_defaults(eval_state, graph)
    rebuild_turn_counts(eval_state, trajectory)
    completed_thread_steps = {sid for sid in data.get("completed_thread_step_ids", []) if sid}
    if not completed_thread_steps:
        completed_thread_steps = completed_thread_step_ids(eval_state)
    return eval_state, trajectory, turn_no, completed, completed_thread_steps


def save_eval_artifacts(
    graph: dict,
    eval_state: dict,
    trajectory: dict,
    checkpoint_path: Path,
    traj_path: Path,
    report_path: Path,
    state_path: Path,
    final_mmd_path: Path,
    public_result_root: Path | None = None,
) -> None:
    reconcile_actual_transitions(trajectory)
    report = build_report(eval_state, trajectory)
    write_json(traj_path, trajectory)
    write_json(report_path, report)
    write_json(state_path, eval_state)
    if public_result_root is not None:
        _write_public_eval_result(public_result_root, graph, trajectory, report)
    final_mmd_path.parent.mkdir(parents=True, exist_ok=True)
    final_mmd_path.write_text(export_final_state_mermaid(graph, eval_state), encoding="utf-8")
    completed_question_ids = [
        t.get("question_id")
        for t in trajectory.get("turns", [])
        if t.get("question_type") in {"main", "macro_main_question", "multi_hop_reasoning"}
    ]
    completed_thread_steps = sorted(completed_thread_step_ids(eval_state))
    write_json(
        checkpoint_path,
        {
            "paper_id": graph.get("paper_id", "unknown"),
            "target_model": trajectory.get("target_model"),
            "turn_no": max((_turn_number(t.get("turn_id", "")) for t in trajectory.get("turns", [])), default=0),
            "completed_question_ids": completed_question_ids,
            "completed_thread_step_ids": completed_thread_steps,
            "scheduler_state": {
                "completed_thread_step_ids": completed_thread_steps,
                "last_turn_id": trajectory.get("turns", [{}])[-1].get("turn_id") if trajectory.get("turns") else None,
            },
            "trajectory": trajectory,
            "eval_state": eval_state,
        },
    )


def _write_public_eval_result(public_result_root: Path, graph: dict, trajectory: dict, report: dict) -> None:
    target_model = _safe_dir_name(trajectory.get("target_model") or "unknown_model")
    paper_id = _safe_dir_name(graph.get("paper_id") or trajectory.get("paper_id") or "unknown_paper")
    out_dir = public_result_root / target_model / paper_id
    write_json(out_dir / "dialogue_trajectory.json", trajectory)
    write_json(out_dir / "evaluation_report.json", report)


def action_for_question_type(question_type: str) -> str:
    return {
        "detail_followup": "detail_followup",
        "hallucination_followup": "hallucination_followup",
        "review_followup": "review_followup",
        "multi_hop_reasoning": "multi_hop_question",
        "main": "next_main_question",
        "macro_main_question": "next_main_question",
        "thread_premise_question": "thread_question",
        "thread_evidence_question": "thread_question",
        "thread_bridge_question": "thread_question",
        "thread_review_question": "thread_question",
        "thread_question": "thread_question",
    }.get(question_type, "next_main_question")


def _turn_number(turn_id: str) -> int:
    match = re.search(r"(\d+)$", str(turn_id))
    return int(match.group(1)) if match else 0


def _safe_dir_name(value: object) -> str:
    safe = re.sub(r"[^\w._-]+", "_", str(value or "").strip())
    return safe.strip("._-") or "unknown"
