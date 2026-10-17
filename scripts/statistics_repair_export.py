"""Export auditable repair tasks independently of final state caches."""

from pathlib import Path

from statistics_inputs import read_json
from statistics_repairs import extract_repair_tasks


REPAIR_COLUMNS = [
    "round", "model", "paper_id", "repair_task_id", "repair_type",
    "root_turn_id", "root_question_id", "root_question_type", "start_turn_index",
    "end_turn_index", "attempts", "turn_span", "success", "exhausted",
    "success_turn_id", "success_turn_index", "final_turn_id", "final_judge_state",
    "trajectory_path",
]


def repair_rows(root: Path, rows: list[dict]) -> list[dict]:
    result = []
    for row in rows:
        if row["status"] != "completed":
            continue
        path = (root / row["report_path"]).with_name("dialogue_trajectory.json").resolve()
        if not path.is_relative_to(root):
            raise ValueError("Repair trajectory path escapes its result directory")
        for task in extract_repair_tasks(read_json(path)):
            result.append({**{key: row[key] for key in ("round", "model", "paper_id")},
                           **task, "trajectory_path": path.relative_to(root).as_posix()})
    return result
