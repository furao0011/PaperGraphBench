from __future__ import annotations

from src.evaluation_stage_runner import EvaluationStageRunner, _task_type_summary
from src.progress import log


class NoRepairEvaluationStageRunner(EvaluationStageRunner):
    """Run the normal contextual schedule while suppressing repair follow-ups."""

    def _enqueue_repair_tasks_from_turn(
        self,
        turn: dict,
        source_task_type: str | None = None,
    ) -> None:
        tasks = turn.get("judge_result", {}).get("recommended_stage_tasks", [])
        if not tasks:
            return
        log(
            "repair tasks suppressed for without-repair ablation",
            source_turn=turn.get("turn_id"),
            source_question_type=turn.get("question_type"),
            task_types=_task_type_summary(tasks),
        )

    def _process_repair_queue(self, macro_id: str | None, turn_no: int) -> int:
        return turn_no

    def _review_allows_hallucination_followup(self) -> bool:
        return False
