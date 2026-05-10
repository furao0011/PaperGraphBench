from __future__ import annotations
from collections.abc import Callable

from src.challenge_scheduler import get_macro_challenge, get_thread_challenge
from src.dialogue_engine import generate_followup_question
from src.eval_turn_runner import EvaluationTurnRunner
from src.evaluation_hallucination_state import (
    mark_coverage_gap_exhausted,
    mark_coverage_gap_followed_up,
    mark_coverage_gap_resolved,
    mark_hallucination_events_exhausted,
    mark_hallucination_events_followed_up,
    mark_hallucination_events_resolved,
)
from src.evaluation_policy import apply_effective_next_action, bounded_env_int, review_target_at_end
from src.evaluation_stage_tasks import (
    action_for_repair_task,
    attach_coverage_gap_ids,
    enqueue_anchor_task,
    repair_task_resolved,
    review_allows_hallucination_followup,
)
from src.evaluation_task_queue import (
    TASK_TYPE_CHALLENGE_EVALUATION,
    TASK_TYPE_DETAIL_COMPLETION,
    TASK_TYPE_HALLUCINATION_REPAIR,
    TASK_TYPE_REVIEW,
    TASK_TYPE_THREAD_REASONING,
    enqueue_stage_tasks,
    ensure_macro_stage_status,
    mark_stage_task_completed,
    mark_stage_task_exhausted,
    mark_stage_task_running,
    next_pending_stage_task,
    record_stage_task_turn,
    stage_task_has_budget,
    update_stage_task_status,
)
from src.model_client import OpenAICompatClient
from src.progress import log
from src.thread_scheduler import get_ready_thread_turn


SaveArtifactsFn = Callable[[dict, dict, int], None]


class EvaluationStageRunner:
    def __init__(
        self,
        runner: EvaluationTurnRunner,
        graph: dict,
        questions: dict,
        trajectory: dict,
        eval_state: dict,
        completed_question_ids: set,
        by_kc: dict[str, dict],
        client: OpenAICompatClient,
        allow_offline_fallback: bool,
        save_artifacts: SaveArtifactsFn,
        max_turns: int,
    ) -> None:
        self.runner = runner
        self.graph = graph
        self.questions = questions
        self.trajectory = trajectory
        self.eval_state = eval_state
        self.completed_question_ids = completed_question_ids
        self.by_kc = by_kc
        self.client = client
        self.allow_offline_fallback = allow_offline_fallback
        self.save_artifacts = save_artifacts
        self.max_turns = max_turns

    def run_macro_stage(self, question: dict, turn_no: int) -> int:
        if self._should_stop(turn_no) or question.get("question_id") in self.completed_question_ids:
            return turn_no

        macro_id = question.get("macro_id")
        macro_status = ensure_macro_stage_status(self.eval_state, macro_id) if macro_id else None
        if macro_status is not None:
            macro_status["status"] = "in_main"

        log("macro stage started", macro_id=macro_id, question_id=question.get("question_id"))
        turn_no, turn, target_kcs, _ = self.runner.run_question_turn(
            question,
            self.trajectory,
            self.eval_state,
            turn_no,
            apply_effective_next_action,
        )
        if not turn:
            return turn_no
        self.completed_question_ids.add(question["question_id"])
        self.save_artifacts(self.eval_state, self.trajectory, turn_no)
        self._enqueue_repair_tasks_from_turn(turn)
        if macro_status is not None:
            macro_status["main_done"] = True
            macro_status["status"] = "in_repair"

        turn_no = self._process_repair_queue(macro_id, turn_no)

        if macro_status is not None:
            macro_status["repair_done"] = True
            macro_status["status"] = "in_thread"
        turn_no = self._run_thread_tasks(turn_no, review_stage=False)

        if macro_status is not None:
            macro_status["thread_done"] = True
            macro_status["status"] = "in_challenge"
        turn_no = self._run_challenge_tasks(
            lambda macro_id=macro_id: get_macro_challenge(
                self.eval_state,
                self.questions.get("challenge_questions", []),
                macro_id,
            ),
            turn_no,
            macro_id=macro_id,
        )

        if macro_status is not None:
            macro_status["challenge_done"] = True
            macro_status["status"] = "failed" if self.eval_state["global_state"]["failed"] else "completed"
        return turn_no

    def run_review_stage(self, turn_no: int) -> int:
        end_targets = self._build_review_followups(turn_no)
        for follow in end_targets:
            if self._should_stop(turn_no):
                break
            task_id = enqueue_anchor_task(
                self.eval_state,
                TASK_TYPE_REVIEW,
                follow["question_id"],
                follow.get("macro_id"),
                follow["question_id"],
                follow.get("target_kc_ids", []),
                turn_no + 1,
            )
            mark_stage_task_running(self.eval_state, task_id)
            turn_no, follow_turn, _ = self.runner.run_followup_turn(
                follow,
                self.trajectory,
                self.eval_state,
                turn_no,
                apply_effective_next_action,
            )
            if not follow_turn:
                self._mark_task_exhausted(task_id, turn_id=None)
                continue
            self.save_artifacts(self.eval_state, self.trajectory, turn_no)
            record_stage_task_turn(self.eval_state, task_id, follow_turn["turn_id"])
            mark_stage_task_completed(self.eval_state, task_id)
            self._enqueue_repair_tasks_from_turn(follow_turn)
            if self._review_allows_hallucination_followup():
                turn_no = self._process_repair_queue(follow_turn.get("macro_id"), turn_no)

        turn_no = self._run_thread_tasks(turn_no, review_stage=True)
        return turn_no

    def _process_repair_queue(self, macro_id: str | None, turn_no: int) -> int:
        repair_types = {TASK_TYPE_HALLUCINATION_REPAIR, TASK_TYPE_DETAIL_COMPLETION}
        while not self._should_stop(turn_no):
            task = next_pending_stage_task(self.eval_state, macro_id=macro_id, task_types=repair_types)
            if not task:
                break
            if not stage_task_has_budget(task):
                self._mark_task_exhausted(task["task_id"], turn_id=None)
                continue
            next_turn_no = self._run_repair_task(task, turn_no)
            if next_turn_no == turn_no:
                break
            turn_no = next_turn_no
        return turn_no

    def _run_repair_task(self, task: dict, turn_no: int) -> int:
        action = action_for_repair_task(task)
        source_turn = self._turn_by_id(task.get("source_turn_id"))
        if not source_turn:
            self._mark_task_exhausted(task["task_id"], turn_id=None)
            return turn_no
        targets = self._task_target_kcs(task, source_turn)
        if not targets:
            self._mark_task_exhausted(task["task_id"], turn_id=None)
            return turn_no

        mark_stage_task_running(self.eval_state, task["task_id"])
        follow = generate_followup_question(
            action,
            source_turn,
            targets,
            self.client,
            allow_offline_fallback=self.allow_offline_fallback,
            repair_context=task.get("repair_context", {}),
        )
        if not follow:
            self._mark_task_exhausted(task["task_id"], turn_id=None)
            return turn_no
        log(
            "repair task scheduled",
            task=task["task_id"],
            task_type=task.get("task_type"),
            repair_question_type=follow.get("question_type"),
            source_turn=source_turn.get("turn_id"),
            targets=",".join(k["kc_id"] for k in targets),
            attempt=f"{task.get('current_turns', 0) + 1}/{task.get('max_turns')}",
        )
        turn_no, follow_turn, _ = self.runner.run_followup_turn(
            follow,
            self.trajectory,
            self.eval_state,
            turn_no,
            apply_effective_next_action,
        )
        if not follow_turn:
            self._mark_task_exhausted(task["task_id"], turn_id=None)
            return turn_no
        self.save_artifacts(self.eval_state, self.trajectory, turn_no)
        record_stage_task_turn(self.eval_state, task["task_id"], follow_turn["turn_id"])
        self._mark_task_followed_up(task["task_id"], follow_turn["turn_id"])
        self._enqueue_repair_tasks_from_turn(
            follow_turn,
            source_task_type=task.get("task_type"),
        )
        self._settle_repair_task(task["task_id"], follow_turn)
        return turn_no

    def _settle_repair_task(self, task_id: str, follow_turn: dict) -> None:
        task = self.eval_state["stage_tasks"][task_id]
        judge_result = follow_turn.get("judge_result", {})
        if repair_task_resolved(task, judge_result):
            self._mark_task_completed(task_id, follow_turn["turn_id"])
            return
        if not stage_task_has_budget(task):
            self._mark_task_exhausted(task_id, follow_turn["turn_id"])
            return
        if task.get("task_type") == TASK_TYPE_DETAIL_COMPLETION:
            remaining = [
                kc_id
                for kc_id in task.get("target_kc_ids", [])
                if kc_id in set(judge_result.get("missing_kc_ids", []))
            ]
            task["target_kc_ids"] = remaining
            task.setdefault("repair_context", {})["remaining_kc_ids"] = remaining
            task.setdefault("repair_context", {}).setdefault("covered_during_repair", [])
            for kc_id in judge_result.get("covered_kc_ids", []):
                if kc_id not in task["repair_context"]["covered_during_repair"]:
                    task["repair_context"]["covered_during_repair"].append(kc_id)
        task["source_turn_id"] = follow_turn["turn_id"]
        task["source_question_id"] = follow_turn["question_id"]
        update_stage_task_status(self.eval_state, task_id, "pending")

    def _run_thread_tasks(self, turn_no: int, review_stage: bool) -> int:
        thread_budget = 1 if review_stage else bounded_env_int("EVAL_THREAD_TURNS_PER_CHECK", 1, 1, 3)
        while thread_budget > 0 and not self._should_stop(turn_no):
            seed = get_ready_thread_turn(self.eval_state, self.graph.get("reasoning_threads", []), review_stage=review_stage)
            if not seed:
                break
            task_id = enqueue_anchor_task(
                self.eval_state,
                TASK_TYPE_THREAD_REASONING,
                seed.get("thread_turn_id") or seed.get("question_id"),
                seed.get("preferred_macro_id") or seed.get("macro_id"),
                seed.get("question_id") or f"Q_{seed.get('thread_turn_id')}",
                seed.get("target_kc_ids", []),
                turn_no + 1,
            )
            mark_stage_task_running(self.eval_state, task_id)
            turn_no, thread_turn = self.runner.run_thread_turn(
                seed,
                self.trajectory,
                self.eval_state,
                turn_no,
                apply_effective_next_action,
            )
            if not thread_turn:
                self._mark_task_exhausted(task_id, turn_id=None)
                break
            self.save_artifacts(self.eval_state, self.trajectory, turn_no)
            record_stage_task_turn(self.eval_state, task_id, thread_turn["turn_id"])
            mark_stage_task_completed(self.eval_state, task_id)
            self._enqueue_repair_tasks_from_turn(thread_turn)
            turn_no = self._process_repair_queue(thread_turn.get("macro_id"), turn_no)
            if not review_stage:
                turn_no = self._run_challenge_tasks(
                    lambda thread_id=thread_turn.get("thread_id"): get_thread_challenge(
                        self.eval_state,
                        self.questions.get("challenge_questions", []),
                        thread_id,
                    ),
                    turn_no,
                    macro_id=thread_turn.get("macro_id"),
                )
            thread_budget -= 1
        return turn_no

    def _run_challenge_tasks(self, challenge_source, turn_no: int, macro_id: str | None) -> int:
        while not self._should_stop(turn_no):
            challenge = challenge_source()
            if not challenge:
                break
            if challenge.get("question_id") in self.completed_question_ids:
                break
            task_id = enqueue_anchor_task(
                self.eval_state,
                TASK_TYPE_CHALLENGE_EVALUATION,
                challenge["question_id"],
                challenge.get("macro_id") or macro_id,
                challenge["question_id"],
                challenge.get("target_kc_ids", []),
                turn_no + 1,
            )
            mark_stage_task_running(self.eval_state, task_id)
            turn_no, challenge_turn, _, _ = self.runner.run_question_turn(
                challenge,
                self.trajectory,
                self.eval_state,
                turn_no,
                apply_effective_next_action,
            )
            if not challenge_turn:
                self._mark_task_exhausted(task_id, turn_id=None)
                break
            self.completed_question_ids.add(challenge["question_id"])
            self.save_artifacts(self.eval_state, self.trajectory, turn_no)
            record_stage_task_turn(self.eval_state, task_id, challenge_turn["turn_id"])
            mark_stage_task_completed(self.eval_state, task_id)
            self._enqueue_repair_tasks_from_turn(challenge_turn)
            turn_no = self._process_repair_queue(macro_id or challenge_turn.get("macro_id"), turn_no)
        return turn_no

    def _build_review_followups(self, turn_no: int) -> list[dict]:
        if not self.trajectory.get("turns") or (self.max_turns and turn_no >= self.max_turns):
            return []
        needed_reviews = max(
            0,
            review_target_at_end() - self.eval_state["global_state"].get("review_question_count", 0),
        )
        review_sources = [
            turn
            for turn in reversed(self.trajectory["turns"])
            if turn.get("question_type") in {"main", "multi_hop_reasoning"}
        ]
        if not review_sources:
            review_sources = [self.trajectory["turns"][-1]]
        end_targets = []
        for idx, source_turn in enumerate(review_sources[:needed_reviews], start=1):
            targets = [self.by_kc[k] for k in source_turn.get("target_kc_ids", []) if k in self.by_kc]
            if not targets:
                continue
            follow = generate_followup_question(
                "review_followup",
                source_turn,
                targets,
                self.client,
                allow_offline_fallback=self.allow_offline_fallback,
            )
            if follow:
                follow["question_id"] = f"{source_turn['question_id']}_R{idx}"
                end_targets.append(follow)
        return end_targets

    def _enqueue_repair_tasks_from_turn(
        self,
        turn: dict,
        source_task_type: str | None = None,
    ) -> None:
        tasks = turn.get("judge_result", {}).get("recommended_stage_tasks", [])
        if source_task_type == TASK_TYPE_HALLUCINATION_REPAIR:
            tasks = [task for task in tasks if task.get("task_type") == TASK_TYPE_HALLUCINATION_REPAIR]
        elif source_task_type == TASK_TYPE_DETAIL_COMPLETION:
            tasks = [task for task in tasks if task.get("task_type") == TASK_TYPE_HALLUCINATION_REPAIR]
        if tasks:
            attach_coverage_gap_ids(tasks, turn)
            added = enqueue_stage_tasks(self.eval_state, tasks)
            log(
                "repair tasks enqueued",
                source_turn=turn.get("turn_id"),
                source_question_type=turn.get("question_type"),
                tasks=",".join(added),
                task_types=_task_type_summary(tasks),
            )

    def _task_target_kcs(self, task: dict, source_turn: dict) -> list[dict]:
        ids = [kid for kid in task.get("target_kc_ids", []) if kid in self.by_kc]
        if not ids:
            ids = [kid for kid in source_turn.get("target_kc_ids", []) if kid in self.by_kc]
        return [self.by_kc[kid] for kid in ids]

    def _turn_by_id(self, turn_id: str | None) -> dict | None:
        if not turn_id:
            return None
        for turn in reversed(self.trajectory.get("turns", [])):
            if turn.get("turn_id") == turn_id:
                return turn
        return None

    def _should_stop(self, turn_no: int) -> bool:
        if self.eval_state["global_state"]["failed"]:
            return True
        return bool(self.max_turns and turn_no >= self.max_turns)

    def _review_allows_hallucination_followup(self) -> bool:
        return review_allows_hallucination_followup()

    def _mark_task_followed_up(self, task_id: str, turn_id: str) -> None:
        task = self.eval_state["stage_tasks"][task_id]
        if task.get("task_type") == TASK_TYPE_HALLUCINATION_REPAIR:
            mark_hallucination_events_followed_up(self.eval_state, task.get("hallucination_event_ids", []), turn_id)
        elif task.get("task_type") == TASK_TYPE_DETAIL_COMPLETION:
            mark_coverage_gap_followed_up(self.eval_state, task.get("coverage_gap_id"), turn_id)

    def _mark_task_completed(self, task_id: str, turn_id: str) -> None:
        task = mark_stage_task_completed(self.eval_state, task_id)
        if task.get("task_type") == TASK_TYPE_HALLUCINATION_REPAIR:
            mark_hallucination_events_resolved(self.eval_state, task.get("hallucination_event_ids", []), turn_id)
        elif task.get("task_type") == TASK_TYPE_DETAIL_COMPLETION:
            mark_coverage_gap_resolved(self.eval_state, task.get("coverage_gap_id"), turn_id)

    def _mark_task_exhausted(self, task_id: str, turn_id: str | None) -> None:
        task = mark_stage_task_exhausted(self.eval_state, task_id)
        if task.get("task_type") == TASK_TYPE_HALLUCINATION_REPAIR:
            mark_hallucination_events_exhausted(self.eval_state, task.get("hallucination_event_ids", []), turn_id)
        elif task.get("task_type") == TASK_TYPE_DETAIL_COMPLETION:
            mark_coverage_gap_exhausted(self.eval_state, task.get("coverage_gap_id"), turn_id)


def _task_type_summary(tasks: list[dict]) -> str:
    by_type: dict[str, int] = {}
    for task in tasks:
        key = task.get("task_type") or "unknown"
        by_type[key] = by_type.get(key, 0) + 1
    return ",".join(f"{key}:{value}" for key, value in sorted(by_type.items()))
