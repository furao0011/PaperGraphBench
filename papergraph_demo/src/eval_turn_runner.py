from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

from src.claim_verifier import append_claim_verification_log, claim_verification_enabled, verify_global_claims
from src.dialogue_engine import generate_thread_question
from src.eval_artifacts import append_turn
from src.evaluation_hallucination_state import record_hallucination_events
from src.evaluation_task_queue import attach_recommended_stage_tasks
from src.judge import judge_answer_with_online_fallback
from src.judge_result_normalizer import normalize_after_global_claim_verification, normalize_judge_result
from src.model_client import OpenAICompatClient
from src.progress import log, span
from src.state_updater import apply_claim_verification_results, apply_judge_result


NextActionFn = Callable[[dict, dict, dict], str]


class EvaluationTurnRunner:
    def __init__(
        self,
        graph: dict,
        by_kc: dict[str, dict],
        paper_text: str,
        client: OpenAICompatClient,
        use_online_eval: bool,
        allow_offline_fallback: bool,
        kc_bank: dict,
        claim_log_path: Path,
    ) -> None:
        self.graph = graph
        self.by_kc = by_kc
        self.paper_text = paper_text
        self.client = client
        self.use_online_eval = use_online_eval
        self.allow_offline_fallback = allow_offline_fallback
        self.kc_bank = kc_bank
        self.claim_log_path = claim_log_path

    def run_question_turn(
        self,
        question: dict,
        trajectory: dict,
        eval_state: dict,
        turn_no: int,
        apply_effective_next_action: NextActionFn,
    ) -> tuple[int, dict | None, list[dict], str | None]:
        turn_no += 1
        turn_id = f"T{turn_no}"
        target_kcs = [self.by_kc[k] for k in question.get("target_kc_ids", []) if k in self.by_kc]
        if not target_kcs:
            return turn_no, None, [], None
        log(
            "turn started",
            turn=turn_id,
            question_id=question.get("question_id"),
            question_type=question.get("question_type"),
            targets=",".join(question.get("target_kc_ids", [])),
        )
        answer, answer_mode = self._answer_turn(turn_id, question["question_text"], trajectory, target_kcs)
        judge_result = self._judge_turn(turn_id, question, answer, target_kcs, trajectory)
        turn_context = _turn_context(turn_id, question)
        judge_result = normalize_judge_result(judge_result, turn_context)
        state_update = apply_judge_result(
            eval_state,
            turn_id=turn_id,
            judge_result=judge_result,
            path_id=question.get("path_id"),
            macro_id=question.get("macro_id"),
            question_type=question.get("question_type"),
            question=question,
        )
        next_action = apply_effective_next_action(
            eval_state,
            judge_result,
            turn_context,
        )
        turn = {
            "turn_id": turn_id,
            "question_id": question["question_id"],
            "question_type": question["question_type"],
            "macro_id": question.get("macro_id"),
            "challenge_type": question.get("challenge_type"),
            "challenge_trigger": question.get("challenge_trigger"),
            "target_failure_mode": question.get("target_failure_mode"),
            "expected_behavior": question.get("expected_behavior"),
            "question_text": question["question_text"],
            "target_kc_ids": question["target_kc_ids"],
            "target_path_id": question.get("path_id"),
            "model_answer": answer,
            "answer_mode": answer_mode,
            "judge_result": judge_result,
            "state_update": state_update,
        }
        self._apply_global_claim_verification(turn, eval_state)
        append_turn(trajectory, turn)
        log(
            "evaluation turn recorded",
            turn=turn_id,
            question_type=question.get("question_type"),
            judge_state=judge_result.get("state"),
            coverage=_coverage_summary(judge_result),
            hallucination_events=_hallucination_event_summary(judge_result),
            stage_tasks=_stage_task_summary(judge_result),
            challenge_result=_challenge_result_summary(judge_result),
            answer_mode=answer_mode,
        )
        return turn_no, turn, target_kcs, next_action

    def run_followup_turn(
        self,
        follow: dict,
        trajectory: dict,
        eval_state: dict,
        turn_no: int,
        apply_effective_next_action: NextActionFn,
    ) -> tuple[int, dict | None, list[dict]]:
        turn_no += 1
        turn_id = f"T{turn_no}"
        target_kcs = [self.by_kc[k] for k in follow.get("target_kc_ids", []) if k in self.by_kc]
        if not target_kcs:
            return turn_no, None, []
        log(
            "follow-up turn started",
            turn=turn_id,
            question_id=follow.get("question_id"),
            question_type=follow.get("question_type"),
            targets=",".join(follow.get("target_kc_ids", [])),
        )
        if follow["question_type"] == "review_followup":
            eval_state["global_state"]["review_question_count"] += 1

        answer, answer_mode = self._answer_turn(turn_id, follow["question_text"], trajectory, target_kcs)
        judge_result = self._judge_turn(turn_id, follow, answer, target_kcs, trajectory)
        turn_context = _turn_context(turn_id, follow)
        judge_result = normalize_judge_result(judge_result, turn_context)
        state_update = apply_judge_result(
            eval_state,
            turn_id=turn_id,
            judge_result=judge_result,
            path_id=follow.get("target_path_id"),
            macro_id=follow.get("macro_id"),
            question_type=follow.get("question_type"),
            question=follow,
        )
        apply_effective_next_action(
            eval_state,
            judge_result,
            turn_context,
        )
        turn = {
            "turn_id": turn_id,
            "question_id": follow["question_id"],
            "question_type": follow["question_type"],
            "macro_id": follow.get("macro_id"),
            "question_text": follow["question_text"],
            "target_kc_ids": follow["target_kc_ids"],
            "target_path_id": follow.get("target_path_id"),
            "model_answer": answer,
            "answer_mode": answer_mode,
            "judge_result": judge_result,
            "state_update": state_update,
        }
        self._apply_global_claim_verification(turn, eval_state)
        append_turn(trajectory, turn)
        log(
            "repair/review turn recorded",
            turn=turn_id,
            question_type=follow.get("question_type"),
            judge_state=judge_result.get("state"),
            coverage=_coverage_summary(judge_result),
            hallucination_events=_hallucination_event_summary(judge_result),
            stage_tasks=_stage_task_summary(judge_result),
            answer_mode=answer_mode,
        )
        return turn_no, turn, target_kcs

    def run_thread_turn(
        self,
        seed: dict,
        trajectory: dict,
        eval_state: dict,
        turn_no: int,
        apply_effective_next_action: NextActionFn,
    ) -> tuple[int, dict | None]:
        target_kcs = [self.by_kc[k] for k in seed.get("target_kc_ids", []) if k in self.by_kc]
        if not target_kcs:
            return turn_no, None
        related_turns = self._related_thread_turns(eval_state, trajectory, seed.get("thread_id"))
        question = generate_thread_question(
            seed,
            target_kcs,
            related_turns,
            dialogue_summary(trajectory.get("turns", [])),
            self.client,
            allow_offline_fallback=self.allow_offline_fallback,
        )
        turn_no += 1
        turn_id = f"T{turn_no}"
        log(
            "thread turn started",
            turn=turn_id,
            question_id=question.get("question_id"),
            question_type=question.get("question_type"),
            thread_id=question.get("thread_id"),
            thread_step=question.get("thread_turn_id"),
            targets=",".join(question.get("target_kc_ids", [])),
        )
        answer, answer_mode = self._answer_turn(turn_id, question["question_text"], trajectory, target_kcs)
        thread_context = {
            "thread_id": question.get("thread_id"),
            "thread_turn_id": question.get("thread_turn_id"),
            "thread_role": question.get("thread_role"),
            "question_goal": question.get("question_goal"),
            "trigger_condition": question.get("trigger_condition", {}),
            "success_criteria": seed.get("success_criteria", []),
            "related_turn_ids": [t.get("turn_id") for t in related_turns],
        }
        judge_result = self._judge_turn(turn_id, question, answer, target_kcs, trajectory, thread_context)
        turn_context = _turn_context(turn_id, question)
        judge_result = normalize_judge_result(judge_result, turn_context)
        state_update = apply_judge_result(
            eval_state,
            turn_id=turn_id,
            judge_result=judge_result,
            path_id=question.get("target_path_id"),
            macro_id=question.get("macro_id"),
            question_type=question.get("question_type"),
            thread_id=question.get("thread_id"),
            thread_step_id=question.get("thread_turn_id"),
            question=question,
        )
        apply_effective_next_action(
            eval_state,
            judge_result,
            turn_context,
        )
        turn = {
            "turn_id": turn_id,
            "question_id": question["question_id"],
            "question_type": question["question_type"],
            "macro_id": question.get("macro_id"),
            "thread_id": question.get("thread_id"),
            "thread_turn_id": question.get("thread_turn_id"),
            "thread_role": question.get("thread_role"),
            "question_text": question["question_text"],
            "target_kc_ids": question["target_kc_ids"],
            "target_path_id": question.get("target_path_id"),
            "model_answer": answer,
            "answer_mode": answer_mode,
            "judge_result": judge_result,
            "state_update": state_update,
        }
        self._apply_global_claim_verification(turn, eval_state)
        append_turn(trajectory, turn)
        log(
            "thread stage turn recorded",
            turn=turn_id,
            judge_state=judge_result.get("state"),
            thread_id=question.get("thread_id"),
            thread_step=question.get("thread_turn_id"),
            coverage=_coverage_summary(judge_result),
            hallucination_events=_hallucination_event_summary(judge_result),
            thread_result=_thread_result_summary(judge_result),
            stage_tasks=_stage_task_summary(judge_result),
            answer_mode=answer_mode,
        )
        return turn_no, turn

    def _answer_turn(self, turn_id: str, question_text: str, trajectory: dict, target_kcs: list[dict]) -> tuple[str, str]:
        model_input = build_eval_prompt(
            paper_text=self.paper_text,
            dialogue_history=dialogue_history_text(trajectory["turns"]),
            question_text=question_text,
        )
        with span("target model answer", turn=turn_id):
            return build_model_answer(self.client, self.use_online_eval, model_input, target_kcs)

    def _judge_turn(
        self,
        turn_id: str,
        question: dict,
        answer: str,
        target_kcs: list[dict],
        trajectory: dict,
        thread_context: dict | None = None,
    ) -> dict:
        with span("judge answer", turn=turn_id):
            judge_context = _question_context(question, thread_context)
            return judge_answer_with_online_fallback(
                question["question_text"],
                answer,
                target_kcs,
                self.client,
                use_online_judge=self.use_online_eval,
                dialogue_summary=dialogue_summary(trajectory["turns"]),
                related_forbidden_claims=related_forbidden_claims(
                    self.graph,
                    question.get("target_kc_ids", []),
                    question.get("path_id") or question.get("target_path_id"),
                ),
                question_type=question["question_type"],
                thread_context=judge_context,
            )

    def _apply_global_claim_verification(self, turn: dict, eval_state: dict) -> None:
        if not claim_verification_enabled():
            return
        with span("global claim verification", turn=turn.get("turn_id")):
            results = verify_global_claims(turn, self.kc_bank, self.client)
        turn["global_claim_verification"] = results
        if not results:
            return
        turn["state_update"]["claim_verification_update"] = apply_claim_verification_results(
            eval_state,
            turn_id=turn["turn_id"],
            results=results,
        )
        _apply_supported_claims_to_judge_result(
            turn["judge_result"],
            turn["state_update"]["claim_verification_update"].get("supported_kc_ids", []),
        )
        append_claim_verification_log(self.claim_log_path, results)
        risky = [r for r in results if r.get("label") in {"CONTRADICTED", "OVERCLAIM"}]
        if risky:
            turn_context = _turn_context_from_turn(turn)
            turn["judge_result"] = normalize_after_global_claim_verification(
                turn["judge_result"],
                turn_context,
                results,
            )
            attach_recommended_stage_tasks(
                turn["judge_result"],
                turn_context,
                "hallucination_followup",
            )
            previous_structured_update = turn.get("state_update", {}).get("structured_update", {})
            previous_event_ids = previous_structured_update.get("hallucination_event_ids", [])
            added_event_ids = [
                event["event_id"]
                for event in record_hallucination_events(eval_state, turn["judge_result"])
            ]
            turn["state_update"]["structured_update"] = {
                "hallucination_event_ids": list(dict.fromkeys(previous_event_ids + added_event_ids)),
                "coverage_gap_id": previous_structured_update.get("coverage_gap_id"),
            }
        else:
            turn_context = _turn_context_from_turn(turn)
            turn["judge_result"] = normalize_judge_result(turn["judge_result"], turn_context)
            attach_recommended_stage_tasks(
                turn["judge_result"],
                turn_context,
                _repair_action_from_judge_result(turn["judge_result"]),
            )

    def _related_thread_turns(self, eval_state: dict, trajectory: dict, thread_id: str | None) -> list[dict]:
        if not thread_id:
            return []
        ids = set(
            eval_state.get("thread_states", {})
            .get(thread_id, {})
            .get("related_turns", [])
        )
        return [t for t in trajectory.get("turns", []) if t.get("turn_id") in ids]


def _turn_context(turn_id: str, question: dict) -> dict:
    return {
        "turn_id": turn_id,
        "question_id": question["question_id"],
        "question_type": question["question_type"],
        "macro_id": question.get("macro_id"),
        "thread_id": question.get("thread_id"),
        "thread_turn_id": question.get("thread_turn_id"),
        "thread_role": question.get("thread_role"),
        "target_kc_ids": question.get("target_kc_ids", []),
        "target_path_id": question.get("path_id") or question.get("target_path_id"),
        "challenge_type": question.get("challenge_type"),
        "challenge_trigger": question.get("challenge_trigger"),
        "target_failure_mode": question.get("target_failure_mode"),
        "expected_behavior": question.get("expected_behavior"),
    }


def _turn_context_from_turn(turn: dict) -> dict:
    return {
        "turn_id": turn["turn_id"],
        "question_id": turn["question_id"],
        "question_type": turn["question_type"],
        "macro_id": turn.get("macro_id"),
        "thread_id": turn.get("thread_id"),
        "thread_turn_id": turn.get("thread_turn_id"),
        "thread_role": turn.get("thread_role"),
        "target_kc_ids": turn.get("target_kc_ids", []),
        "target_path_id": turn.get("target_path_id"),
        "challenge_type": turn.get("challenge_type"),
        "challenge_trigger": turn.get("challenge_trigger"),
        "target_failure_mode": turn.get("target_failure_mode"),
        "expected_behavior": turn.get("expected_behavior"),
    }


def _apply_supported_claims_to_judge_result(judge_result: dict, supported_kc_ids: list[str]) -> None:
    if not supported_kc_ids:
        return
    supported_set = set(supported_kc_ids)
    covered = list(dict.fromkeys(list(judge_result.get("covered_kc_ids", [])) + supported_kc_ids))
    missing = [kc_id for kc_id in judge_result.get("missing_kc_ids", []) if kc_id not in supported_set]
    judge_result["covered_kc_ids"] = covered
    judge_result["missing_kc_ids"] = missing
    coverage = judge_result.setdefault("coverage", {})
    coverage["covered_kc_ids"] = covered
    coverage["missing_kc_ids"] = missing
    coverage["coverage_complete"] = not missing
    if judge_result.get("state") == "INCOMPLETE" and not missing:
        judge_result["state"] = "MAIN_PROGRESS"


def _repair_action_from_judge_result(judge_result: dict) -> str:
    if judge_result.get("state") in {"HALLUCINATION", "MISLED", "GLOBAL_OVERCLAIM"}:
        return "hallucination_followup"
    if judge_result.get("missing_kc_ids"):
        return "detail_followup"
    return "next_main_question"


def select_followup_target_kcs(
    action: str,
    judge_result: dict,
    by_kc: dict[str, dict],
    fallback_kcs: list[dict],
    last_turn: dict | None = None,
    trajectory: dict | None = None,
) -> list[dict]:
    if action == "detail_followup":
        ids = judge_result.get("missing_kc_ids", [])
    elif action == "hallucination_followup":
        ids = judge_result.get("covered_kc_ids", []) + judge_result.get("missing_kc_ids", [])
    else:
        ids = []
    selected = [by_kc[kid] for kid in ids if kid in by_kc]
    return selected or fallback_kcs[:1]


def dialogue_summary(turns: list[dict], keep_last: int = 4) -> str:
    if not turns:
        return ""
    items = turns[-keep_last:]
    lines = []
    for t in items:
        q = t.get("question_text", "")[:180]
        a = t.get("model_answer", "")[:220]
        lines.append(f"{t.get('turn_id')}: Q={q} | A={a}")
    return "\n".join(lines)


def dialogue_history_text(turns: list[dict]) -> str:
    if not turns:
        return "No previous turns."
    return "\n".join(
        f"{t['turn_id']} Q:{t['question_text']} A:{t['model_answer']}"
        for t in turns
    )


def build_eval_prompt(paper_text: str, dialogue_history: str, question_text: str) -> str:
    return (
        "```original paper\n"
        f"{paper_text}\n"
        "```\n\n"
        "[dialogue history]\n"
        f"{dialogue_history}\n\n"
        "[current question]\n"
        f"{question_text}"
    )


def build_model_answer(
    client: OpenAICompatClient | None,
    use_online_eval: bool,
    prompt: str,
    target_kcs: list[dict],
) -> tuple[str, str]:
    if use_online_eval and client and client.is_ready():
        ans = client.chat_text(
            system_prompt="Answer the paper-evaluation question based only on the provided original paper and dialogue context.",
            user_prompt=prompt,
        )
        return ans, "online"
    if os.getenv("ALLOW_MOCK_EVAL", "false").lower() in {"1", "true", "yes", "on"}:
        joined = "; ".join(k["full_claim"] for k in target_kcs[:2])
        return f"Based on the paper, {joined}", "mock"
    raise RuntimeError("Online evaluation requires a configured model and USE_ONLINE_EVAL=true. Set ALLOW_MOCK_EVAL=true only for local debugging.")


def _coverage_summary(judge_result: dict) -> str:
    covered = len(judge_result.get("covered_kc_ids", []))
    missing = len(judge_result.get("missing_kc_ids", []))
    complete = judge_result.get("coverage", {}).get("coverage_complete")
    return f"covered={covered},missing={missing},complete={complete}"


def _hallucination_event_summary(judge_result: dict) -> str:
    events = judge_result.get("hallucination_events", []) or []
    if not events:
        return "none"
    by_type: dict[str, int] = {}
    for event in events:
        key = event.get("hallucination_type") or "unknown"
        by_type[key] = by_type.get(key, 0) + 1
    return ",".join(f"{key}:{value}" for key, value in sorted(by_type.items()))


def _stage_task_summary(judge_result: dict) -> str:
    tasks = judge_result.get("recommended_stage_tasks", []) or []
    if not tasks:
        return "none"
    by_type: dict[str, int] = {}
    for task in tasks:
        key = task.get("task_type") or "unknown"
        by_type[key] = by_type.get(key, 0) + 1
    return ",".join(f"{key}:{value}" for key, value in sorted(by_type.items()))


def _thread_result_summary(judge_result: dict) -> str:
    result = judge_result.get("thread_result") or {}
    if not result:
        return "none"
    return "success={success},partial={partial},failed={failed},path={path}".format(
        success=result.get("success"),
        partial=result.get("partial"),
        failed=result.get("failed"),
        path=result.get("reasoning_path_result"),
    )


def _challenge_result_summary(judge_result: dict) -> str:
    result = judge_result.get("challenge_result") or {}
    if not result:
        return "none"
    return "resisted={resisted},failed={failed},incomplete={incomplete}".format(
        resisted=result.get("resisted"),
        failed=result.get("failed"),
        incomplete=result.get("incomplete"),
    )


def related_forbidden_claims(graph: dict, target_kc_ids: list[str], path_id: str | None) -> list[dict]:
    out: list[dict] = []
    tset = set(target_kc_ids)
    for e in graph.get("reasoning_edges", []):
        if e.get("source") in tset or e.get("target") in tset:
            out.extend(e.get("forbidden_claims", []))
    if path_id:
        for p in graph.get("reasoning_paths", []):
            if p.get("path_id") == path_id:
                out.extend(p.get("forbidden_claims", []))
                break
    return out


def _question_context(question: dict, base_context: dict | None) -> dict:
    context = dict(base_context or {})
    if question.get("question_type") == "challenge_question":
        context.update(
            {
                "challenge_type": question.get("challenge_type"),
                "target_failure_mode": question.get("target_failure_mode"),
                "expected_behavior": question.get("expected_behavior"),
                "surface_intent": question.get("surface_intent"),
                "evidence": question.get("evidence", []),
                "challenge_trigger": question.get("challenge_trigger"),
                "needs_human_review": question.get("needs_human_review", False),
                "all_solvers_failed": question.get("all_solvers_failed", False),
                "solver_trial_summary": question.get("solver_trial_summary", {}),
            }
        )
    return context
