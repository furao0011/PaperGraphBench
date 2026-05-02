from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

from src.claim_verifier import append_claim_verification_log, claim_verification_enabled, verify_global_claims
from src.dialogue_engine import generate_thread_question
from src.eval_artifacts import append_turn
from src.judge import judge_answer_with_online_fallback
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
        judge_result = normalize_judge_result_for_turn(judge_result, answer, question["question_type"])
        state_update = apply_judge_result(
            eval_state,
            turn_id=turn_id,
            judge_result=judge_result,
            path_id=question.get("path_id"),
            macro_id=question.get("macro_id"),
            question_type=question.get("question_type"),
        )
        next_action = apply_effective_next_action(
            eval_state,
            judge_result,
            {"question_type": question["question_type"], "macro_id": question.get("macro_id")},
        )
        turn = {
            "turn_id": turn_id,
            "question_id": question["question_id"],
            "question_type": question["question_type"],
            "macro_id": question.get("macro_id"),
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
            "turn judged",
            turn=turn_id,
            state=judge_result.get("state"),
            next_action=next_action,
            covered=len(judge_result.get("covered_kc_ids", [])),
            missing=len(judge_result.get("missing_kc_ids", [])),
            hallucinations=len(judge_result.get("hallucinated_claims", [])),
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
        if follow["question_type"] == "misleading_followup":
            eval_state["global_state"]["misleading_question_count"] += 1
            macro_id = follow.get("macro_id")
            if macro_id:
                eval_state.setdefault("macro_states", {}).setdefault(macro_id, {})
                current = eval_state["macro_states"][macro_id].get("misleading_question_count", 0)
                eval_state["macro_states"][macro_id]["misleading_question_count"] = current + 1
        if follow["question_type"] == "review_followup":
            eval_state["global_state"]["review_question_count"] += 1

        answer, answer_mode = self._answer_turn(turn_id, follow["question_text"], trajectory, target_kcs)
        judge_result = self._judge_turn(turn_id, follow, answer, target_kcs, trajectory)
        judge_result = normalize_judge_result_for_turn(judge_result, answer, follow["question_type"])
        state_update = apply_judge_result(
            eval_state,
            turn_id=turn_id,
            judge_result=judge_result,
            path_id=follow.get("target_path_id"),
            macro_id=follow.get("macro_id"),
            question_type=follow.get("question_type"),
        )
        apply_effective_next_action(
            eval_state,
            judge_result,
            {"question_type": follow["question_type"], "macro_id": follow.get("macro_id")},
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
            "follow-up turn judged",
            turn=turn_id,
            state=judge_result.get("state"),
            next_action=judge_result.get("next_action"),
            covered=len(judge_result.get("covered_kc_ids", [])),
            missing=len(judge_result.get("missing_kc_ids", [])),
            hallucinations=len(judge_result.get("hallucinated_claims", [])),
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
        judge_result = normalize_judge_result_for_turn(judge_result, answer, question["question_type"])
        state_update = apply_judge_result(
            eval_state,
            turn_id=turn_id,
            judge_result=judge_result,
            path_id=question.get("target_path_id"),
            macro_id=question.get("macro_id"),
            question_type=question.get("question_type"),
            thread_id=question.get("thread_id"),
            thread_step_id=question.get("thread_turn_id"),
        )
        apply_effective_next_action(
            eval_state,
            judge_result,
            {"question_type": question["question_type"], "macro_id": question.get("macro_id")},
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
            "thread turn judged",
            turn=turn_id,
            state=judge_result.get("state"),
            next_action=judge_result.get("next_action"),
            thread_id=question.get("thread_id"),
            thread_step=question.get("thread_turn_id"),
            covered=len(judge_result.get("covered_kc_ids", [])),
            missing=len(judge_result.get("missing_kc_ids", [])),
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
                thread_context=thread_context,
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
        append_claim_verification_log(self.claim_log_path, results)
        risky = [r for r in results if r.get("label") in {"CONTRADICTED", "OVERCLAIM"}]
        if risky:
            turn["judge_result"]["global_claim_verification"] = results
            turn["judge_result"]["state"] = "GLOBAL_OVERCLAIM"
            turn["judge_result"]["next_action"] = "hallucination_followup"
            turn["judge_result"]["policy_next_action"] = "hallucination_followup"

    def _related_thread_turns(self, eval_state: dict, trajectory: dict, thread_id: str | None) -> list[dict]:
        if not thread_id:
            return []
        ids = set(
            eval_state.get("thread_states", {})
            .get(thread_id, {})
            .get("related_turns", [])
        )
        return [t for t in trajectory.get("turns", []) if t.get("turn_id") in ids]


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
    elif action == "misleading_followup" and last_turn and trajectory:
        macro_id = last_turn.get("macro_id")
        used = {
            kid
            for turn in trajectory.get("turns", [])
            if turn.get("question_type") == "misleading_followup" and turn.get("macro_id") == macro_id
            for kid in turn.get("target_kc_ids", [])
        }
        macro_kcs = [
            kc
            for kc in by_kc.values()
            if kc.get("macro_id") == macro_id and kc.get("kc_id") not in used
        ]
        fallback_ids = {k.get("kc_id") for k in fallback_kcs}
        candidates = [k for k in fallback_kcs if k.get("kc_id") not in used]
        candidates.extend(k for k in macro_kcs if k.get("kc_id") not in fallback_ids)
        return candidates[:1] or fallback_kcs[:1]
    else:
        ids = []
    selected = [by_kc[kid] for kid in ids if kid in by_kc]
    return selected or fallback_kcs[:1]


def normalize_judge_result_for_turn(judge_result: dict, answer: str, question_type: str) -> dict:
    if judge_result.get("state") == "INCOMPLETE" and not judge_result.get("missing_kc_ids"):
        fixed = dict(judge_result)
        fixed["state"] = "MAIN_PROGRESS"
        fixed["next_action"] = "next_main_question"
        explanation = fixed.get("judge_explanation", "")
        fixed["judge_explanation"] = (
            explanation
            + " Normalized: all target KCs were covered; incompleteness only concerned off-target material."
        ).strip()
        return fixed
    return judge_result


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
