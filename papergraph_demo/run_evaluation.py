import json
import os
from pathlib import Path

from src.config import load_settings
from src.dialogue_engine import generate_followup_question
from src.eval_artifacts import load_eval_checkpoint, save_eval_artifacts
from src.eval_turn_runner import EvaluationTurnRunner, select_followup_target_kcs
from src.mermaid_exporter import export_final_thread_state_mermaid
from src.model_client import ModelConfig, OpenAICompatClient
from src.paper_context import load_full_paper_text
from src.policy_controller import choose_next_action
from src.progress import log, span
from src.question_generator import normalize_question_bundle
from src.state_updater import initialize_eval_state
from src.thread_scheduler import (
    THREAD_QUESTION_TYPES,
    ensure_thread_states,
    get_ready_thread_turn,
)


BASE_DIR = Path(__file__).resolve().parent
GRAPH_PATH = BASE_DIR / "data" / "graphs" / "master_graph.json"
QUESTION_PATH = BASE_DIR / "data" / "questions" / "question_templates.json"
TRAJ_PATH = BASE_DIR / "data" / "outputs" / "dialogue_trajectory.json"
REPORT_PATH = BASE_DIR / "data" / "outputs" / "evaluation_report.json"
STATE_PATH = BASE_DIR / "data" / "graphs" / "eval_state_graph.json"
FINAL_MMD_PATH = BASE_DIR / "data" / "graphs" / "final_state_graph.mmd"
FINAL_THREAD_MMD_PATH = BASE_DIR / "data" / "graphs" / "final_thread_state_graph.mmd"
EVAL_CHECKPOINT_PATH = BASE_DIR / "data" / "outputs" / "evaluation_checkpoint.json"
CLAIM_LOG_PATH = BASE_DIR / "data" / "outputs" / "claim_verification_log.json"


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _save_eval_artifacts(graph: dict, eval_state: dict, trajectory: dict, checkpoint_path: Path) -> None:
    save_eval_artifacts(
        graph,
        eval_state,
        trajectory,
        checkpoint_path,
        TRAJ_PATH,
        REPORT_PATH,
        STATE_PATH,
        FINAL_MMD_PATH,
    )
    FINAL_THREAD_MMD_PATH.write_text(export_final_thread_state_mermaid(graph, eval_state), encoding="utf-8")


def _ensure_eval_state_defaults(eval_state: dict, graph: dict) -> None:
    eval_state.setdefault("macro_states", {})
    for macro in graph.get("macro_nodes", []):
        macro_id = macro.get("macro_id")
        if not macro_id:
            continue
        eval_state["macro_states"].setdefault(macro_id, {})
        eval_state["macro_states"][macro_id].setdefault("status", "not_started")
        eval_state["macro_states"][macro_id].setdefault("main_question_asked", False)
        eval_state["macro_states"][macro_id].setdefault("covered_kc_ids", [])
        eval_state["macro_states"][macro_id].setdefault("missing_kc_ids", [])
        eval_state["macro_states"][macro_id].setdefault("related_turns", [])
        eval_state["macro_states"][macro_id].setdefault("misleading_question_count", 0)
        eval_state["macro_states"][macro_id].setdefault("bank_kc_count", macro.get("bank_kc_count", len(macro.get("kc_ids", []))))
        eval_state["macro_states"][macro_id].setdefault("active_kc_count", len(macro.get("kc_ids", [])))
    ensure_thread_states(eval_state, graph.get("reasoning_threads", []))
    eval_state.setdefault("claim_verification_states", {})
    eval_state.setdefault("global_state", {})
    eval_state["global_state"].setdefault("misleading_question_count", 0)
    eval_state["global_state"].setdefault("review_question_count", 0)
    eval_state["global_state"].setdefault("global_overclaim_count", 0)
    eval_state["global_state"].setdefault("global_contradicted_claim_count", 0)
    eval_state["global_state"].setdefault("not_enough_info_claim_count", 0)
    eval_state["global_state"].setdefault("thread_bridge_tested_count", 0)
    eval_state["global_state"].setdefault("thread_bridge_success_count", 0)
    eval_state["global_state"].setdefault("evaluation_status", "not_started")
    eval_state["global_state"].setdefault("completion_reason", None)
    eval_state["global_state"].setdefault("completed_at_turn", None)


def _rebuild_macro_misleading_counts(eval_state: dict, trajectory: dict) -> None:
    counts: dict[str, int] = {}
    total = 0
    review_total = 0
    for turn in trajectory.get("turns", []):
        if turn.get("question_type") == "review_followup":
            review_total += 1
            continue
        if turn.get("question_type") != "misleading_followup":
            continue
        macro_id = turn.get("macro_id")
        if not macro_id:
            continue
        counts[macro_id] = counts.get(macro_id, 0) + 1
        total += 1
    for macro_id, count in counts.items():
        eval_state.setdefault("macro_states", {}).setdefault(macro_id, {})
        current = eval_state["macro_states"][macro_id].get("misleading_question_count", 0)
        eval_state["macro_states"][macro_id]["misleading_question_count"] = max(current, count)
    current_total = eval_state.get("global_state", {}).get("misleading_question_count", 0)
    eval_state.setdefault("global_state", {})["misleading_question_count"] = max(current_total, total)
    current_reviews = eval_state["global_state"].get("review_question_count", 0)
    eval_state["global_state"]["review_question_count"] = max(current_reviews, review_total)


def _load_kc_bank(graph: dict) -> dict:
    raw_path = graph.get("kc_bank_path")
    if not raw_path:
        return {"paper_id": graph.get("paper_id"), "kc_nodes": graph.get("kc_nodes", [])}
    path = Path(raw_path)
    if not path.is_absolute():
        path = BASE_DIR / path
    if not path.exists():
        raise FileNotFoundError(f"KC Bank not found for claim verification: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _repair_questions_for_graph(graph: dict, questions: dict) -> dict:
    return {
        "paper_id": graph.get("paper_id"),
        **normalize_question_bundle(graph, questions),
    }


def _is_immediate_followup(action: str) -> bool:
    return action in {
        "detail_followup",
        "hallucination_followup",
        "misleading_followup",
    }


def _bounded_env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    raw = os.getenv(name)
    try:
        value = int(raw) if raw is not None and raw.strip() else default
    except ValueError:
        value = default
    return max(minimum, min(maximum, value))


def _misleading_target_per_macro() -> int:
    return _bounded_env_int("EVAL_MISLEADING_PER_MACRO", 1, 1, 2)


def _review_target_at_end() -> int:
    return _bounded_env_int("EVAL_REVIEW_AT_END", 2, 2, 3)


def _macro_misleading_count(eval_state: dict, macro_id: str | None) -> int:
    if not macro_id:
        return 0
    return int(
        eval_state.get("macro_states", {})
        .get(macro_id, {})
        .get("misleading_question_count", 0)
    )


def _choose_next_action_for_turn(eval_state: dict, judge_result: dict, turn: dict) -> str:
    base_action = choose_next_action(eval_state, judge_result)
    if base_action in {"hallucination_followup", "detail_followup", "end_failed"}:
        return base_action
    macro_id = turn.get("macro_id")
    question_type = turn.get("question_type")
    if (
        macro_id
        and question_type not in {"review_followup", "multi_hop_reasoning", *THREAD_QUESTION_TYPES}
        and _macro_misleading_count(eval_state, macro_id) < _misleading_target_per_macro()
    ):
        return "misleading_followup"
    return base_action


def _apply_effective_next_action(eval_state: dict, judge_result: dict, turn: dict) -> str:
    next_action = _choose_next_action_for_turn(eval_state, judge_result, turn)
    judge_result["next_action"] = next_action
    judge_result["policy_next_action"] = next_action
    return next_action


def _mark_evaluation_running(eval_state: dict) -> None:
    global_state = eval_state.setdefault("global_state", {})
    if global_state.get("evaluation_status") not in {"completed", "failed"}:
        global_state["evaluation_status"] = "running"
        global_state["completion_reason"] = None
        global_state["completed_at_turn"] = None


def _mark_evaluation_finished(eval_state: dict, status: str, reason: str, turn_no: int) -> None:
    global_state = eval_state.setdefault("global_state", {})
    global_state["evaluation_status"] = status
    global_state["completion_reason"] = reason
    global_state["completed_at_turn"] = turn_no
    if status == "failed":
        global_state["failed"] = True
        global_state["failure_reason"] = global_state.get("failure_reason") or reason


def _final_evaluation_status(eval_state: dict, turn_no: int, max_turns: int) -> tuple[str, str]:
    global_state = eval_state.get("global_state", {})
    if global_state.get("failed"):
        return "failed", global_state.get("failure_reason") or "failed"
    if max_turns and turn_no >= max_turns:
        return "stopped_by_max_turns", f"EVAL_MAX_TURNS reached: {max_turns}"
    return "completed", "all scheduled macro, follow-up, thread, and review turns finished"


def main() -> None:
    settings = load_settings(BASE_DIR.parent)
    use_online_eval = _env_bool("USE_ONLINE_EVAL")
    allow_mock_eval = _env_bool("ALLOW_MOCK_EVAL")
    allow_offline_fallback = (
        _env_bool("ALLOW_OFFLINE_FALLBACK")
        or allow_mock_eval
    )
    resume = _env_bool("PAPERGRAPH_RESUME") or _env_bool("EVAL_RESUME")
    restart = _env_bool("PAPERGRAPH_RESTART") or _env_bool("EVAL_RESTART")
    checkpoint_path = Path(os.getenv("EVAL_CHECKPOINT_PATH", str(EVAL_CHECKPOINT_PATH)))
    target_model = settings.llm_model or "mock-model"
    log(
        "evaluation configuration loaded",
        use_online_eval=use_online_eval,
        allow_mock_eval=allow_mock_eval,
        resume=resume,
        restart=restart,
        checkpoint=checkpoint_path,
    )

    log("loading graph and questions", graph=GRAPH_PATH, questions=QUESTION_PATH)
    graph = json.loads(GRAPH_PATH.read_text(encoding="utf-8"))
    questions = _repair_questions_for_graph(graph, json.loads(QUESTION_PATH.read_text(encoding="utf-8")))
    by_kc = {k["kc_id"]: k for k in graph.get("kc_nodes", [])}
    kc_bank = _load_kc_bank(graph)
    with span("load paper text"):
        paper_text = load_full_paper_text(graph, BASE_DIR)
    log(
        "evaluation inputs ready",
        kcs=len(by_kc),
        main_questions=len(questions.get("macro_main_questions", questions.get("main_questions", []))),
        thread_question_seeds=len(questions.get("thread_question_seeds", [])),
        multi_hop_questions=len(questions.get("multi_hop_questions", [])),
        kc_bank_kcs=len(kc_bank.get("kc_nodes", [])),
        paper_chars=len(paper_text),
    )

    client = OpenAICompatClient(ModelConfig(settings.api_key, settings.base_url, settings.llm_model))
    if (not use_online_eval or not client.is_ready()) and not allow_mock_eval:
        raise RuntimeError("Formal evaluation requires USE_ONLINE_EVAL=true and configured API_KEY/BASE_URL/LLM_MODEL. Set ALLOW_MOCK_EVAL=true only for local debugging.")
    checkpoint = load_eval_checkpoint(
        checkpoint_path,
        graph,
        target_model,
        _ensure_eval_state_defaults,
        _rebuild_macro_misleading_counts,
    ) if resume and not restart else None
    if checkpoint:
        eval_state, trajectory, turn_no, completed_question_ids, completed_thread_step_ids = checkpoint
        log(
            "evaluation checkpoint loaded",
            turns=len(trajectory.get("turns", [])),
            completed_questions=len(completed_question_ids),
            completed_thread_steps=len(completed_thread_step_ids),
            checkpoint=checkpoint_path,
        )
    else:
        eval_state = initialize_eval_state(graph, target_model=target_model)
        _ensure_eval_state_defaults(eval_state, graph)
        trajectory = {"paper_id": graph["paper_id"], "target_model": target_model, "turns": []}
        turn_no = 0
        completed_question_ids = set()
        completed_thread_step_ids = set()
    max_turns = int(os.getenv("EVAL_MAX_TURNS", "0") or "0")
    _mark_evaluation_running(eval_state)
    runner = EvaluationTurnRunner(
        graph=graph,
        by_kc=by_kc,
        paper_text=paper_text,
        client=client,
        use_online_eval=use_online_eval,
        allow_offline_fallback=allow_offline_fallback,
        kc_bank=kc_bank,
        claim_log_path=CLAIM_LOG_PATH,
    )

    macro_queue = questions.get("macro_main_questions") or questions.get("main_questions", [])
    queue = macro_queue + questions.get("multi_hop_questions", [])
    if completed_question_ids:
        queue = [q for q in queue if q.get("question_id") not in completed_question_ids]
    log("evaluation queue prepared", questions=len(queue), max_turns=max_turns, starting_turn=turn_no)
    try:
        for q in queue:
            if eval_state["global_state"]["failed"]:
                break
            if max_turns and turn_no >= max_turns:
                break
            turn_no, turn, target_kcs, next_action = runner.run_question_turn(
                q,
                trajectory,
                eval_state,
                turn_no,
                _apply_effective_next_action,
            )
            if not turn:
                continue
            completed_question_ids.add(q["question_id"])
            _save_eval_artifacts(graph, eval_state, trajectory, checkpoint_path)

            follow_depth = 0
            last_turn = trajectory["turns"][-1]
            last_judge = last_turn["judge_result"]
            last_targets = target_kcs
            seen_follow_keys = set()
            while follow_depth < 3 and not eval_state["global_state"]["failed"]:
                follow_action = last_judge.get("policy_next_action") or last_judge.get("next_action")
                if not follow_action:
                    follow_action = _apply_effective_next_action(eval_state, last_judge, last_turn)
                if not _is_immediate_followup(follow_action):
                    break
                if max_turns and turn_no >= max_turns:
                    break
                follow_targets = select_followup_target_kcs(
                    follow_action,
                    last_judge,
                    by_kc,
                    last_targets,
                    last_turn=last_turn,
                    trajectory=trajectory,
                )
                follow_key = (follow_action, tuple(k["kc_id"] for k in follow_targets))
                if follow_key in seen_follow_keys:
                    break
                seen_follow_keys.add(follow_key)
                follow = generate_followup_question(
                    follow_action,
                    last_turn,
                    follow_targets,
                    client,
                    allow_offline_fallback=allow_offline_fallback,
                )
                if not follow:
                    break
                log(
                    "immediate follow-up scheduled",
                    action=follow_action,
                    source_turn=last_turn.get("turn_id"),
                    targets=",".join(k["kc_id"] for k in follow_targets),
                )
                turn_no, last_turn, last_targets = runner.run_followup_turn(
                    follow,
                    trajectory,
                    eval_state,
                    turn_no,
                    _apply_effective_next_action,
                )
                if not last_turn:
                    break
                _save_eval_artifacts(graph, eval_state, trajectory, checkpoint_path)
                last_judge = last_turn["judge_result"]
                follow_depth += 1

            thread_budget = _bounded_env_int("EVAL_THREAD_TURNS_PER_CHECK", 1, 1, 3)
            while thread_budget > 0 and not eval_state["global_state"]["failed"]:
                if max_turns and turn_no >= max_turns:
                    break
                seed = get_ready_thread_turn(eval_state, graph.get("reasoning_threads", []), review_stage=False)
                if not seed:
                    break
                turn_no, thread_turn = runner.run_thread_turn(
                    seed,
                    trajectory,
                    eval_state,
                    turn_no,
                    _apply_effective_next_action,
                )
                if not thread_turn:
                    break
                _save_eval_artifacts(graph, eval_state, trajectory, checkpoint_path)
                thread_budget -= 1
            if eval_state["global_state"]["failed"]:
                log("evaluation failed", reason=eval_state["global_state"].get("failure_reason"))
                break
    except KeyboardInterrupt:
        _mark_evaluation_finished(eval_state, "interrupted", "KeyboardInterrupt", turn_no)
        _save_eval_artifacts(graph, eval_state, trajectory, checkpoint_path)
        log("evaluation interrupted; checkpoint saved", turns=len(trajectory.get("turns", [])), checkpoint=checkpoint_path)
        print(f"Evaluation interrupted. Checkpoint saved: {checkpoint_path}")
        return

    try:
        # End-of-dialogue review checks: keep these out of per-macro follow-up chains.
        end_targets = []
        if trajectory["turns"] and not (max_turns and turn_no >= max_turns):
            needed_reviews = max(
                0,
                _review_target_at_end() - eval_state["global_state"].get("review_question_count", 0),
            )
            review_sources = [
                t
                for t in reversed(trajectory["turns"])
                if t.get("question_type") in {"main", "multi_hop_reasoning"}
            ]
            if not review_sources:
                review_sources = [trajectory["turns"][-1]]
            for idx, source_turn in enumerate(review_sources[:needed_reviews], start=1):
                tks = [by_kc[k] for k in source_turn.get("target_kc_ids", []) if k in by_kc]
                if not tks:
                    continue
                q = generate_followup_question("review_followup", source_turn, tks, client, allow_offline_fallback=allow_offline_fallback)
                if q:
                    q["question_id"] = f"{source_turn['question_id']}_R{idx}"
                    end_targets.append(q)

        for follow in end_targets:
            if eval_state["global_state"]["failed"]:
                break
            if max_turns and turn_no >= max_turns:
                break
            turn_no, _, _ = runner.run_followup_turn(
                follow,
                trajectory,
                eval_state,
                turn_no,
                _apply_effective_next_action,
            )
            _save_eval_artifacts(graph, eval_state, trajectory, checkpoint_path)

        while not eval_state["global_state"]["failed"]:
            if max_turns and turn_no >= max_turns:
                break
            seed = get_ready_thread_turn(eval_state, graph.get("reasoning_threads", []), review_stage=True)
            if not seed:
                break
            turn_no, thread_turn = runner.run_thread_turn(
                seed,
                trajectory,
                eval_state,
                turn_no,
                _apply_effective_next_action,
            )
            if not thread_turn:
                break
            _save_eval_artifacts(graph, eval_state, trajectory, checkpoint_path)
    except KeyboardInterrupt:
        _mark_evaluation_finished(eval_state, "interrupted", "KeyboardInterrupt", turn_no)
        _save_eval_artifacts(graph, eval_state, trajectory, checkpoint_path)
        log("evaluation interrupted; checkpoint saved", turns=len(trajectory.get("turns", [])), checkpoint=checkpoint_path)
        print(f"Evaluation interrupted. Checkpoint saved: {checkpoint_path}")
        return

    final_status, final_reason = _final_evaluation_status(eval_state, turn_no, max_turns)
    _mark_evaluation_finished(eval_state, final_status, final_reason, turn_no)
    _save_eval_artifacts(graph, eval_state, trajectory, checkpoint_path)
    log(
        "evaluation artifacts written",
        turns=len(trajectory.get("turns", [])),
        trajectory=TRAJ_PATH,
        report=REPORT_PATH,
        state=STATE_PATH,
    )
    print(f"Trajectory written: {TRAJ_PATH}")
    print(f"Report written: {REPORT_PATH}")
    print(f"Eval state written: {STATE_PATH}")


if __name__ == "__main__":
    main()
