import json
import os
from pathlib import Path

from src.artifact_layout import PaperArtifactLayout
from src.challenge_scheduler import ensure_challenge_states
from src.config import load_settings
from src.eval_artifacts import load_eval_checkpoint, save_eval_artifacts
from src.eval_turn_runner import EvaluationTurnRunner
from src.evaluation_inputs import load_kc_bank, repair_questions_for_graph
from src.evaluation_stage_runner import EvaluationStageRunner
from src.evaluation_state import (
    ensure_eval_state_defaults,
    final_evaluation_status,
    mark_evaluation_finished,
    mark_evaluation_running,
    rebuild_eval_turn_counts,
)
from src.mermaid_exporter import export_final_thread_state_mermaid
from src.model_client import ModelConfig, OpenAICompatClient
from src.paper_context import load_full_paper_text
from src.progress import log, span
from src.state_updater import initialize_eval_state


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


def _apply_paper_layout_from_env() -> PaperArtifactLayout | None:
    paper_id = os.getenv("PAPER_ID", "").strip()
    if not paper_id:
        return None
    layout = PaperArtifactLayout(BASE_DIR, paper_id)
    _configure_layout_paths(layout)
    return layout


def _configure_layout_paths(layout: PaperArtifactLayout) -> None:
    global GRAPH_PATH, QUESTION_PATH, TRAJ_PATH, REPORT_PATH, STATE_PATH
    global FINAL_MMD_PATH, FINAL_THREAD_MMD_PATH, EVAL_CHECKPOINT_PATH, CLAIM_LOG_PATH

    GRAPH_PATH = layout.final("master_graph")
    QUESTION_PATH = layout.final("question_templates")
    TRAJ_PATH = layout.final("dialogue_trajectory")
    REPORT_PATH = layout.final("evaluation_report")
    STATE_PATH = layout.final("eval_state_graph")
    FINAL_MMD_PATH = layout.cache_file("evaluation", "final_state_graph")
    FINAL_THREAD_MMD_PATH = layout.cache_file("evaluation", "final_thread_state_graph")
    EVAL_CHECKPOINT_PATH = layout.cache_file("evaluation", "evaluation_checkpoint")
    CLAIM_LOG_PATH = layout.cache_file("evaluation", "claim_verification_log")


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Formal evaluation requires {name} for the evaluated target model.")
    return value


def _env_path(name: str, default: Path) -> Path:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return Path(raw.strip())


def _build_eval_target_client() -> OpenAICompatClient:
    return OpenAICompatClient(
        ModelConfig(
            api_key=_required_env("EVAL_TARGET_API_KEY"),
            base_url=_required_env("EVAL_TARGET_BASE_URL"),
            llm_model=_required_env("EVAL_TARGET_MODEL"),
        )
    )


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


def main() -> None:
    settings = load_settings(BASE_DIR.parent)
    _apply_paper_layout_from_env()
    graph_override = os.getenv("PAPERGRAPH_GRAPH_PATH", "").strip()
    question_override = os.getenv("PAPERGRAPH_QUESTION_PATH", "").strip()
    if graph_override:
        global GRAPH_PATH
        GRAPH_PATH = Path(graph_override)
    if question_override:
        global QUESTION_PATH
        QUESTION_PATH = Path(question_override)
    use_online_eval = _env_bool("USE_ONLINE_EVAL")
    allow_mock_eval = _env_bool("ALLOW_MOCK_EVAL")
    allow_offline_fallback = (
        _env_bool("ALLOW_OFFLINE_FALLBACK")
        or allow_mock_eval
    )
    resume = _env_bool("PAPERGRAPH_RESUME") or _env_bool("EVAL_RESUME")
    restart = _env_bool("PAPERGRAPH_RESTART") or _env_bool("EVAL_RESTART")
    checkpoint_path = _env_path("EVAL_CHECKPOINT_PATH", EVAL_CHECKPOINT_PATH)
    client = OpenAICompatClient(ModelConfig(settings.api_key, settings.base_url, settings.llm_model))
    target_client = _build_eval_target_client() if use_online_eval else client
    target_model = target_client.cfg.llm_model if target_client and target_client.is_ready() else "mock-model"
    log(
        "evaluation configuration loaded",
        use_online_eval=use_online_eval,
        allow_mock_eval=allow_mock_eval,
        judge_model=settings.llm_model,
        target_model=target_model,
        resume=resume,
        restart=restart,
        checkpoint=checkpoint_path,
    )

    log("loading graph and questions", graph=GRAPH_PATH, questions=QUESTION_PATH)
    graph = json.loads(GRAPH_PATH.read_text(encoding="utf-8"))
    questions = repair_questions_for_graph(graph, json.loads(QUESTION_PATH.read_text(encoding="utf-8")))
    by_kc = {k["kc_id"]: k for k in graph.get("kc_nodes", [])}
    kc_bank = load_kc_bank(graph, BASE_DIR)
    with span("load Storybench text"):
        paper_text = load_full_paper_text(graph, BASE_DIR)
    log(
        "evaluation inputs ready",
        kcs=len(by_kc),
        main_questions=len(questions.get("macro_main_questions", questions.get("main_questions", []))),
        thread_question_seeds=len(questions.get("thread_question_seeds", [])),
        challenge_questions=len(questions.get("challenge_questions", [])),
        thread_challenge_questions=len(questions.get("thread_challenge_questions", [])),
        legacy_path_questions=len(questions.get("multi_hop_questions", [])),
        kc_bank_kcs=len(kc_bank.get("kc_nodes", [])),
        paper_chars=len(paper_text),
    )

    if (not use_online_eval or not client.is_ready()) and not allow_mock_eval:
        raise RuntimeError("Formal evaluation requires USE_ONLINE_EVAL=true and configured API_KEY/BASE_URL/LLM_MODEL. Set ALLOW_MOCK_EVAL=true only for local debugging.")
    if use_online_eval and not allow_mock_eval and not target_client.is_ready():
        raise RuntimeError("Formal evaluation requires configured EVAL_TARGET_API_KEY/EVAL_TARGET_BASE_URL/EVAL_TARGET_MODEL for the evaluated model.")
    checkpoint = load_eval_checkpoint(
        checkpoint_path,
        graph,
        target_model,
        ensure_eval_state_defaults,
        rebuild_eval_turn_counts,
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
        ensure_eval_state_defaults(eval_state, graph)
        trajectory = {"paper_id": graph["paper_id"], "target_model": target_model, "turns": []}
        turn_no = 0
        completed_question_ids = set()
        completed_thread_step_ids = set()
    max_turns = int(os.getenv("EVAL_MAX_TURNS", "0") or "0")
    mark_evaluation_running(eval_state)
    ensure_challenge_states(
        eval_state,
        questions.get("challenge_questions", []) + questions.get("thread_challenge_questions", []),
    )
    runner = EvaluationTurnRunner(
        graph=graph,
        by_kc=by_kc,
        paper_text=paper_text,
        client=client,
        target_client=target_client,
        use_online_eval=use_online_eval,
        allow_offline_fallback=allow_offline_fallback,
        kc_bank=kc_bank,
        claim_log_path=CLAIM_LOG_PATH,
    )
    save_current_artifacts = lambda state, turns, _turn_no: _save_eval_artifacts(
        graph,
        state,
        turns,
        checkpoint_path,
    )
    stage_runner = EvaluationStageRunner(
        runner=runner,
        graph=graph,
        questions=questions,
        trajectory=trajectory,
        eval_state=eval_state,
        completed_question_ids=completed_question_ids,
        by_kc=by_kc,
        client=client,
        allow_offline_fallback=allow_offline_fallback,
        save_artifacts=save_current_artifacts,
        max_turns=max_turns,
    )

    macro_queue = questions.get("macro_main_questions") or questions.get("main_questions", [])
    queue = macro_queue + questions.get("multi_hop_questions", [])
    if completed_question_ids:
        queue = [q for q in queue if q.get("question_id") not in completed_question_ids]
    log("macro-stage evaluation queue prepared", questions=len(queue), max_turns=max_turns, starting_turn=turn_no)
    try:
        for q in queue:
            if eval_state["global_state"]["failed"]:
                break
            if max_turns and turn_no >= max_turns:
                break
            turn_no = stage_runner.run_macro_stage(q, turn_no)
            if eval_state["global_state"]["failed"]:
                log("evaluation failed", reason=eval_state["global_state"].get("failure_reason"))
                break
    except KeyboardInterrupt:
        mark_evaluation_finished(eval_state, "interrupted", "KeyboardInterrupt", turn_no)
        _save_eval_artifacts(graph, eval_state, trajectory, checkpoint_path)
        log("evaluation interrupted; checkpoint saved", turns=len(trajectory.get("turns", [])), checkpoint=checkpoint_path)
        print(f"Evaluation interrupted. Checkpoint saved: {checkpoint_path}")
        return

    try:
        turn_no = stage_runner.run_review_stage(turn_no)
    except KeyboardInterrupt:
        mark_evaluation_finished(eval_state, "interrupted", "KeyboardInterrupt", turn_no)
        _save_eval_artifacts(graph, eval_state, trajectory, checkpoint_path)
        log("evaluation interrupted; checkpoint saved", turns=len(trajectory.get("turns", [])), checkpoint=checkpoint_path)
        print(f"Evaluation interrupted. Checkpoint saved: {checkpoint_path}")
        return

    final_status, final_reason = final_evaluation_status(eval_state, turn_no, max_turns)
    mark_evaluation_finished(eval_state, final_status, final_reason, turn_no)
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
