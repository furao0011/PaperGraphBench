import json
import os
import re
from pathlib import Path

from src.config import load_settings
from src.dialogue_engine import generate_followup_question
from src.judge import judge_answer_with_online_fallback
from src.mermaid_exporter import export_final_state_mermaid
from src.model_client import ModelConfig, OpenAICompatClient
from src.paper_parser import load_paper_text, load_paper_text_from_dir
from src.policy_controller import choose_next_action
from src.progress import log, span
from src.question_generator import normalize_question_bundle
from src.reporter import build_report
from src.state_updater import apply_judge_result, initialize_eval_state


BASE_DIR = Path(__file__).resolve().parent
GRAPH_PATH = BASE_DIR / "data" / "graphs" / "master_graph.json"
QUESTION_PATH = BASE_DIR / "data" / "questions" / "question_templates.json"
TRAJ_PATH = BASE_DIR / "data" / "outputs" / "dialogue_trajectory.json"
REPORT_PATH = BASE_DIR / "data" / "outputs" / "evaluation_report.json"
STATE_PATH = BASE_DIR / "data" / "graphs" / "eval_state_graph.json"
FINAL_MMD_PATH = BASE_DIR / "data" / "graphs" / "final_state_graph.mmd"
EVAL_CHECKPOINT_PATH = BASE_DIR / "data" / "outputs" / "evaluation_checkpoint.json"


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _turn_number(turn_id: str) -> int:
    match = re.search(r"(\d+)$", str(turn_id))
    return int(match.group(1)) if match else 0


def _save_eval_artifacts(graph: dict, eval_state: dict, trajectory: dict, checkpoint_path: Path) -> None:
    report = build_report(eval_state, trajectory)
    _write_json(TRAJ_PATH, trajectory)
    _write_json(REPORT_PATH, report)
    _write_json(STATE_PATH, eval_state)
    FINAL_MMD_PATH.parent.mkdir(parents=True, exist_ok=True)
    FINAL_MMD_PATH.write_text(export_final_state_mermaid(graph, eval_state), encoding="utf-8")
    completed_question_ids = [
        t.get("question_id")
        for t in trajectory.get("turns", [])
        if t.get("question_type") in {"main", "multi_hop_reasoning"}
    ]
    _write_json(
        checkpoint_path,
        {
            "paper_id": graph.get("paper_id", "unknown"),
            "target_model": trajectory.get("target_model"),
            "turn_no": max((_turn_number(t.get("turn_id", "")) for t in trajectory.get("turns", [])), default=0),
            "completed_question_ids": completed_question_ids,
            "trajectory": trajectory,
            "eval_state": eval_state,
        },
    )


def _load_eval_checkpoint(checkpoint_path: Path, graph: dict, target_model: str) -> tuple[dict, dict, int, set[str]] | None:
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
            if t.get("question_type") in {"main", "multi_hop_reasoning"} and t.get("question_id")
        }
    return eval_state, trajectory, turn_no, completed


def _mock_answer(question: str, target_kcs: list[dict]) -> str:
    joined = "; ".join(k["full_claim"] for k in target_kcs[:2])
    return f"Based on the paper, {joined}"


def _build_model_answer(client: OpenAICompatClient | None, use_online_eval: bool, prompt: str, target_kcs: list[dict]) -> tuple[str, str]:
    if use_online_eval and client and client.is_ready():
        ans = client.chat_text(
            system_prompt="Answer the paper-evaluation question based only on provided context.",
            user_prompt=prompt,
        )
        return ans, "online"
    if os.getenv("ALLOW_MOCK_EVAL", "false").lower() in {"1", "true", "yes", "on"}:
        return _mock_answer(prompt, target_kcs), "mock"
    raise RuntimeError("Online evaluation requires a configured model and USE_ONLINE_EVAL=true. Set ALLOW_MOCK_EVAL=true only for local debugging.")


def _dialogue_summary(turns: list[dict], keep_last: int = 4) -> str:
    if not turns:
        return ""
    items = turns[-keep_last:]
    lines = []
    for t in items:
        q = t.get("question_text", "")[:180]
        a = t.get("model_answer", "")[:220]
        lines.append(f"{t.get('turn_id')}: Q={q} | A={a}")
    return "\n".join(lines)


def _related_forbidden_claims(graph: dict, target_kc_ids: list[str], path_id: str | None) -> list[dict]:
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


def _load_full_paper_text(graph: dict) -> str:
    paper_path = Path(graph.get("paper_text_path", ""))
    if paper_path.is_dir():
        text = load_paper_text_from_dir(paper_path)
    elif paper_path.is_file():
        text = load_paper_text(paper_path)
    else:
        text = "\n".join(k["full_claim"] for k in graph.get("kc_nodes", []))
    limit = int(os.getenv("EVAL_PAPER_CHAR_LIMIT", "0") or "0")
    return text[:limit] if limit > 0 else text


def _repair_questions_for_graph(graph: dict, questions: dict) -> dict:
    return {
        "paper_id": graph.get("paper_id"),
        **normalize_question_bundle(graph, questions),
    }


def _select_followup_target_kcs(
    action: str,
    judge_result: dict,
    by_kc: dict[str, dict],
    fallback_kcs: list[dict],
) -> list[dict]:
    if action == "detail_followup":
        ids = judge_result.get("missing_kc_ids", [])
    elif action == "hallucination_followup":
        ids = judge_result.get("covered_kc_ids", []) + judge_result.get("missing_kc_ids", [])
    else:
        ids = []
    selected = [by_kc[kid] for kid in ids if kid in by_kc]
    return selected or fallback_kcs[:1]


def _normalize_judge_result_for_turn(judge_result: dict, answer: str, question_type: str) -> dict:
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
    if question_type != "hallucination_followup" or judge_result.get("state") != "HALLUCINATION":
        return judge_result
    answer_l = answer.lower()
    correction_cues = [
        "not supported",
        "unsupported",
        "not mentioned",
        "does not appear",
        "no mention",
        "no detailed discussion",
        "not in the paper",
        "needs correction",
        "should be corrected",
        "partially supported",
        "不支持",
        "未提到",
        "没有提到",
        "需要修正",
    ]
    if any(cue in answer_l for cue in correction_cues):
        fixed = dict(judge_result)
        fixed["state"] = "SELF_CORRECTED" if fixed.get("covered_kc_ids") else "MAIN_PROGRESS"
        fixed["next_action"] = "next_main_question"
        fixed["hallucinated_claims"] = []
        fixed["matched_forbidden_claims"] = []
        explanation = fixed.get("judge_explanation", "")
        fixed["judge_explanation"] = (
            explanation
            + " Normalized: this hallucination follow-up answer retracts or marks suspected claims as unsupported."
        ).strip()
        return fixed
    return judge_result


def _run_followup_turn(
    follow: dict,
    by_kc: dict[str, dict],
    trajectory: dict,
    eval_state: dict,
    graph: dict,
    client: OpenAICompatClient,
    use_online_eval: bool,
    turn_no: int,
    allow_offline_fallback: bool,
) -> tuple[int, dict | None, list[dict]]:
    turn_no += 1
    f_turn_id = f"T{turn_no}"
    tks = [by_kc[k] for k in follow.get("target_kc_ids", []) if k in by_kc]
    if not tks:
        return turn_no, None, []
    log(
        "follow-up turn started",
        turn=f_turn_id,
        question_id=follow.get("question_id"),
        question_type=follow.get("question_type"),
        targets=",".join(follow.get("target_kc_ids", [])),
    )
    if follow["question_type"] == "misleading_followup":
        eval_state["global_state"]["misleading_question_count"] += 1
    if follow["question_type"] == "review_followup":
        eval_state["global_state"]["review_question_count"] += 1
    history_text = "\n".join([f"{t['turn_id']} Q:{t['question_text']} A:{t['model_answer']}" for t in trajectory["turns"]])
    f_input = f"[dialogue history]\n{history_text}\n\nCurrent Question:\n{follow['question_text']}"
    with span("target model answer", turn=f_turn_id):
        f_answer, f_mode = _build_model_answer(client, use_online_eval, f_input, tks)
    f_dsum = _dialogue_summary(trajectory["turns"])
    f_rel_forbidden = _related_forbidden_claims(
        graph, follow.get("target_kc_ids", []), follow.get("target_path_id")
    )
    with span("judge answer", turn=f_turn_id):
        f_judge = judge_answer_with_online_fallback(
            follow["question_text"],
            f_answer,
            tks,
            client,
            use_online_judge=use_online_eval,
            dialogue_summary=f_dsum,
            related_forbidden_claims=f_rel_forbidden,
        )
    f_judge = _normalize_judge_result_for_turn(f_judge, f_answer, follow["question_type"])
    f_state_update = apply_judge_result(eval_state, turn_id=f_turn_id, judge_result=f_judge, path_id=follow.get("target_path_id"))
    turn = {
        "turn_id": f_turn_id,
        "question_id": follow["question_id"],
        "question_type": follow["question_type"],
        "macro_id": follow.get("macro_id"),
        "question_text": follow["question_text"],
        "target_kc_ids": follow["target_kc_ids"],
        "target_path_id": follow.get("target_path_id"),
        "model_answer": f_answer,
        "answer_mode": f_mode,
        "judge_result": f_judge,
        "state_update": f_state_update,
    }
    trajectory["turns"].append(turn)
    log(
        "follow-up turn judged",
        turn=f_turn_id,
        state=f_judge.get("state"),
        next_action=f_judge.get("next_action"),
        covered=len(f_judge.get("covered_kc_ids", [])),
        missing=len(f_judge.get("missing_kc_ids", [])),
        hallucinations=len(f_judge.get("hallucinated_claims", [])),
        answer_mode=f_mode,
    )
    return turn_no, turn, tks


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
    with span("load paper text"):
        paper_text = _load_full_paper_text(graph)
    log(
        "evaluation inputs ready",
        kcs=len(by_kc),
        main_questions=len(questions.get("main_questions", [])),
        multi_hop_questions=len(questions.get("multi_hop_questions", [])),
        paper_chars=len(paper_text),
    )

    client = OpenAICompatClient(ModelConfig(settings.api_key, settings.base_url, settings.llm_model))
    if (not use_online_eval or not client.is_ready()) and not allow_mock_eval:
        raise RuntimeError("Formal evaluation requires USE_ONLINE_EVAL=true and configured API_KEY/BASE_URL/LLM_MODEL. Set ALLOW_MOCK_EVAL=true only for local debugging.")
    checkpoint = _load_eval_checkpoint(checkpoint_path, graph, target_model) if resume and not restart else None
    if checkpoint:
        eval_state, trajectory, turn_no, completed_question_ids = checkpoint
        log(
            "evaluation checkpoint loaded",
            turns=len(trajectory.get("turns", [])),
            completed_questions=len(completed_question_ids),
            checkpoint=checkpoint_path,
        )
    else:
        eval_state = initialize_eval_state(graph, target_model=target_model)
        trajectory = {"paper_id": graph["paper_id"], "target_model": target_model, "turns": []}
        turn_no = 0
        completed_question_ids = set()
    max_turns = int(os.getenv("EVAL_MAX_TURNS", "0") or "0")

    queue = questions.get("main_questions", []) + questions.get("multi_hop_questions", [])
    if completed_question_ids:
        queue = [q for q in queue if q.get("question_id") not in completed_question_ids]
    log("evaluation queue prepared", questions=len(queue), max_turns=max_turns, starting_turn=turn_no)
    try:
        for q in queue:
            if eval_state["global_state"]["failed"]:
                break
            if max_turns and turn_no >= max_turns:
                break
            turn_no += 1
            turn_id = f"T{turn_no}"
            target_kcs = [by_kc[k] for k in q.get("target_kc_ids", []) if k in by_kc]
            if not target_kcs:
                continue
            log(
                "turn started",
                turn=turn_id,
                question_id=q.get("question_id"),
                question_type=q.get("question_type"),
                targets=",".join(q.get("target_kc_ids", [])),
            )
            history_text = "\n".join([f"{t['turn_id']} Q:{t['question_text']} A:{t['model_answer']}" for t in trajectory["turns"]])
            if turn_no == 1:
                model_input = f"[paper text]\n{paper_text}\n\nQuestion:\n{q['question_text']}"
            else:
                model_input = f"[dialogue history]\n{history_text}\n\nCurrent Question:\n{q['question_text']}"

            with span("target model answer", turn=turn_id):
                answer, answer_mode = _build_model_answer(client, use_online_eval, model_input, target_kcs)
            dsum = _dialogue_summary(trajectory["turns"])
            rel_forbidden = _related_forbidden_claims(graph, q.get("target_kc_ids", []), q.get("path_id"))
            with span("judge answer", turn=turn_id):
                judge_result = judge_answer_with_online_fallback(
                    q["question_text"],
                    answer,
                    target_kcs,
                    client,
                    use_online_judge=use_online_eval,
                    dialogue_summary=dsum,
                    related_forbidden_claims=rel_forbidden,
                )
            judge_result = _normalize_judge_result_for_turn(judge_result, answer, q["question_type"])
            next_action = choose_next_action(eval_state, judge_result)
            judge_result["next_action"] = next_action
            state_update = apply_judge_result(eval_state, turn_id=turn_id, judge_result=judge_result, path_id=q.get("path_id"))

            trajectory["turns"].append(
                {
                    "turn_id": turn_id,
                    "question_id": q["question_id"],
                    "question_type": q["question_type"],
                    "macro_id": q.get("macro_id"),
                    "question_text": q["question_text"],
                    "target_kc_ids": q["target_kc_ids"],
                    "target_path_id": q.get("path_id"),
                    "model_answer": answer,
                    "answer_mode": answer_mode,
                    "judge_result": judge_result,
                    "state_update": state_update,
                }
            )
            completed_question_ids.add(q["question_id"])
            _save_eval_artifacts(graph, eval_state, trajectory, checkpoint_path)
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

            follow_depth = 0
            last_turn = trajectory["turns"][-1]
            last_judge = judge_result
            last_targets = target_kcs
            seen_follow_keys = set()
            while follow_depth < 3 and not eval_state["global_state"]["failed"]:
                follow_action = choose_next_action(eval_state, last_judge)
                if follow_action not in {"detail_followup", "hallucination_followup"}:
                    break
                if max_turns and turn_no >= max_turns:
                    break
                follow_targets = _select_followup_target_kcs(follow_action, last_judge, by_kc, last_targets)
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
                turn_no, last_turn, last_targets = _run_followup_turn(
                    follow,
                    by_kc,
                    trajectory,
                    eval_state,
                    graph,
                    client,
                    use_online_eval,
                    turn_no,
                    allow_offline_fallback,
                )
                if not last_turn:
                    break
                _save_eval_artifacts(graph, eval_state, trajectory, checkpoint_path)
                last_judge = last_turn["judge_result"]
                follow_depth += 1
            if eval_state["global_state"]["failed"]:
                log("evaluation failed", reason=eval_state["global_state"].get("failure_reason"))
                break
    except KeyboardInterrupt:
        _save_eval_artifacts(graph, eval_state, trajectory, checkpoint_path)
        log("evaluation interrupted; checkpoint saved", turns=len(trajectory.get("turns", [])), checkpoint=checkpoint_path)
        print(f"Evaluation interrupted. Checkpoint saved: {checkpoint_path}")
        return

    try:
        # Guidance-aligned quotas: misleading 1-2 times, review 1-2 times near end.
        end_targets = []
        if trajectory["turns"] and not (max_turns and turn_no >= max_turns):
            last_turn = trajectory["turns"][-1]
            tks = [by_kc[k] for k in last_turn.get("target_kc_ids", []) if k in by_kc]
            if eval_state["global_state"]["misleading_question_count"] < 1:
                q = generate_followup_question("misleading_followup", last_turn, tks, client, allow_offline_fallback=allow_offline_fallback)
                if q:
                    end_targets.append(q)
            if eval_state["global_state"]["review_question_count"] < 1:
                q = generate_followup_question("review_followup", last_turn, tks, client, allow_offline_fallback=allow_offline_fallback)
                if q:
                    end_targets.append(q)

        for follow in end_targets:
            if eval_state["global_state"]["failed"]:
                break
            if max_turns and turn_no >= max_turns:
                break
            turn_no, _, _ = _run_followup_turn(
                follow,
                by_kc,
                trajectory,
                eval_state,
                graph,
                client,
                use_online_eval,
                turn_no,
                allow_offline_fallback,
            )
            _save_eval_artifacts(graph, eval_state, trajectory, checkpoint_path)
    except KeyboardInterrupt:
        _save_eval_artifacts(graph, eval_state, trajectory, checkpoint_path)
        log("evaluation interrupted; checkpoint saved", turns=len(trajectory.get("turns", [])), checkpoint=checkpoint_path)
        print(f"Evaluation interrupted. Checkpoint saved: {checkpoint_path}")
        return

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
