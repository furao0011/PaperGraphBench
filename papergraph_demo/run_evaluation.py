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
    _ensure_eval_state_defaults(eval_state, graph)
    _rebuild_macro_misleading_counts(eval_state, trajectory)
    return eval_state, trajectory, turn_no, completed


def _ensure_eval_state_defaults(eval_state: dict, graph: dict) -> None:
    eval_state.setdefault("macro_states", {})
    for macro in graph.get("macro_nodes", []):
        macro_id = macro.get("macro_id")
        if not macro_id:
            continue
        eval_state["macro_states"].setdefault(macro_id, {})
        eval_state["macro_states"][macro_id].setdefault("misleading_question_count", 0)
    eval_state.setdefault("global_state", {})
    eval_state["global_state"].setdefault("misleading_question_count", 0)
    eval_state["global_state"].setdefault("review_question_count", 0)


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


def _mock_answer(question: str, target_kcs: list[dict]) -> str:
    joined = "; ".join(k["full_claim"] for k in target_kcs[:2])
    return f"Based on the paper, {joined}"


def _build_model_answer(client: OpenAICompatClient | None, use_online_eval: bool, prompt: str, target_kcs: list[dict]) -> tuple[str, str]:
    if use_online_eval and client and client.is_ready():
        ans = client.chat_text(
            system_prompt="Answer the paper-evaluation question based only on the provided original paper and dialogue context.",
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


def _dialogue_history_text(turns: list[dict]) -> str:
    if not turns:
        return "No previous turns."
    return "\n".join(
        f"{t['turn_id']} Q:{t['question_text']} A:{t['model_answer']}"
        for t in turns
    )


def _build_eval_prompt(paper_text: str, dialogue_history: str, question_text: str) -> str:
    return (
        "```original paper\n"
        f"{paper_text}\n"
        "```\n\n"
        "[dialogue history]\n"
        f"{dialogue_history}\n\n"
        "[current question]\n"
        f"{question_text}"
    )


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
        and question_type not in {"review_followup", "multi_hop_reasoning"}
        and _macro_misleading_count(eval_state, macro_id) < _misleading_target_per_macro()
    ):
        return "misleading_followup"
    return base_action


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
    return judge_result


def _run_followup_turn(
    follow: dict,
    by_kc: dict[str, dict],
    trajectory: dict,
    eval_state: dict,
    graph: dict,
    paper_text: str,
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
        macro_id = follow.get("macro_id")
        if macro_id:
            eval_state.setdefault("macro_states", {}).setdefault(macro_id, {})
            current = eval_state["macro_states"][macro_id].get("misleading_question_count", 0)
            eval_state["macro_states"][macro_id]["misleading_question_count"] = current + 1
    if follow["question_type"] == "review_followup":
        eval_state["global_state"]["review_question_count"] += 1
    f_input = _build_eval_prompt(
        paper_text=paper_text,
        dialogue_history=_dialogue_history_text(trajectory["turns"]),
        question_text=follow["question_text"],
    )
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
            question_type=follow["question_type"],
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
        _ensure_eval_state_defaults(eval_state, graph)
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
            model_input = _build_eval_prompt(
                paper_text=paper_text,
                dialogue_history=_dialogue_history_text(trajectory["turns"]),
                question_text=q["question_text"],
            )

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
                    question_type=q["question_type"],
                )
            judge_result = _normalize_judge_result_for_turn(judge_result, answer, q["question_type"])
            next_action = _choose_next_action_for_turn(
                eval_state,
                judge_result,
                {"question_type": q["question_type"], "macro_id": q.get("macro_id")},
            )
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
                follow_action = _choose_next_action_for_turn(eval_state, last_judge, last_turn)
                if not _is_immediate_followup(follow_action):
                    break
                if max_turns and turn_no >= max_turns:
                    break
                follow_targets = _select_followup_target_kcs(
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
                turn_no, last_turn, last_targets = _run_followup_turn(
                    follow,
                    by_kc,
                    trajectory,
                    eval_state,
                    graph,
                    paper_text,
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
            turn_no, _, _ = _run_followup_turn(
                follow,
                by_kc,
                trajectory,
                eval_state,
                graph,
                paper_text,
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
