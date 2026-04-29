import json
import os
from pathlib import Path

from src.config import load_settings
from src.dialogue_engine import generate_followup_question
from src.judge import judge_answer_with_online_fallback
from src.mermaid_exporter import export_final_state_mermaid
from src.model_client import ModelConfig, OpenAICompatClient
from src.policy_controller import choose_next_action
from src.reporter import build_report
from src.state_updater import apply_judge_result, initialize_eval_state


BASE_DIR = Path(__file__).resolve().parent
GRAPH_PATH = BASE_DIR / "data" / "graphs" / "master_graph.json"
QUESTION_PATH = BASE_DIR / "data" / "questions" / "question_templates.json"
TRAJ_PATH = BASE_DIR / "data" / "outputs" / "dialogue_trajectory.json"
REPORT_PATH = BASE_DIR / "data" / "outputs" / "evaluation_report.json"
STATE_PATH = BASE_DIR / "data" / "graphs" / "eval_state_graph.json"
FINAL_MMD_PATH = BASE_DIR / "data" / "graphs" / "final_state_graph.mmd"


def _mock_answer(question: str, target_kcs: list[dict]) -> str:
    joined = "; ".join(k["full_claim"] for k in target_kcs[:2])
    return f"Based on the paper, {joined}"


def _build_model_answer(client: OpenAICompatClient | None, use_online_eval: bool, prompt: str, target_kcs: list[dict]) -> tuple[str, str]:
    if use_online_eval and client and client.is_ready():
        try:
            ans = client.chat_text(
                system_prompt="Answer the paper-evaluation question based only on provided context.",
                user_prompt=prompt,
            )
            return ans, "online"
        except Exception as e:
            return f"{_mock_answer(prompt, target_kcs)} (fallback: {type(e).__name__})", "fallback"
    return _mock_answer(prompt, target_kcs), "mock"


def main() -> None:
    settings = load_settings(BASE_DIR.parent)
    use_online_eval = os.getenv("USE_ONLINE_EVAL", "false").lower() in {"1", "true", "yes", "on"}
    target_model = settings.llm_model or "mock-model"

    graph = json.loads(GRAPH_PATH.read_text(encoding="utf-8"))
    questions = json.loads(QUESTION_PATH.read_text(encoding="utf-8"))
    by_kc = {k["kc_id"]: k for k in graph.get("kc_nodes", [])}
    paper_text_hint = "\n".join(k["full_claim"] for k in graph.get("kc_nodes", []))

    client = OpenAICompatClient(ModelConfig(settings.api_key, settings.base_url, settings.llm_model))
    eval_state = initialize_eval_state(graph, target_model=target_model)
    trajectory = {"paper_id": graph["paper_id"], "target_model": target_model, "turns": []}

    queue = questions.get("main_questions", []) + questions.get("multi_hop_questions", [])
    turn_no = 0
    for q in queue:
        if eval_state["global_state"]["failed"]:
            break
        turn_no += 1
        turn_id = f"T{turn_no}"
        target_kcs = [by_kc[k] for k in q.get("target_kc_ids", []) if k in by_kc]
        if not target_kcs:
            continue
        history_text = "\n".join([f"{t['turn_id']} Q:{t['question_text']} A:{t['model_answer']}" for t in trajectory["turns"]])
        if turn_no == 1:
            model_input = f"[paper text]\n{paper_text_hint}\n\nQuestion:\n{q['question_text']}"
        else:
            model_input = f"[dialogue history]\n{history_text}\n\nCurrent Question:\n{q['question_text']}"

        answer, answer_mode = _build_model_answer(client, use_online_eval, model_input, target_kcs)
        judge_result = judge_answer_with_online_fallback(
            q["question_text"], answer, target_kcs, client, use_online_judge=use_online_eval
        )
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

        # Runtime dynamic follow-up generation (v0 constrained): at most one per turn.
        follow = generate_followup_question(next_action, trajectory["turns"][-1], target_kcs, client)
        if follow:
            if follow["question_type"] == "misleading_followup":
                eval_state["global_state"]["misleading_question_count"] += 1
            if follow["question_type"] == "review_followup":
                eval_state["global_state"]["review_question_count"] += 1

            turn_no += 1
            f_turn_id = f"T{turn_no}"
            history_text = "\n".join([f"{t['turn_id']} Q:{t['question_text']} A:{t['model_answer']}" for t in trajectory["turns"]])
            f_input = f"[dialogue history]\n{history_text}\n\nCurrent Question:\n{follow['question_text']}"
            f_answer, f_mode = _build_model_answer(client, use_online_eval, f_input, target_kcs)
            f_judge = judge_answer_with_online_fallback(
                follow["question_text"], f_answer, target_kcs, client, use_online_judge=use_online_eval
            )
            f_state_update = apply_judge_result(eval_state, turn_id=f_turn_id, judge_result=f_judge, path_id=follow.get("target_path_id"))

            trajectory["turns"].append(
                {
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
            )
            if eval_state["global_state"]["failed"]:
                break

    # Guidance-aligned quotas: misleading 1-2 times, review 1-2 times near end.
    end_targets = []
    if trajectory["turns"]:
        last_turn = trajectory["turns"][-1]
        tks = [by_kc[k] for k in last_turn.get("target_kc_ids", []) if k in by_kc]
        if eval_state["global_state"]["misleading_question_count"] < 1:
            q = generate_followup_question("misleading_followup", last_turn, tks, client)
            if q:
                end_targets.append(q)
                eval_state["global_state"]["misleading_question_count"] += 1
        if eval_state["global_state"]["review_question_count"] < 1:
            q = generate_followup_question("review_followup", last_turn, tks, client)
            if q:
                end_targets.append(q)
                eval_state["global_state"]["review_question_count"] += 1

    for follow in end_targets:
        if eval_state["global_state"]["failed"]:
            break
        turn_no += 1
        f_turn_id = f"T{turn_no}"
        tks = [by_kc[k] for k in follow.get("target_kc_ids", []) if k in by_kc]
        history_text = "\n".join([f"{t['turn_id']} Q:{t['question_text']} A:{t['model_answer']}" for t in trajectory["turns"]])
        f_input = f"[dialogue history]\n{history_text}\n\nCurrent Question:\n{follow['question_text']}"
        f_answer, f_mode = _build_model_answer(client, use_online_eval, f_input, tks)
        f_judge = judge_answer_with_online_fallback(
            follow["question_text"], f_answer, tks, client, use_online_judge=use_online_eval
        )
        f_state_update = apply_judge_result(eval_state, turn_id=f_turn_id, judge_result=f_judge, path_id=follow.get("target_path_id"))
        trajectory["turns"].append(
            {
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
        )

    report = build_report(eval_state, trajectory)
    TRAJ_PATH.parent.mkdir(parents=True, exist_ok=True)
    TRAJ_PATH.write_text(json.dumps(trajectory, ensure_ascii=False, indent=2), encoding="utf-8")
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    STATE_PATH.write_text(json.dumps(eval_state, ensure_ascii=False, indent=2), encoding="utf-8")
    FINAL_MMD_PATH.write_text(export_final_state_mermaid(graph, eval_state), encoding="utf-8")
    print(f"Trajectory written: {TRAJ_PATH}")
    print(f"Report written: {REPORT_PATH}")
    print(f"Eval state written: {STATE_PATH}")


if __name__ == "__main__":
    main()
