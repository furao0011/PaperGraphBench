from __future__ import annotations

import argparse
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

from src.artifact_layout import PaperArtifactLayout
from src.challenge_scheduler import ensure_challenge_states
from src.config import load_dotenv, load_settings
from src.eval_artifacts import load_eval_checkpoint, reconcile_actual_transitions, write_json
from src.eval_turn_runner import EvaluationTurnRunner
from src.evaluation_inputs import load_kc_bank, repair_questions_for_graph
from src.evaluation_state import (
    ensure_eval_state_defaults,
    final_evaluation_status,
    mark_evaluation_finished,
    mark_evaluation_running,
    rebuild_eval_turn_counts,
)
from src.mermaid_exporter import export_final_state_mermaid, export_final_thread_state_mermaid
from src.model_client import ModelConfig, OpenAICompatClient
from src.no_repair_stage_runner import NoRepairEvaluationStageRunner
from src.paper_context import load_full_paper_text
from src.progress import log, span
from src.reporter import build_report
from src.state_updater import initialize_eval_state
from src.thread_scheduler import completed_thread_step_ids


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
DEFAULT_RESULT_ROOT = PROJECT_ROOT / "eval_result_withoutRepair"
DEFAULT_MODELS = [
    "gpt-5-mini",
    "gpt-5",
    "ark-doubao-seed-2.0-pro-260215",
    "ark-doubao-seed-2.0-mini-260215",
]
MODEL_ALIASES = {
    "gpt5mini": "gpt-5-mini",
    "gpt5": "gpt-5",
    "doubaopro": "ark-doubao-seed-2.0-pro-260215",
    "doubaomini": "ark-doubao-seed-2.0-mini-260215",
}
EVALUATION_MODE = "with_context_without_repair"


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    args = _parse_args()
    models = _resolve_models(args.models)
    paper_ids = _resolve_paper_ids(args.paper_ids)
    result_root = Path(args.result_root or os.getenv("WITHOUT_REPAIR_RESULT_ROOT") or DEFAULT_RESULT_ROOT)

    if not models:
        raise RuntimeError("No withoutRepair evaluation models configured.")
    if not paper_ids:
        raise RuntimeError("No evaluable papers found under papergraph_demo/data.")
    if args.paper_batch_size <= 0:
        raise ValueError("--paper-batch-size must be positive.")
    if args.model_workers_per_paper <= 0:
        raise ValueError("--model-workers-per-paper must be positive.")

    print(
        "[without-repair] start | papers={papers} models={models} result_root={root} "
        "paper_batch_size={paper_batch_size} model_workers_per_paper={model_workers}".format(
            papers=len(paper_ids),
            models=len(models),
            root=result_root,
            paper_batch_size=args.paper_batch_size,
            model_workers=args.model_workers_per_paper,
        ),
        flush=True,
    )
    failures: list[dict] = []
    for batch_no, paper_batch in enumerate(_chunks(paper_ids, args.paper_batch_size), start=1):
        failures.extend(
            _run_paper_batch(
                batch_no=batch_no,
                paper_ids=paper_batch,
                models=models,
                result_root=result_root,
                max_workers=args.paper_batch_size * args.model_workers_per_paper,
                force=args.force,
                dry_run=args.dry_run,
            )
        )

    if failures:
        message = "; ".join(
            f"{item['model']}/{item['paper_id']}: {item['error_type']}: {item['error']}"
            for item in failures[:8]
        )
        raise RuntimeError(f"withoutRepair batch finished with {len(failures)} failed runs: {message}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run contextual full-dialogue evaluation while suppressing repair follow-ups."
    )
    parser.add_argument(
        "--models",
        nargs="*",
        help="Model list. Supports aliases: gpt5mini, gpt5, doubaopro, doubaomini.",
    )
    parser.add_argument(
        "--paper-ids",
        nargs="*",
        help="Optional paper_id subset. Defaults to every evaluable directory under papergraph_demo/data.",
    )
    parser.add_argument(
        "--result-root",
        default="",
        help="Output root. Defaults to WITHOUT_REPAIR_RESULT_ROOT or ./eval_result_withoutRepair.",
    )
    parser.add_argument(
        "--paper-batch-size",
        type=int,
        default=int(os.getenv("WITHOUT_REPAIR_PAPER_BATCH_SIZE", "2") or "2"),
        help="How many papers run at the same time. Default: 2.",
    )
    parser.add_argument(
        "--model-workers-per-paper",
        type=int,
        default=int(os.getenv("WITHOUT_REPAIR_MODEL_WORKERS_PER_PAPER", "4") or "4"),
        help="How many model/paper evaluations can run concurrently per paper batch slot. Default: 4.",
    )
    parser.add_argument("--force", action="store_true", help="Ignore existing final result and checkpoint.")
    parser.add_argument("--dry-run", action="store_true", help="Print scheduled runs without calling models.")
    return parser.parse_args()


def _run_paper_batch(
    batch_no: int,
    paper_ids: list[str],
    models: list[str],
    result_root: Path,
    max_workers: int,
    force: bool,
    dry_run: bool,
) -> list[dict]:
    jobs = [(paper_id, model) for paper_id in paper_ids for model in models]
    print(
        f"[without-repair] paper batch started | batch={batch_no} papers={len(paper_ids)} jobs={len(jobs)} workers={max_workers}",
        flush=True,
    )
    failures: list[dict] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_job = {
            executor.submit(_run_paper_model, paper_id, model, result_root, force, dry_run): (paper_id, model)
            for paper_id, model in jobs
        }
        for future in as_completed(future_to_job):
            paper_id, model = future_to_job[future]
            try:
                future.result()
            except Exception as exc:
                failure = {
                    "paper_id": paper_id,
                    "model": model,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "updated_at": _now_iso(),
                }
                failures.append(failure)
                print(
                    f"[without-repair] failed | model={model} paper_id={paper_id} error={type(exc).__name__}: {exc}",
                    flush=True,
                )
    print(
        f"[without-repair] paper batch finished | batch={batch_no} failures={len(failures)}",
        flush=True,
    )
    return failures


def _run_paper_model(
    paper_id: str,
    model: str,
    result_root: Path,
    force: bool,
    dry_run: bool,
) -> None:
    out_dir = _public_result_dir(result_root, model, paper_id)
    paths = _result_paths(out_dir)
    if not force and _completed_result_exists(paths):
        print(f"[without-repair] skip existing | model={model} paper_id={paper_id}", flush=True)
        return
    if dry_run:
        print(f"[without-repair] would run | model={model} paper_id={paper_id} out_dir={out_dir}", flush=True)
        return

    context = _load_context(paper_id, model, out_dir)
    graph = context["graph"]
    checkpoint = None
    if not force:
        checkpoint = load_eval_checkpoint(
            paths["checkpoint"],
            graph,
            model,
            ensure_eval_state_defaults,
            rebuild_eval_turn_counts,
        )
    if checkpoint:
        eval_state, trajectory, turn_no, completed_question_ids, _completed_thread_step_ids = checkpoint
        log(
            "without-repair checkpoint loaded",
            model=model,
            paper_id=paper_id,
            turns=len(trajectory.get("turns", [])),
            completed_questions=len(completed_question_ids),
            checkpoint=paths["checkpoint"],
        )
    else:
        eval_state = initialize_eval_state(graph, target_model=model)
        ensure_eval_state_defaults(eval_state, graph)
        trajectory = {
            "paper_id": graph["paper_id"],
            "target_model": model,
            "evaluation_mode": EVALUATION_MODE,
            "ablation": {
                "full_dialogue_context": True,
                "repair_tasks_executed": False,
                "detail_followups_executed": False,
                "hallucination_followups_executed": False,
            },
            "turns": [],
        }
        turn_no = 0
        completed_question_ids = set()

    max_turns = int(os.getenv("WITHOUT_REPAIR_MAX_TURNS") or os.getenv("EVAL_MAX_TURNS", "0") or "0")
    mark_evaluation_running(eval_state)
    ensure_challenge_states(
        eval_state,
        context["questions"].get("challenge_questions", []) + context["questions"].get("thread_challenge_questions", []),
    )
    runner = EvaluationTurnRunner(
        graph=graph,
        by_kc=context["by_kc"],
        paper_text=context["paper_text"],
        client=context["judge_client"],
        target_client=context["target_client"],
        use_online_eval=True,
        allow_offline_fallback=False,
        kc_bank=context["kc_bank"],
        claim_log_path=paths["claim_log"],
    )

    def save_current_artifacts(state: dict, turns: dict, _turn_no: int) -> None:
        _save_without_repair_artifacts(graph, state, turns, paths["checkpoint"], paths)

    stage_runner = NoRepairEvaluationStageRunner(
        runner=runner,
        graph=graph,
        questions=context["questions"],
        trajectory=trajectory,
        eval_state=eval_state,
        completed_question_ids=completed_question_ids,
        by_kc=context["by_kc"],
        client=context["judge_client"],
        allow_offline_fallback=False,
        save_artifacts=save_current_artifacts,
        max_turns=max_turns,
    )

    macro_queue = context["questions"].get("macro_main_questions") or context["questions"].get("main_questions", [])
    queue = macro_queue + context["questions"].get("multi_hop_questions", [])
    if completed_question_ids:
        queue = [q for q in queue if q.get("question_id") not in completed_question_ids]

    print(
        f"[without-repair] paper/model started | model={model} paper_id={paper_id} questions={len(queue)} max_turns={max_turns}",
        flush=True,
    )
    with span("without-repair evaluation", model=model, paper_id=paper_id):
        for question in queue:
            if eval_state["global_state"]["failed"]:
                break
            if max_turns and turn_no >= max_turns:
                break
            turn_no = stage_runner.run_macro_stage(question, turn_no)
            if eval_state["global_state"]["failed"]:
                break
        turn_no = stage_runner.run_review_stage(turn_no)

    final_status, final_reason = final_evaluation_status(eval_state, turn_no, max_turns)
    mark_evaluation_finished(eval_state, final_status, final_reason, turn_no)
    _save_without_repair_artifacts(graph, eval_state, trajectory, paths["checkpoint"], paths)
    print(
        f"[without-repair] paper/model finished | model={model} paper_id={paper_id} turns={len(trajectory.get('turns', []))}",
        flush=True,
    )


def _load_context(paper_id: str, model: str, out_dir: Path) -> dict[str, Any]:
    settings = load_settings(PROJECT_ROOT)
    layout = PaperArtifactLayout(BASE_DIR, paper_id)
    graph = json.loads(layout.final("master_graph").read_text(encoding="utf-8"))
    raw_questions = json.loads(layout.final("question_templates").read_text(encoding="utf-8"))
    questions = repair_questions_for_graph(graph, raw_questions)
    by_kc = {kc["kc_id"]: kc for kc in graph.get("kc_nodes", []) if kc.get("kc_id")}
    kc_bank = load_kc_bank(graph, BASE_DIR)
    paper_text = load_full_paper_text(graph, BASE_DIR)

    judge_client = OpenAICompatClient(ModelConfig(settings.api_key, settings.base_url, settings.llm_model))
    if not judge_client.is_ready():
        raise RuntimeError("withoutRepair evaluation requires configured judge API_KEY/BASE_URL/LLM_MODEL.")
    target_client = _build_target_client(model)
    if not target_client.is_ready():
        raise RuntimeError(
            f"withoutRepair evaluation requires configured target model API for {model}: "
            "EVAL_TARGET_API_KEY/EVAL_TARGET_BASE_URL or model-specific overrides."
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    return {
        "graph": graph,
        "questions": questions,
        "by_kc": by_kc,
        "kc_bank": kc_bank,
        "paper_text": paper_text,
        "judge_client": judge_client,
        "target_client": target_client,
    }


def _save_without_repair_artifacts(
    graph: dict,
    eval_state: dict,
    trajectory: dict,
    checkpoint_path: Path,
    paths: dict[str, Path],
) -> None:
    trajectory["evaluation_mode"] = EVALUATION_MODE
    trajectory.setdefault("ablation", {})
    trajectory["ablation"].update(
        {
            "full_dialogue_context": True,
            "repair_tasks_executed": False,
            "detail_followups_executed": False,
            "hallucination_followups_executed": False,
        }
    )
    reconcile_actual_transitions(trajectory)
    report = build_report(eval_state, trajectory)
    report["evaluation_mode"] = EVALUATION_MODE
    report["without_repair_metrics"] = _without_repair_metrics(eval_state, trajectory)
    write_json(paths["trajectory"], trajectory)
    write_json(paths["report"], report)
    write_json(paths["state"], eval_state)
    paths["final_mmd"].parent.mkdir(parents=True, exist_ok=True)
    paths["final_mmd"].write_text(export_final_state_mermaid(graph, eval_state), encoding="utf-8")
    paths["final_thread_mmd"].write_text(export_final_thread_state_mermaid(graph, eval_state), encoding="utf-8")
    completed_question_ids = [
        turn.get("question_id")
        for turn in trajectory.get("turns", [])
        if turn.get("question_type") in {"main", "macro_main_question", "multi_hop_reasoning"}
    ]
    completed_thread_steps = sorted(completed_thread_step_ids(eval_state))
    write_json(
        checkpoint_path,
        {
            "paper_id": graph.get("paper_id", "unknown"),
            "target_model": trajectory.get("target_model"),
            "evaluation_mode": EVALUATION_MODE,
            "turn_no": max((_turn_number(t.get("turn_id", "")) for t in trajectory.get("turns", [])), default=0),
            "completed_question_ids": completed_question_ids,
            "completed_thread_step_ids": completed_thread_steps,
            "scheduler_state": {
                "completed_thread_step_ids": completed_thread_steps,
                "last_turn_id": trajectory.get("turns", [{}])[-1].get("turn_id") if trajectory.get("turns") else None,
            },
            "trajectory": trajectory,
            "eval_state": eval_state,
            "updated_at": _now_iso(),
        },
    )


def _without_repair_metrics(eval_state: dict, trajectory: dict) -> dict:
    turns = trajectory.get("turns", [])
    return {
        "macro_contextual_metrics": _macro_once_metrics(turns),
        "challenge_contextual_metrics": _challenge_once_metrics(turns),
        "hallucination_event_metrics": _hallucination_event_metrics(eval_state, turns),
    }


def _macro_once_metrics(turns: list[dict]) -> dict:
    macro_turns = [turn for turn in turns if turn.get("question_type") in {"macro_main_question", "main"}]
    target_mentions = 0
    covered_mentions = 0
    full_success = 0
    per_macro = []
    for turn in macro_turns:
        target_ids = _dedupe_strings(turn.get("target_kc_ids", []))
        covered_ids = set(_dedupe_strings(turn.get("judge_result", {}).get("covered_kc_ids", [])))
        covered_target_ids = [kc_id for kc_id in target_ids if kc_id in covered_ids]
        target_mentions += len(target_ids)
        covered_mentions += len(covered_target_ids)
        fully_covered = bool(target_ids) and len(covered_target_ids) == len(target_ids)
        if fully_covered:
            full_success += 1
        per_macro.append(
            {
                "question_id": turn.get("question_id"),
                "macro_id": turn.get("macro_id"),
                "target_kc_count": len(target_ids),
                "covered_target_kc_count": len(covered_target_ids),
                "fully_covered_once": fully_covered,
                "coverage_ratio": _safe_ratio(len(covered_target_ids), len(target_ids)),
            }
        )
    return {
        "macro_question_count": len(macro_turns),
        "fully_covered_once_count": full_success,
        "fully_covered_once_rate": _safe_ratio(full_success, len(macro_turns)),
        "target_kc_mentions": target_mentions,
        "covered_target_kc_mentions": covered_mentions,
        "overall_target_kc_coverage_ratio": _safe_ratio(covered_mentions, target_mentions),
        "per_macro": per_macro,
    }


def _challenge_once_metrics(turns: list[dict]) -> dict:
    challenge_turns = [
        turn
        for turn in turns
        if turn.get("question_type") in {"challenge_question", "thread_challenge_question"}
    ]
    failed = 0
    resisted = 0
    incomplete = 0
    text_total = 0
    text_failed = 0
    multimodal_total = 0
    multimodal_failed = 0
    failed_by_challenge_type: dict[str, int] = {}
    failed_by_failure_mode: dict[str, int] = {}
    per_question = []
    for turn in challenge_turns:
        challenge_result = turn.get("judge_result", {}).get("challenge_result") or {}
        is_failed = bool(challenge_result.get("failed"))
        is_resisted = bool(challenge_result.get("resisted"))
        is_incomplete = bool(challenge_result.get("incomplete"))
        is_multimodal = bool(turn.get("requires_multimodal_input"))
        if is_multimodal:
            multimodal_total += 1
        else:
            text_total += 1
        if is_failed:
            failed += 1
            if is_multimodal:
                multimodal_failed += 1
            else:
                text_failed += 1
            challenge_type = str(turn.get("challenge_type") or "unknown")
            failure_mode = str(turn.get("target_failure_mode") or "unknown")
            failed_by_challenge_type[challenge_type] = failed_by_challenge_type.get(challenge_type, 0) + 1
            failed_by_failure_mode[failure_mode] = failed_by_failure_mode.get(failure_mode, 0) + 1
        if is_resisted:
            resisted += 1
        if is_incomplete:
            incomplete += 1
        per_question.append(
            {
                "question_id": turn.get("question_id"),
                "question_type": turn.get("question_type"),
                "challenge_type": turn.get("challenge_type"),
                "target_failure_mode": turn.get("target_failure_mode"),
                "requires_multimodal_input": is_multimodal,
                "state": challenge_result.get("state") or turn.get("judge_result", {}).get("state"),
                "failed": is_failed,
                "resisted": is_resisted,
                "incomplete": is_incomplete,
            }
        )
    total = len(challenge_turns)
    return {
        "challenge_question_count": total,
        "failed_count": failed,
        "failed_rate": _safe_ratio(failed, total),
        "resisted_count": resisted,
        "resisted_rate": _safe_ratio(resisted, total),
        "incomplete_count": incomplete,
        "incomplete_rate": _safe_ratio(incomplete, total),
        "text_challenge_count": text_total,
        "text_failed_count": text_failed,
        "text_failed_rate": _safe_ratio(text_failed, text_total),
        "multimodal_challenge_count": multimodal_total,
        "multimodal_failed_count": multimodal_failed,
        "multimodal_failed_rate": _safe_ratio(multimodal_failed, multimodal_total),
        "failed_by_challenge_type": dict(sorted(failed_by_challenge_type.items())),
        "failed_by_failure_mode": dict(sorted(failed_by_failure_mode.items())),
        "per_question": per_question,
    }


def _hallucination_event_metrics(eval_state: dict, turns: list[dict]) -> dict:
    events = list(eval_state.get("hallucination_events", {}).values())
    by_type: dict[str, int] = {}
    by_status: dict[str, int] = {}
    for event in events:
        htype = str(event.get("hallucination_type") or "unknown")
        status = str(event.get("status") or "unknown")
        by_type[htype] = by_type.get(htype, 0) + 1
        by_status[status] = by_status.get(status, 0) + 1
    turn_event_count = sum(
        len(turn.get("judge_result", {}).get("hallucination_events", []) or [])
        for turn in turns
    )
    return {
        "hallucination_event_count": len(events),
        "hallucination_event_mentions_in_turns": turn_event_count,
        "hallucination_event_rate_per_turn": _safe_ratio(len(events), len(turns)),
        "hallucination_by_type": dict(sorted(by_type.items())),
        "hallucination_by_status": dict(sorted(by_status.items())),
    }


def _result_paths(out_dir: Path) -> dict[str, Path]:
    cache_dir = out_dir / "cache"
    return {
        "trajectory": out_dir / "dialogue_trajectory.json",
        "report": out_dir / "evaluation_report.json",
        "state": out_dir / "eval_state_graph.json",
        "checkpoint": cache_dir / "evaluation_checkpoint.json",
        "claim_log": cache_dir / "claim_verification_log.json",
        "final_mmd": cache_dir / "final_state_graph.mmd",
        "final_thread_mmd": cache_dir / "final_thread_state_graph.mmd",
    }


def _completed_result_exists(paths: dict[str, Path]) -> bool:
    trajectory_path = paths["trajectory"]
    report_path = paths["report"]
    if not trajectory_path.exists() or not report_path.exists():
        return False
    trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if trajectory.get("evaluation_mode") != EVALUATION_MODE:
        return False
    if report.get("evaluation_mode") != EVALUATION_MODE:
        return False
    status = (report.get("summary") or {}).get("evaluation_status")
    return status == "completed"


def _build_target_client(model: str) -> OpenAICompatClient:
    suffix = _model_env_suffix(model)
    api_key = os.getenv(f"EVAL_TARGET_API_KEY_{suffix}") or os.getenv("EVAL_TARGET_API_KEY", "")
    base_url = os.getenv(f"EVAL_TARGET_BASE_URL_{suffix}") or os.getenv("EVAL_TARGET_BASE_URL", "")
    return OpenAICompatClient(ModelConfig(api_key=api_key, base_url=base_url, llm_model=model))


def _resolve_models(raw_models: list[str] | None) -> list[str]:
    chunks = raw_models if raw_models else [os.getenv("WITHOUT_REPAIR_MODELS", "")]
    models = [_normalize_model_name(item) for item in _split_values(chunks)]
    return models or DEFAULT_MODELS


def _resolve_paper_ids(raw_paper_ids: list[str] | None) -> list[str]:
    requested = _split_values(raw_paper_ids or [])
    if requested:
        return [_assert_evaluable_paper(paper_id) for paper_id in requested]
    paper_ids = []
    data_root = BASE_DIR / "data"
    if not data_root.exists():
        return paper_ids
    for path in sorted(data_root.iterdir(), key=lambda p: p.name):
        if not path.is_dir():
            continue
        try:
            paper_id = _assert_evaluable_paper(path.name)
        except FileNotFoundError:
            continue
        paper_ids.append(paper_id)
    return paper_ids


def _assert_evaluable_paper(paper_id: str) -> str:
    layout = PaperArtifactLayout(BASE_DIR, paper_id)
    required = [layout.final("master_graph"), layout.final("question_templates")]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Paper {paper_id!r} is not evaluable; missing: {', '.join(missing)}")
    return layout.paper_id


def _public_result_dir(result_root: Path, model: str, paper_id: str) -> Path:
    return result_root / _safe_dir_name(model) / _safe_dir_name(paper_id)


def _chunks(values: list[str], size: int) -> list[list[str]]:
    return [values[idx : idx + size] for idx in range(0, len(values), size)]


def _split_values(values: list[str]) -> list[str]:
    items = []
    seen = set()
    for value in values:
        for item in re.split(r"[\n,]+", value):
            item = item.strip()
            if not item or item in seen:
                continue
            items.append(item)
            seen.add(item)
    return items


def _normalize_model_name(model: str) -> str:
    compact = re.sub(r"[^A-Za-z0-9]+", "", model).lower()
    return MODEL_ALIASES.get(compact, model)


def _dedupe_strings(values: list) -> list[str]:
    out = []
    seen = set()
    for value in values:
        item = str(value).strip()
        if not item or item in seen:
            continue
        out.append(item)
        seen.add(item)
    return out


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return numerator / denominator


def _safe_dir_name(value: object) -> str:
    safe = re.sub(r"[^\w._-]+", "_", str(value or "").strip())
    return safe.strip("._-") or "unknown"


def _model_env_suffix(model: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9]+", "_", model.strip()).strip("_")
    return safe.upper()


def _turn_number(turn_id: str) -> int:
    match = re.search(r"(\d+)$", str(turn_id))
    return int(match.group(1)) if match else 0


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


if __name__ == "__main__":
    main()
