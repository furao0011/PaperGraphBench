from __future__ import annotations

import argparse
import json
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

from src.artifact_layout import PaperArtifactLayout
from src.config import load_dotenv, load_settings
from src.eval_turn_runner import (
    _question_context,
    _question_metadata,
    _turn_asset_references,
    _turn_multimodal_input,
    _turn_requires_multimodal_input,
    asset_context_for_prompt,
    build_eval_prompt,
    build_model_answer,
    related_forbidden_claims,
)
from src.evaluation_inputs import load_kc_bank, repair_questions_for_graph
from src.judge import judge_answer_with_online_fallback
from src.judge_result_normalizer import normalize_judge_result
from src.model_client import ModelConfig, OpenAICompatClient
from src.multimodal_question_assets import question_image_paths
from src.paper_context import load_full_paper_text


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
DEFAULT_RESULT_ROOT = PROJECT_ROOT / "eval_result_withoutTurn"
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


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    args = _parse_args()
    models = _resolve_models(args.models)
    paper_ids = _resolve_paper_ids(args.paper_ids)
    result_root = Path(args.result_root or os.getenv("WITHOUT_TURN_RESULT_ROOT") or DEFAULT_RESULT_ROOT)

    if not models:
        raise RuntimeError("No withoutTurn evaluation models configured.")
    if not paper_ids:
        raise RuntimeError("No evaluable papers found under papergraph_demo/data.")

    print(
        "[without-turn] start | papers={papers} models={models} result_root={root} paper_workers={workers}".format(
            papers=len(paper_ids),
            models=len(models),
            root=result_root,
            workers=args.paper_workers,
        ),
        flush=True,
    )
    for model in models:
        _run_model_batch(
            model=model,
            paper_ids=paper_ids,
            result_root=result_root,
            paper_workers=args.paper_workers,
            force=args.force,
            dry_run=args.dry_run,
            include_thread_challenges=args.include_thread_challenges,
            continue_on_failure=args.continue_on_failure,
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run independent single-turn QA+judge ablation for fixed macro/challenge questions."
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
        help="Output root. Defaults to WITHOUT_TURN_RESULT_ROOT or ./eval_result_withoutTurn.",
    )
    parser.add_argument(
        "--paper-workers",
        type=int,
        default=2,
        help="Concurrent answer+judge workers per paper. With four papers, default total is eight workers.",
    )
    parser.add_argument(
        "--include-thread-challenges",
        action="store_true",
        help="Also include thread_challenge_questions. Default uses macro_main_questions and challenge_questions only.",
    )
    parser.add_argument("--force", action="store_true", help="Ignore existing final result and checkpoint.")
    parser.add_argument("--dry-run", action="store_true", help="Print scheduled runs without calling models.")
    parser.add_argument(
        "--continue-on-failure",
        action="store_true",
        help="Continue remaining paper/model runs after a failed paper/model pair.",
    )
    return parser.parse_args()


def _run_model_batch(
    model: str,
    paper_ids: list[str],
    result_root: Path,
    paper_workers: int,
    force: bool,
    dry_run: bool,
    include_thread_challenges: bool,
    continue_on_failure: bool,
) -> None:
    if paper_workers <= 0:
        raise ValueError("--paper-workers must be positive.")
    print(f"[without-turn] model started | model={model} papers={len(paper_ids)}", flush=True)
    with ThreadPoolExecutor(max_workers=len(paper_ids)) as executor:
        futures = {
            executor.submit(
                _run_paper_model,
                paper_id,
                model,
                result_root,
                paper_workers,
                force,
                dry_run,
                include_thread_challenges,
            ): paper_id
            for paper_id in paper_ids
        }
        for future in as_completed(futures):
            paper_id = futures[future]
            try:
                future.result()
            except Exception as exc:
                print(
                    f"[without-turn] failed | model={model} paper_id={paper_id} error={type(exc).__name__}: {exc}",
                    flush=True,
                )
                if not continue_on_failure:
                    raise
    print(f"[without-turn] model finished | model={model}", flush=True)


def _run_paper_model(
    paper_id: str,
    model: str,
    result_root: Path,
    paper_workers: int,
    force: bool,
    dry_run: bool,
    include_thread_challenges: bool,
) -> None:
    out_dir = _public_result_dir(result_root, model, paper_id)
    trajectory_path = out_dir / "dialogue_trajectory.json"
    report_path = out_dir / "evaluation_report.json"
    checkpoint_path = out_dir / "without_turn_checkpoint.json"
    if not force and trajectory_path.exists() and report_path.exists():
        print(f"[without-turn] skip existing | model={model} paper_id={paper_id}", flush=True)
        return

    context = _load_paper_context(paper_id, model, include_thread_challenges)
    questions = context["questions"]
    completed, turns, errors = _load_checkpoint(checkpoint_path, paper_id, model, force)
    pending_questions = [question for question in questions if question["question_id"] not in completed]

    if dry_run:
        print(
            f"[without-turn] would run | model={model} paper_id={paper_id} total={len(questions)} pending={len(pending_questions)}",
            flush=True,
        )
        return
    if not pending_questions and turns:
        trajectory = _build_trajectory(paper_id, model, turns)
        report = _build_report(paper_id, model, questions, turns, errors)
        _write_json(trajectory_path, trajectory)
        _write_json(report_path, report)
        checkpoint_path.unlink(missing_ok=True)
        print(f"[without-turn] finalized from checkpoint | model={model} paper_id={paper_id}", flush=True)
        return

    print(
        f"[without-turn] paper started | model={model} paper_id={paper_id} total={len(questions)} pending={len(pending_questions)}",
        flush=True,
    )
    lock = threading.Lock()
    first_error: Exception | None = None
    with ThreadPoolExecutor(max_workers=paper_workers) as executor:
        future_to_question = {
            executor.submit(_run_single_question, context, question): question
            for question in pending_questions
        }
        for future in as_completed(future_to_question):
            question = future_to_question[future]
            try:
                turn = future.result()
            except Exception as exc:
                error = {
                    "question_id": question.get("question_id"),
                    "question_type": question.get("question_type"),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "updated_at": _now_iso(),
                }
                with lock:
                    errors.append(error)
                    _write_checkpoint(checkpoint_path, paper_id, model, completed, turns, errors)
                if first_error is None:
                    first_error = exc
                continue
            with lock:
                completed.add(turn["question_id"])
                turns.append(turn)
                turns.sort(key=lambda item: item.get("question_order", 0))
                _write_checkpoint(checkpoint_path, paper_id, model, completed, turns, errors)
                print(
                    "[without-turn] question completed | model={model} paper_id={paper_id} question_id={qid} done={done}/{total}".format(
                        model=model,
                        paper_id=paper_id,
                        qid=turn["question_id"],
                        done=len(completed),
                        total=len(questions),
                    ),
                    flush=True,
                )

    if first_error is not None:
        raise RuntimeError(
            f"withoutTurn evaluation failed for model={model} paper_id={paper_id}; "
            f"completed={len(completed)}/{len(questions)} errors={len(errors)}. "
            f"Checkpoint: {checkpoint_path}"
        ) from first_error

    trajectory = _build_trajectory(paper_id, model, turns)
    report = _build_report(paper_id, model, questions, turns, errors)
    _write_json(trajectory_path, trajectory)
    _write_json(report_path, report)
    checkpoint_path.unlink(missing_ok=True)
    print(f"[without-turn] paper finished | model={model} paper_id={paper_id}", flush=True)


def _load_paper_context(paper_id: str, model: str, include_thread_challenges: bool) -> dict[str, Any]:
    settings = load_settings(PROJECT_ROOT)
    layout = PaperArtifactLayout(BASE_DIR, paper_id)
    graph = json.loads(layout.final("master_graph").read_text(encoding="utf-8"))
    raw_questions = json.loads(layout.final("question_templates").read_text(encoding="utf-8"))
    questions_payload = repair_questions_for_graph(graph, raw_questions)
    questions = _select_fixed_questions(questions_payload, include_thread_challenges)
    by_kc = {kc["kc_id"]: kc for kc in graph.get("kc_nodes", []) if kc.get("kc_id")}
    kc_bank = load_kc_bank(graph, BASE_DIR)
    paper_text = load_full_paper_text(graph, BASE_DIR)

    judge_client = OpenAICompatClient(ModelConfig(settings.api_key, settings.base_url, settings.llm_model))
    if not judge_client.is_ready():
        raise RuntimeError("withoutTurn evaluation requires configured judge API_KEY/BASE_URL/LLM_MODEL.")
    target_client = _build_target_client(model)
    if not target_client.is_ready() and not _env_bool("ALLOW_MOCK_EVAL", False):
        raise RuntimeError(
            f"withoutTurn evaluation requires configured target model API for {model}: "
            "EVAL_TARGET_API_KEY/EVAL_TARGET_BASE_URL or model-specific overrides."
        )

    return {
        "paper_id": paper_id,
        "model": model,
        "graph": graph,
        "questions": questions,
        "by_kc": by_kc,
        "kc_bank": kc_bank,
        "paper_text": paper_text,
        "judge_client": judge_client,
        "target_client": target_client,
    }


def _select_fixed_questions(questions_payload: dict, include_thread_challenges: bool) -> list[dict]:
    selected: list[dict] = []
    for key in ("macro_main_questions", "challenge_questions"):
        for question in questions_payload.get(key, []):
            selected.append(dict(question))
    if include_thread_challenges:
        for question in questions_payload.get("thread_challenge_questions", []):
            selected.append(dict(question))

    out = []
    seen = set()
    for idx, question in enumerate(selected, start=1):
        question_id = str(question.get("question_id") or "").strip()
        if not question_id:
            raise ValueError(f"Question #{idx} has no question_id.")
        if question_id in seen:
            raise ValueError(f"Duplicate fixed question_id={question_id}.")
        if not question.get("target_kc_ids"):
            raise ValueError(f"Question {question_id} has no target_kc_ids.")
        question["question_order"] = idx
        seen.add(question_id)
        out.append(question)
    return out


def _run_single_question(context: dict[str, Any], question: dict) -> dict:
    by_kc = context["by_kc"]
    target_kcs = [by_kc[kc_id] for kc_id in question.get("target_kc_ids", []) if kc_id in by_kc]
    if len(target_kcs) != len(question.get("target_kc_ids", [])):
        missing = [kc_id for kc_id in question.get("target_kc_ids", []) if kc_id not in by_kc]
        raise ValueError(f"Question {question.get('question_id')} references missing target KCs: {missing}")

    turn_id = f"WT{int(question['question_order']):04d}"
    prompt = build_eval_prompt(
        paper_text=context["paper_text"],
        dialogue_history="No previous turns.",
        question_text=question["question_text"],
        asset_context=asset_context_for_prompt(question),
    )
    answer, answer_mode = build_model_answer(
        client=context["target_client"],
        use_online_eval=True,
        prompt=prompt,
        target_kcs=target_kcs,
        image_paths=question_image_paths(question),
    )
    judge_result = judge_answer_with_online_fallback(
        question["question_text"],
        answer,
        target_kcs,
        context["judge_client"],
        use_online_judge=True,
        dialogue_summary="",
        related_forbidden_claims=related_forbidden_claims(
            context["graph"],
            question.get("target_kc_ids", []),
            question.get("path_id") or question.get("target_path_id"),
        ),
        question_type=question["question_type"],
        thread_context=_question_context(question, None),
    )
    judge_result = normalize_judge_result(judge_result, _turn_context(turn_id, question))
    return {
        "turn_id": turn_id,
        "question_order": question["question_order"],
        "question_id": question["question_id"],
        "question_type": question["question_type"],
        "macro_id": question.get("macro_id"),
        "thread_id": question.get("thread_id") or question.get("target_thread_id"),
        "thread_turn_id": question.get("thread_turn_id") or question.get("target_thread_turn_id"),
        "thread_role": question.get("thread_role"),
        "challenge_type": question.get("challenge_type"),
        "challenge_scope": question.get("challenge_scope"),
        "challenge_trigger": question.get("challenge_trigger"),
        "target_failure_mode": question.get("target_failure_mode"),
        "expected_behavior": question.get("expected_behavior"),
        "question_text": question["question_text"],
        "target_kc_ids": question["target_kc_ids"],
        "target_path_id": question.get("path_id") or question.get("target_path_id"),
        "question_metadata": _question_metadata(question),
        "model_answer": answer,
        "answer_mode": answer_mode,
        "requires_multimodal_input": _turn_requires_multimodal_input(question),
        "asset_references": _turn_asset_references(question),
        "multimodal_input": _turn_multimodal_input(question),
        "judge_result": judge_result,
        "without_turn": {
            "independent_question": True,
            "dialogue_history": "No previous turns.",
            "repair_tasks_executed": False,
            "followups_executed": False,
        },
    }


def _turn_context(turn_id: str, question: dict) -> dict:
    return {
        "turn_id": turn_id,
        "question_id": question["question_id"],
        "question_type": question["question_type"],
        "macro_id": question.get("macro_id"),
        "thread_id": question.get("thread_id") or question.get("target_thread_id"),
        "thread_turn_id": question.get("thread_turn_id") or question.get("target_thread_turn_id"),
        "thread_role": question.get("thread_role"),
        "target_kc_ids": question.get("target_kc_ids", []),
        "target_path_id": question.get("path_id") or question.get("target_path_id"),
        "challenge_type": question.get("challenge_type"),
        "challenge_trigger": question.get("challenge_trigger"),
        "target_failure_mode": question.get("target_failure_mode"),
        "expected_behavior": question.get("expected_behavior"),
        "modality_pool": question.get("modality_pool"),
        "requires_multimodal_input": _turn_requires_multimodal_input(question),
        "asset_references": _turn_asset_references(question),
        "multimodal_input": _turn_multimodal_input(question),
    }


def _build_target_client(model: str) -> OpenAICompatClient:
    suffix = _model_env_suffix(model)
    api_key = os.getenv(f"EVAL_TARGET_API_KEY_{suffix}") or os.getenv("EVAL_TARGET_API_KEY", "")
    base_url = os.getenv(f"EVAL_TARGET_BASE_URL_{suffix}") or os.getenv("EVAL_TARGET_BASE_URL", "")
    return OpenAICompatClient(ModelConfig(api_key=api_key, base_url=base_url, llm_model=model))


def _build_trajectory(paper_id: str, model: str, turns: list[dict]) -> dict:
    return {
        "paper_id": paper_id,
        "target_model": model,
        "evaluation_mode": "without_turn_single_qa",
        "turns": sorted(turns, key=lambda item: item.get("question_order", 0)),
    }


def _build_report(paper_id: str, model: str, questions: list[dict], turns: list[dict], errors: list[dict]) -> dict:
    state_counts: dict[str, int] = {}
    type_counts: dict[str, int] = {}
    challenge_counts: dict[str, int] = {}
    multimodal_count = 0
    covered_total = 0
    missing_total = 0
    for turn in turns:
        question_type = str(turn.get("question_type") or "unknown")
        type_counts[question_type] = type_counts.get(question_type, 0) + 1
        state = str(turn.get("judge_result", {}).get("state") or "unknown")
        state_counts[state] = state_counts.get(state, 0) + 1
        covered_total += len(turn.get("judge_result", {}).get("covered_kc_ids", []) or [])
        missing_total += len(turn.get("judge_result", {}).get("missing_kc_ids", []) or [])
        if turn.get("requires_multimodal_input"):
            multimodal_count += 1
        challenge_result = turn.get("judge_result", {}).get("challenge_result") or {}
        if challenge_result:
            if challenge_result.get("failed"):
                key = "failed"
            elif challenge_result.get("resisted"):
                key = "resisted"
            elif challenge_result.get("incomplete"):
                key = "incomplete"
            else:
                key = "other"
            challenge_counts[key] = challenge_counts.get(key, 0) + 1

    total = len(questions)
    completed = len(turns)
    macro_metrics = _macro_without_turn_metrics(turns)
    challenge_metrics = _challenge_without_turn_metrics(turns)
    return {
        "paper_id": paper_id,
        "target_model": model,
        "evaluation_mode": "without_turn_single_qa",
        "status": "completed" if completed == total and not errors else "failed",
        "summary": {
            "total_questions": total,
            "completed_questions": completed,
            "failed_questions": len(errors),
            "multimodal_questions": multimodal_count,
            "question_type_counts": type_counts,
            "judge_state_counts": state_counts,
            "challenge_result_counts": challenge_counts,
            "covered_kc_mentions": covered_total,
            "missing_kc_mentions": missing_total,
            "macro_without_turn_metrics": macro_metrics,
            "challenge_without_turn_metrics": challenge_metrics,
        },
        "errors": errors,
        "updated_at": _now_iso(),
    }


def _macro_without_turn_metrics(turns: list[dict]) -> dict:
    macro_turns = [
        turn
        for turn in turns
        if turn.get("question_type") in {"macro_main_question", "main"}
    ]
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


def _challenge_without_turn_metrics(turns: list[dict]) -> dict:
    challenge_turns = [
        turn
        for turn in turns
        if turn.get("question_type") in {"challenge_question", "thread_challenge_question"}
    ]
    total = len(challenge_turns)
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


def _load_checkpoint(
    checkpoint_path: Path,
    paper_id: str,
    model: str,
    force: bool,
) -> tuple[set[str], list[dict], list[dict]]:
    if force or not checkpoint_path.exists():
        return set(), [], []
    data = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    if data.get("paper_id") != paper_id or data.get("target_model") != model:
        raise ValueError(f"Checkpoint identity mismatch: {checkpoint_path}")
    turns = [turn for turn in data.get("turns", []) if isinstance(turn, dict)]
    completed = {str(qid) for qid in data.get("completed_question_ids", []) if qid}
    if not completed:
        completed = {str(turn.get("question_id")) for turn in turns if turn.get("question_id")}
    errors = [item for item in data.get("errors", []) if isinstance(item, dict)]
    return completed, turns, errors


def _write_checkpoint(
    checkpoint_path: Path,
    paper_id: str,
    model: str,
    completed: set[str],
    turns: list[dict],
    errors: list[dict],
) -> None:
    _write_json(
        checkpoint_path,
        {
            "paper_id": paper_id,
            "target_model": model,
            "evaluation_mode": "without_turn_single_qa",
            "completed_question_ids": sorted(completed),
            "turns": sorted(turns, key=lambda item: item.get("question_order", 0)),
            "errors": errors,
            "updated_at": _now_iso(),
        },
    )


def _resolve_models(raw_models: list[str] | None) -> list[str]:
    chunks = raw_models if raw_models else [os.getenv("WITHOUT_TURN_MODELS", "")]
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


def _safe_dir_name(value: object) -> str:
    safe = re.sub(r"[^\w._-]+", "_", str(value or "").strip())
    return safe.strip("._-") or "unknown"


def _model_env_suffix(model: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9]+", "_", model.strip()).strip("_")
    return safe.upper()


def _write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


if __name__ == "__main__":
    main()
