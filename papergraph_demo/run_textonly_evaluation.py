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
from src.batch_progress import BatchTask, PaperBatchProgress
from src.config import load_dotenv, load_settings
from src.model_client import ModelConfig, OpenAICompatClient
from src.textonly_benchmark import (
    TEXTONLY_EVAL_MODE,
    build_textonly_report,
    completed_textonly_result_exists,
    load_paper_clean_text,
    run_textonly_question,
    textonly_no_repair_questions,
)


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
DEFAULT_RESULT_ROOT = PROJECT_ROOT / "eval_result_textOnly"
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
    result_root = Path(args.result_root or os.getenv("TEXTONLY_EVAL_RESULT_ROOT") or DEFAULT_RESULT_ROOT)
    if not models:
        raise RuntimeError("No text-only evaluation models configured.")
    if not paper_ids:
        raise RuntimeError("No textonly_question_templates.json files found under data/.")
    if args.workers <= 0:
        raise ValueError("--workers must be positive.")

    tasks = [BatchTask(paper_id=paper_id, total=len(models)) for paper_id in paper_ids]
    failures: list[dict] = []
    with PaperBatchProgress("textonly-eval", tasks) as progress:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_paper = {
                executor.submit(
                    _run_paper_models,
                    progress,
                    paper_id,
                    models,
                    result_root,
                    args.force,
                    args.dry_run,
                    args.continue_on_failure,
                ): paper_id
                for paper_id in paper_ids
            }
            for future in as_completed(future_to_paper):
                paper_id = future_to_paper[future]
                try:
                    failures.extend(future.result())
                except Exception as exc:
                    failures.append(
                        {
                            "paper_id": paper_id,
                            "model": None,
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                            "updated_at": _now_iso(),
                        }
                    )
                    if not args.continue_on_failure:
                        for pending in future_to_paper:
                            pending.cancel()
                        raise
    if failures:
        raise RuntimeError(f"Text-only evaluation finished with {len(failures)} failed model runs.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate no-graph text-only benchmark question packages.")
    parser.add_argument("--models", nargs="*", help="Target model list. Supports aliases: gpt5mini, gpt5, doubaopro, doubaomini.")
    parser.add_argument("--paper-ids", nargs="*", help="Optional paper_id subset. Defaults to every paper with textonly_question_templates.json.")
    parser.add_argument("--result-root", default="", help="Defaults to TEXTONLY_EVAL_RESULT_ROOT or ./eval_result_textOnly.")
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.getenv("TEXTONLY_EVAL_WORKERS", "4") or "4"),
        help="Maximum concurrent papers. The configured models run sequentially inside each paper worker.",
    )
    parser.add_argument("--force", action="store_true", help="Ignore existing final result and checkpoint.")
    parser.add_argument("--dry-run", action="store_true", help="Print scheduled runs without calling models.")
    parser.add_argument("--continue-on-failure", action="store_true", help="Continue remaining models and papers, then fail after the full batch.")
    return parser.parse_args()


def _run_paper_models(
    progress: PaperBatchProgress,
    paper_id: str,
    models: list[str],
    result_root: Path,
    force: bool,
    dry_run: bool,
    continue_on_failure: bool,
) -> list[dict]:
    failures = []
    progress.start(paper_id, f"running 0/{len(models)} models")
    for model_index, model in enumerate(models, start=1):
        progress.update(
            paper_id,
            status=f"{model}: starting ({model_index}/{len(models)})",
        )
        try:
            outcome = _run_paper_model(
                paper_id=paper_id,
                model=model,
                result_root=result_root,
                force=force,
                dry_run=dry_run,
                progress=progress,
                model_index=model_index,
                model_total=len(models),
            )
        except Exception as exc:
            failures.append(
                {
                    "paper_id": paper_id,
                    "model": model,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "updated_at": _now_iso(),
                }
            )
            progress.update(
                paper_id,
                status=f"{model}: failed ({model_index}/{len(models)})",
                advance=1,
                emit=True,
            )
            if not continue_on_failure:
                progress.fail(paper_id, exc)
                raise
            continue
        progress.update(
            paper_id,
            status=f"{model}: {outcome} ({model_index}/{len(models)})",
            advance=1,
            emit=True,
        )
    if failures:
        progress.finish(paper_id, f"finished with failures={len(failures)}")
    else:
        progress.finish(paper_id, "completed")
    return failures


def _run_paper_model(
    paper_id: str,
    model: str,
    result_root: Path,
    force: bool,
    dry_run: bool,
    progress: PaperBatchProgress,
    model_index: int,
    model_total: int,
) -> str:
    out_dir = _result_dir(result_root, model, paper_id)
    paths = _result_paths(out_dir)
    if not force and completed_textonly_result_exists(paths["trajectory"], paths["report"]):
        return "skipped"
    if dry_run:
        return "dry-run"

    context = _load_context(paper_id, model)
    question_ids = [question["question_id"] for question in context["questions"]]
    completed, turns, errors = _load_checkpoint(paths["checkpoint"], paper_id, model, question_ids, force)
    pending = [question for question in context["questions"] if question["question_id"] not in completed]
    if not pending and turns:
        _write_final_outputs(paths, paper_id, model, turns, errors)
        paths["checkpoint"].unlink(missing_ok=True)
        return "finalized from checkpoint"


    progress.update(
        paper_id,
        status=(
            f"{model}: questions {len(completed)}/{len(context['questions'])} "
            f"({model_index}/{model_total})"
        ),
    )
    first_error: Exception | None = None
    for question in pending:
        try:
            turn = run_textonly_question(
                question=question,
                paper_text=context["paper_text"],
                target_client=context["target_client"],
                judge_client=context["judge_client"],
                turn_id=f"TXT{int(question['question_order']):04d}",
                previous_turns=turns,
                use_online_eval=True,
            )
        except Exception as exc:
            error = {
                "question_id": question.get("question_id"),
                "question_type": question.get("question_type"),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "updated_at": _now_iso(),
            }
            errors.append(error)
            _write_checkpoint(paths["checkpoint"], paper_id, model, question_ids, completed, turns, errors)
            if first_error is None:
                first_error = exc
            continue
        completed.add(turn["question_id"])
        turns.append(turn)
        turns.sort(key=lambda item: item.get("question_order", 0))
        _write_checkpoint(paths["checkpoint"], paper_id, model, question_ids, completed, turns, errors)
        progress.update(
            paper_id,
            status=(
                f"{model}: {turn['question_id']} {len(completed)}/{len(context['questions'])} "
                f"({model_index}/{model_total})"
            ),
        )

    if first_error is not None:
        raise RuntimeError(
            f"text-only evaluation failed for paper_id={paper_id} model={model}; "
            f"completed={len(completed)}/{len(context['questions'])} errors={len(errors)}. "
            f"Checkpoint: {paths['checkpoint']}"
        ) from first_error
    _write_final_outputs(paths, paper_id, model, turns, errors)
    paths["checkpoint"].unlink(missing_ok=True)
    return "completed"


def _load_context(paper_id: str, model: str) -> dict[str, Any]:
    settings = load_settings(PROJECT_ROOT)
    layout = PaperArtifactLayout(PROJECT_ROOT, paper_id)
    package_path = layout.root / "textonly_question_templates.json"
    if not package_path.exists():
        raise FileNotFoundError(f"Text-only package not found: {package_path}")
    package = json.loads(package_path.read_text(encoding="utf-8"))
    if package.get("challenge_pipeline", {}).get("mode") != "plan_solver_filter":
        raise ValueError(
            f"Text-only package was not solver-filtered; rebuild with --force: {package_path}"
        )
    questions = textonly_no_repair_questions(package)
    if not questions:
        raise ValueError(f"Text-only package has no questions: {package_path}")
    judge_client = OpenAICompatClient(ModelConfig(settings.api_key, settings.base_url, settings.llm_model))
    if not judge_client.is_ready():
        raise RuntimeError("Text-only evaluation requires configured judge API_KEY/BASE_URL/LLM_MODEL.")
    target_client = _build_target_client(model)
    if not target_client.is_ready():
        raise RuntimeError(
            f"Text-only evaluation requires configured target model API for {model}: "
            "EVAL_TARGET_API_KEY/EVAL_TARGET_BASE_URL or model-specific overrides."
        )
    return {
        "paper_text": load_paper_clean_text(PROJECT_ROOT, paper_id, limit_env="TEXTONLY_EVAL_PAPER_CHAR_LIMIT"),
        "questions": questions,
        "judge_client": judge_client,
        "target_client": target_client,
    }


def _write_final_outputs(paths: dict[str, Path], paper_id: str, model: str, turns: list[dict], errors: list[dict]) -> None:
    trajectory = {
        "paper_id": paper_id,
        "target_model": model,
        "evaluation_mode": TEXTONLY_EVAL_MODE,
        "ablation": {
            "graph_guided_question_generation": False,
            "full_dialogue_context": True,
            "repair_tasks_executed": False,
            "kc_coverage_metrics_computed": False,
        },
        "turns": sorted(turns, key=lambda item: item.get("question_order", 0)),
    }
    report = build_textonly_report(paper_id, model, trajectory["turns"], errors)
    _write_json(paths["trajectory"], trajectory)
    _write_json(paths["report"], report)


def _load_checkpoint(
    checkpoint_path: Path,
    paper_id: str,
    model: str,
    question_ids: list[str],
    force: bool,
) -> tuple[set[str], list[dict], list[dict]]:
    if force or not checkpoint_path.exists():
        return set(), [], []
    data = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    if data.get("paper_id") != paper_id or data.get("target_model") != model:
        raise ValueError(f"Checkpoint identity mismatch: {checkpoint_path}")
    if data.get("evaluation_mode") != TEXTONLY_EVAL_MODE:
        raise ValueError(f"Checkpoint evaluation mode mismatch; rerun with --force: {checkpoint_path}")
    if data.get("question_ids") != question_ids:
        raise ValueError(f"Checkpoint question package mismatch; rerun with --force: {checkpoint_path}")
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
    question_ids: list[str],
    completed: set[str],
    turns: list[dict],
    errors: list[dict],
) -> None:
    _write_json(
        checkpoint_path,
        {
            "paper_id": paper_id,
            "target_model": model,
            "evaluation_mode": TEXTONLY_EVAL_MODE,
            "question_ids": question_ids,
            "completed_question_ids": sorted(completed),
            "turns": sorted(turns, key=lambda item: item.get("question_order", 0)),
            "errors": errors,
            "updated_at": _now_iso(),
        },
    )


def _resolve_models(raw_models: list[str] | None) -> list[str]:
    chunks = raw_models if raw_models else [os.getenv("TEXTONLY_EVAL_MODELS", "")]
    models = [_normalize_model_name(item) for item in _split_values(chunks)]
    return models or DEFAULT_MODELS


def _resolve_paper_ids(raw_paper_ids: list[str] | None) -> list[str]:
    requested = _split_values(raw_paper_ids or [])
    if requested:
        return [_assert_textonly_paper(paper_id) for paper_id in requested]
    paper_ids = []
    data_root = PROJECT_ROOT / "data"
    if not data_root.exists():
        return paper_ids
    for path in sorted(data_root.iterdir(), key=lambda p: p.name):
        if not path.is_dir():
            continue
        try:
            paper_ids.append(_assert_textonly_paper(path.name))
        except FileNotFoundError:
            continue
    return paper_ids


def _assert_textonly_paper(paper_id: str) -> str:
    layout = PaperArtifactLayout(PROJECT_ROOT, paper_id)
    package_path = layout.root / "textonly_question_templates.json"
    paper_path = layout.root / "paper_clean_text.md"
    missing = [str(path) for path in (package_path, paper_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Paper {paper_id!r} is not text-only evaluable; missing: {', '.join(missing)}")
    return layout.paper_id


def _result_dir(result_root: Path, model: str, paper_id: str) -> Path:
    return result_root / _safe_dir_name(model) / _safe_dir_name(paper_id)


def _result_paths(out_dir: Path) -> dict[str, Path]:
    return {
        "trajectory": out_dir / "dialogue_trajectory.json",
        "report": out_dir / "evaluation_report.json",
        "checkpoint": out_dir / "cache" / "textonly_evaluation_checkpoint.json",
    }


def _build_target_client(model: str) -> OpenAICompatClient:
    suffix = _model_env_suffix(model)
    api_key = os.getenv(f"EVAL_TARGET_API_KEY_{suffix}") or os.getenv("EVAL_TARGET_API_KEY", "")
    base_url = os.getenv(f"EVAL_TARGET_BASE_URL_{suffix}") or os.getenv("EVAL_TARGET_BASE_URL", "")
    return OpenAICompatClient(ModelConfig(api_key=api_key, base_url=base_url, llm_model=model))


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


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


if __name__ == "__main__":
    main()
