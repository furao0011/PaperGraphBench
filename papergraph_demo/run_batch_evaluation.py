from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from src.artifact_layout import PaperArtifactLayout
from src.batch_progress import BatchTask, PaperBatchProgress
from src.config import load_dotenv


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
DEFAULT_RAW_PAPER_ROOT = PROJECT_ROOT / "rawPaper"
DEFAULT_EVAL_RESULT_ROOT = PROJECT_ROOT / "eval_result"
DEFAULT_LOG_ROOT = PROJECT_ROOT / "logs" / "main_evaluation"
DEFAULT_MODELS = [
    "gpt-5-mini",
    "gpt-5",
    "ark-doubao-seed-2.0-pro-260215",
    "ark-doubao-seed-2.0-mini-260215",
]


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    args = _parse_args()
    models = _resolve_models(args.models)
    raw_root = Path(args.raw_paper_root or os.getenv("RAW_PAPER_ROOT") or DEFAULT_RAW_PAPER_ROOT).resolve()
    paper_ids = _resolve_paper_ids(args.paper_ids, raw_root)
    result_root = Path(args.eval_result_root or os.getenv("EVAL_RESULT_ROOT") or DEFAULT_EVAL_RESULT_ROOT).resolve()
    log_root = Path(args.log_dir or os.getenv("EVAL_BATCH_LOG_DIR") or DEFAULT_LOG_ROOT).resolve()

    if not models:
        raise RuntimeError("No evaluation models configured. Use --models or EVAL_BATCH_MODELS.")
    if not paper_ids:
        raise RuntimeError(f"No evaluable papers selected from {raw_root}.")
    if args.workers <= 0:
        raise ValueError("--workers must be positive.")

    tasks = [BatchTask(paper_id=paper_id, total=len(models)) for paper_id in paper_ids]
    failures: list[dict] = []
    with PaperBatchProgress("main-eval", tasks) as progress:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_paper = {
                executor.submit(
                    _run_paper_models,
                    progress=progress,
                    paper_id=paper_id,
                    models=models,
                    result_root=result_root,
                    log_root=log_root,
                    force=args.force,
                    dry_run=args.dry_run,
                    continue_on_failure=args.continue_on_failure,
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
                        }
                    )
                    if not args.continue_on_failure:
                        for pending in future_to_paper:
                            pending.cancel()
                        raise

    if failures:
        raise RuntimeError(f"Batch evaluation finished with {len(failures)} failed paper/model runs.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run formal evaluations with one worker owning one paper and evaluating its models sequentially."
    )
    parser.add_argument(
        "--models",
        nargs="*",
        help="Target model list. Accepts space-separated values or comma/newline-separated chunks.",
    )
    parser.add_argument(
        "--paper-ids",
        nargs="*",
        help="Optional paper_id subset. Defaults to papers under RAW_PAPER_ROOT.",
    )
    parser.add_argument(
        "--raw-paper-root",
        default="",
        help="Paper selection root. Defaults to RAW_PAPER_ROOT or ./rawPaper.",
    )
    parser.add_argument(
        "--eval-result-root",
        default="",
        help="Public eval_result root. Defaults to EVAL_RESULT_ROOT or ./eval_result.",
    )
    parser.add_argument("--log-dir", default="", help="Per-paper/model subprocess log root.")
    parser.add_argument("--force", action="store_true", help="Restart selected runs even when final results exist.")
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.getenv("EVAL_BATCH_WORKERS", "2") or "2"),
        help="Maximum concurrent papers. Models remain sequential inside each paper.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Show resolved paper/model jobs without executing them.")
    parser.add_argument(
        "--continue-on-failure",
        action="store_true",
        help="Continue remaining models and papers after a failed evaluation.",
    )
    return parser.parse_args()


def _resolve_models(raw_models: list[str] | None) -> list[str]:
    chunks = raw_models if raw_models else [os.getenv("EVAL_BATCH_MODELS", "")]
    models = _split_values(chunks)
    return models or DEFAULT_MODELS


def _resolve_paper_ids(raw_paper_ids: list[str] | None, raw_root: Path) -> list[str]:
    requested = _split_values(raw_paper_ids or [])
    if requested:
        return [_assert_evaluable_paper(paper_id) for paper_id in requested]
    if not raw_root.is_dir():
        return []
    paper_ids = []
    for path in sorted(raw_root.iterdir(), key=lambda item: item.name):
        if not path.is_dir() or path.name.endswith(".tmp"):
            continue
        paper_ids.append(_assert_evaluable_paper(path.name))
    return paper_ids


def _assert_evaluable_paper(paper_id: str) -> str:
    layout = PaperArtifactLayout(BASE_DIR, paper_id)
    required = [
        layout.final("master_graph"),
        layout.final("question_templates"),
        layout.final("paper_clean_text"),
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Paper {paper_id!r} is not evaluable; missing: {', '.join(missing)}")
    return layout.paper_id


def _run_paper_models(
    *,
    progress: PaperBatchProgress,
    paper_id: str,
    models: list[str],
    result_root: Path,
    log_root: Path,
    force: bool,
    dry_run: bool,
    continue_on_failure: bool,
) -> list[dict]:
    failures: list[dict] = []
    progress.start(paper_id, "starting")
    for model in models:
        public_dir = _public_result_dir(result_root, model, paper_id)
        try:
            if not force and _has_completed_public_result(public_dir):
                progress.update(
                    paper_id,
                    status=f"skipped {model}",
                    advance=1,
                    emit=True,
                )
                continue
            if dry_run:
                action = "would restart" if force else "would run/resume"
                progress.update(
                    paper_id,
                    status=f"{action} {model}",
                    advance=1,
                    emit=True,
                )
                continue

            progress.update(paper_id, status=f"evaluating {model}", emit=True)
            _run_one_evaluation(
                paper_id=paper_id,
                model=model,
                result_root=result_root,
                artifact_dir=public_dir,
                log_root=log_root,
                force=force,
            )
            if not _has_completed_public_result(public_dir):
                raise RuntimeError(
                    f"Evaluation process exited successfully but no completed report was written: {public_dir}"
                )
            progress.update(
                paper_id,
                status=f"completed {model}",
                advance=1,
                emit=True,
            )
        except Exception as exc:
            failures.append(
                {
                    "paper_id": paper_id,
                    "model": model,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            progress.update(
                paper_id,
                status=f"failed {model}: {type(exc).__name__}",
                advance=1,
                emit=True,
            )
            if not continue_on_failure:
                progress.fail(paper_id, exc)
                raise
    if failures:
        progress.finish(paper_id, f"finished with {len(failures)} failure(s)")
    else:
        progress.finish(paper_id, "completed")
    return failures


def _run_one_evaluation(
    *,
    paper_id: str,
    model: str,
    result_root: Path,
    artifact_dir: Path,
    log_root: Path,
    force: bool,
) -> None:
    env = os.environ.copy()
    env["PAPER_ID"] = paper_id
    env["USE_ONLINE_EVAL"] = "true"
    env["EVAL_TARGET_MODEL"] = model
    env["EVAL_RESULT_ROOT"] = str(result_root)
    env["EVAL_ARTIFACT_DIR"] = str(artifact_dir)
    env["PAPERGRAPH_RESTART"] = "true" if force else "false"
    env["EVAL_RESTART"] = "true" if force else "false"
    env["PAPERGRAPH_RESUME"] = "false" if force else "true"
    env["EVAL_RESUME"] = "false" if force else "true"
    env.pop("EVAL_CHECKPOINT_PATH", None)

    model_env_suffix = _model_env_suffix(model)
    for name in ("EVAL_TARGET_API_KEY", "EVAL_TARGET_BASE_URL"):
        override = os.getenv(f"{name}_{model_env_suffix}", "").strip()
        if override:
            env[name] = override

    log_path = log_root / _safe_dir_name(paper_id) / f"{_safe_dir_name(model)}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(
            f"\n[{datetime.now().isoformat(timespec='seconds')}] "
            f"paper_id={paper_id} model={model} force={force}\n"
        )
        log_file.flush()
        completed = subprocess.run(
            [sys.executable, str(BASE_DIR / "run_evaluation.py")],
            cwd=str(PROJECT_ROOT),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
    if completed.returncode:
        raise RuntimeError(
            f"Evaluation failed for paper_id={paper_id} model={model} "
            f"with exit code {completed.returncode}; log={log_path}"
        )


def _public_result_dir(result_root: Path, model: str, paper_id: str) -> Path:
    return result_root / _safe_dir_name(model) / _safe_dir_name(paper_id)


def _has_completed_public_result(public_dir: Path) -> bool:
    trajectory_path = public_dir / "dialogue_trajectory.json"
    report_path = public_dir / "evaluation_report.json"
    if not trajectory_path.exists() or not report_path.exists():
        return False
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return report.get("summary", {}).get("evaluation_status") == "completed"


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


def _safe_dir_name(value: object) -> str:
    safe = re.sub(r"[^\w._-]+", "_", str(value or "").strip())
    return safe.strip("._-") or "unknown"


def _model_env_suffix(model: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9]+", "_", model.strip()).strip("_")
    return safe.upper()


if __name__ == "__main__":
    main()
