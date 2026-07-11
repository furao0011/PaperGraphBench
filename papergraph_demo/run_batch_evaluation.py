from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from src.artifact_layout import PaperArtifactLayout
from src.config import load_dotenv


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
DEFAULT_EVAL_RESULT_ROOT = PROJECT_ROOT / "eval_result"
DEFAULT_MODELS = [
    "qwen3.6-plus"
]


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    args = _parse_args()
    models = _resolve_models(args.models)
    paper_ids = _resolve_paper_ids(args.paper_ids)
    result_root = Path(args.eval_result_root or os.getenv("EVAL_RESULT_ROOT") or DEFAULT_EVAL_RESULT_ROOT)

    if not models:
        raise RuntimeError("No evaluation models configured. Use --models or EVAL_BATCH_MODELS.")
    if not paper_ids:
        raise RuntimeError("No evaluable papers found under papergraph_demo/data.")

    if args.workers <= 0:
        raise ValueError("--workers must be positive.")
    print(
        f"[batch-eval] papers={len(paper_ids)} models={len(models)} "
        f"workers={args.workers} result_root={result_root}",
        flush=True,
    )
    jobs = [(paper_id, model) for paper_id in paper_ids for model in models]
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_job = {
            executor.submit(_run_job, paper_id, model, result_root, args.force, args.dry_run): (paper_id, model)
            for paper_id, model in jobs
        }
        for future in as_completed(future_to_job):
            paper_id, model = future_to_job[future]
            try:
                code = future.result()
            except Exception as exc:
                code = 1
                print(
                    f"[batch-eval] failed | paper_id={paper_id} model={model} "
                    f"error={type(exc).__name__}: {exc}",
                    flush=True,
                )
            if code:
                failures.append((paper_id, model, code))
                if not args.continue_on_failure:
                    for pending in future_to_job:
                        pending.cancel()
                    raise SystemExit(code)
    if failures:
        raise SystemExit(f"Batch evaluation finished with {len(failures)} failed paper/model jobs.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run formal evaluations for every paper/model pair.")
    parser.add_argument(
        "--models",
        nargs="*",
        help="Target model list. Accepts space-separated values or comma/newline-separated chunks.",
    )
    parser.add_argument(
        "--paper-ids",
        nargs="*",
        help="Optional paper_id subset. Defaults to every evaluable directory under papergraph_demo/data.",
    )
    parser.add_argument(
        "--eval-result-root",
        default="",
        help="Public eval_result root. Defaults to EVAL_RESULT_ROOT or ./eval_result.",
    )
    parser.add_argument("--force", action="store_true", help="Restart even when a checkpoint or final result exists.")
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.getenv("EVAL_BATCH_WORKERS", "4") or "4"),
        help="Maximum concurrent paper/model evaluation processes.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print scheduled runs without executing evaluation.")
    parser.add_argument(
        "--continue-on-failure",
        action="store_true",
        help="Continue remaining paper/model pairs after a failed evaluation.",
    )
    return parser.parse_args()


def _resolve_models(raw_models: list[str] | None) -> list[str]:
    chunks = raw_models if raw_models else [os.getenv("EVAL_BATCH_MODELS", "")]
    models = _split_values(chunks)
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
    required = [
        layout.final("master_graph"),
        layout.final("question_templates"),
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Paper {paper_id!r} is not evaluable; missing: {', '.join(missing)}")
    return layout.paper_id


def _run_job(paper_id: str, model: str, result_root: Path, force: bool, dry_run: bool) -> int:
    public_dir = _public_result_dir(result_root, model, paper_id)
    if not force and _has_completed_public_result(public_dir):
        print(f"[batch-eval] skip completed | paper_id={paper_id} model={model}", flush=True)
        return 0
    if dry_run:
        action = "restart" if force else "resume/run"
        print(
            f"[batch-eval] would {action} | paper_id={paper_id} model={model} out_dir={public_dir}",
            flush=True,
        )
        return 0
    return _run_one_evaluation(paper_id, model, result_root, public_dir, force)


def _run_one_evaluation(
    paper_id: str,
    model: str,
    result_root: Path,
    artifact_dir: Path,
    force: bool,
) -> int:
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

    print(f"[batch-eval] run | paper_id={paper_id} model={model}", flush=True)
    completed = subprocess.run(
        [sys.executable, str(BASE_DIR / "run_evaluation.py")],
        cwd=str(PROJECT_ROOT),
        env=env,
    )
    return completed.returncode


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
