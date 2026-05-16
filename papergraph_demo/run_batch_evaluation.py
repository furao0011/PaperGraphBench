from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

from src.artifact_layout import PaperArtifactLayout
from src.config import load_dotenv


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
DEFAULT_EVAL_RESULT_ROOT = PROJECT_ROOT / "eval_result"
DEFAULT_MODELS = [
    "fireworks/kimi-k2p5",
    "ark-doubao-seed-2.0-pro-260215",
    "ark-doubao-seed-2.0-mini-260215",
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

    print(
        f"[batch-eval] papers={len(paper_ids)} models={len(models)} result_root={result_root}",
        flush=True,
    )
    for paper_id in paper_ids:
        for model in models:
            public_dir = _public_result_dir(result_root, model, paper_id)
            if not args.force and _has_public_result(public_dir):
                print(f"[batch-eval] skip existing | paper_id={paper_id} model={model}", flush=True)
                continue
            if args.dry_run:
                print(f"[batch-eval] would run | paper_id={paper_id} model={model}", flush=True)
                continue
            code = _run_one_evaluation(paper_id, model, result_root)
            if code != 0:
                message = f"[batch-eval] failed | paper_id={paper_id} model={model} exit_code={code}"
                print(message, flush=True)
                if not args.continue_on_failure:
                    raise SystemExit(code)
            else:
                print(f"[batch-eval] completed | paper_id={paper_id} model={model}", flush=True)


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
    parser.add_argument("--force", action="store_true", help="Run even when public result files already exist.")
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


def _run_one_evaluation(paper_id: str, model: str, result_root: Path) -> int:
    env = os.environ.copy()
    env["PAPER_ID"] = paper_id
    env["USE_ONLINE_EVAL"] = "true"
    env["EVAL_TARGET_MODEL"] = model
    env["EVAL_RESULT_ROOT"] = str(result_root)
    env["PAPERGRAPH_RESTART"] = "true"
    env["EVAL_RESTART"] = "true"
    env["PAPERGRAPH_RESUME"] = "false"
    env["EVAL_RESUME"] = "false"

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


def _has_public_result(public_dir: Path) -> bool:
    return (
        (public_dir / "dialogue_trajectory.json").exists()
        and (public_dir / "evaluation_report.json").exists()
    )


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
