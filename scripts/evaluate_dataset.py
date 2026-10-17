"""Test existing benchmark packages and automatically export result statistics."""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "papergraph_demo"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from measurement_helpers import run_stage, source_revision, write_json
from src.artifact_layout import safe_path_component
from src.config import load_environment
from src.pipeline_artifacts import validate_graph_inputs, validate_question_inputs
from src.pipeline_cache import build_stage_signature
from src.pipeline_cli import main as pipeline_main
from src.pipeline_plans import plan_jobs, select_models


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", help="Private API configuration file (default: config.env).")
    selection = result.add_mutually_exclusive_group()
    selection.add_argument("--paper-id", help="Test one existing data/<paper-id> package.")
    selection.add_argument("--all", action="store_true", help="Test every existing package (the default).")
    result.add_argument("--models", nargs="+", help="Target model IDs; defaults to the configured target selection.")
    result.add_argument("--workers", type=int, default=1, help="Concurrent papers (default: 1).")
    result.add_argument("--repeats", type=int, default=1, help="Independent evaluation rounds (default: 1).")
    result.add_argument("--data-root", default="data", help="Existing package root, relative to this repository; default data/ explicitly overrides PAPERGRAPH_DATA_ROOT.")
    result.add_argument("--output-root", default="outputs/testing", help="Parent of a new unique test session.")
    result.add_argument("--dry-run", action="store_true", help="Print the evaluation plan without API calls or output files.")
    return result


def _path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def _signatures(jobs, models: list[str]) -> dict[str, str]:
    return {f"{job.paper_id}/{model}": build_stage_signature(job, "evaluate", model)
            for job in jobs for model in models}


def _assert_unchanged(jobs, models: list[str], expected: dict[str, str]) -> None:
    if _signatures(jobs, models) != expected:
        raise RuntimeError("Existing test packages changed during evaluation; keep inputs unchanged and start a new session.")


def _test(args) -> int:
    if args.workers < 1 or args.repeats < 1:
        raise ValueError("--workers and --repeats must be positive.")
    config = load_environment(ROOT, args.config)
    os.environ["PAPERGRAPH_DATA_ROOT"] = str(_path(args.data_root))
    selection = SimpleNamespace(command="evaluate", paper_id=args.paper_id, all=not args.paper_id,
                                pdf=None, input=None)
    jobs = plan_jobs(selection)
    configured = bool(args.models or os.getenv("EVAL_TARGET_MODEL", "").strip() or os.getenv("EVAL_BATCH_MODELS", "").strip())
    models = select_models(args.models) if configured else []
    if len({os.path.normcase(safe_path_component(model, "model")) for model in models}) != len(models):
        raise ValueError("Selected model IDs produce duplicate output directories; use distinct model IDs.")
    session = _path(args.output_root) / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8])
    common = [*(["--paper-id", args.paper_id] if args.paper_id else ["--all"]),
              "--workers", str(args.workers), "--force"]
    if config.is_file():
        common += ["--config", str(config)]
    command = ["evaluate", *common, *(["--models", *models] if models else [])]
    print(f"[testing] existing dataset={os.environ['PAPERGRAPH_DATA_ROOT']} papers={len(jobs)} models={','.join(models) or '(configure target)'} repeats={args.repeats}")
    print(f"[testing] NO graph construction or question generation; session={session}")
    for index in range(1, args.repeats + 1):
        print(f"[plan] round-{index:02d}: {' '.join(command)} -> {session / f'round-{index:02d}' / 'evaluation'}")
    print(f"[plan] automatic statistics -> {session / 'statistics'}")
    if args.dry_run:
        return 0
    for job in jobs:
        validate_question_inputs(job.paper_id, validate_graph_inputs(job.paper_id))
    check = ["check", "--stage", "evaluate", *(["--config", str(config)] if config.is_file() else []),
             *(["--models", *models] if models else [])]
    if pipeline_main(check):
        raise ValueError("Fill the Text/Judge and Target API keys, BASE URLs and model IDs in config.env before testing.")
    models = select_models(args.models)
    signatures = _signatures(jobs, models)
    expected_tasks = [{"round": f"round-{index:02d}", "paper_id": job.paper_id, "model": model}
                      for index in range(1, args.repeats + 1) for job in jobs for model in models]
    manifest = {"version": 1, "status": "running", "paper_ids": [job.paper_id for job in jobs],
                "target_models": models, "paper_count": len(jobs), "repeats": args.repeats,
                "judge_model": os.getenv("LLM_MODEL", ""),
                "workers": args.workers, "data_root": os.environ["PAPERGRAPH_DATA_ROOT"],
                "source_revision": source_revision(ROOT), "expected_tasks": expected_tasks,
                "input_signatures": signatures, "started_at": datetime.now(timezone.utc).isoformat()}
    session.mkdir(parents=True, exist_ok=False)
    write_json(session / "testing.json", manifest)
    (session / "main.log").touch()
    try:
        for index in range(1, args.repeats + 1):
            label = f"round-{index:02d}"
            _assert_unchanged(jobs, models, signatures)
            env = {**os.environ, "EVAL_RESULT_ROOT": str(session / label / "evaluation"),
                   "PAPERGRAPH_LOG_ROOT": str(session / "logs" / label)}
            print(f"[testing] starting {label}; progress log={session / 'main.log'}", flush=True)
            run_stage(command, env, session / "main.log", project_root=ROOT)
            _assert_unchanged(jobs, models, signatures)
        from summarize_results import print_summary, summarize

        summary = summarize(session, session / "statistics", expected_tasks)
        if summary.get("status") != "completed":
            raise RuntimeError("Statistics did not confirm every planned evaluation as completed.")
        print_summary(summary)
        manifest["status"] = "completed"
        manifest["statistics"] = "statistics"
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        write_json(session / "testing.json", manifest)
        print(f"[testing] completed; summary CSV={session / 'statistics' / 'summary.csv'}")
        print(f"[testing] per-paper CSV={session / 'statistics' / 'per_paper.csv'}; main log={session / 'main.log'}")
        return 0
    except BaseException:
        manifest["status"] = "failed"
        write_json(session / "testing.json", manifest)
        raise


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    previous = os.environ.copy()
    try:
        return _test(args)
    except KeyboardInterrupt:
        print("Testing interrupted; session marked failed. Rerun to start a fresh session.", file=sys.stderr)
        return 130
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    finally:
        os.environ.clear()
        os.environ.update(previous)


if __name__ == "__main__":
    raise SystemExit(main())
