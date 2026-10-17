"""One documented interface for configuration and the pure-AI pipeline."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from contextlib import contextmanager
from pathlib import Path

from src.config import load_environment
from src.pipeline_plans import PROJECT_ROOT, plan_jobs, select_models
from src.pipeline_runner import execute_jobs, target_environment
from src.preflight import validate_config


CHECK_STAGES = {
    "parse": ["parse_pdf"],
    "build": ["build_graph"],
    "questions": ["generate_questions"],
    "evaluate": ["evaluate"],
    "all": ["build_graph", "generate_questions", "evaluate"],
}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="PaperGraph-Bench: configure, build, generate questions, and evaluate.")
    result.add_argument("--config", help="Configuration file, relative to the repository root (default: config.env).")
    commands = result.add_subparsers(dest="command", required=True)
    for command in ("init", "check", "parse", "run", "build", "questions", "evaluate"):
        sub = commands.add_parser(command, help={
            "init": "Create a private config.env from the visible template.",
            "check": "Validate configuration without API calls, or explicitly probe APIs.",
            "parse": "Parse PDF files with OCR.",
            "run": "Run graph construction, question generation and evaluation in order.",
            "build": "Build paper graphs from parsed Markdown.",
            "questions": "Generate question packages from paper graphs.",
            "evaluate": "Evaluate existing packages from data/.",
        }[command])
        sub.add_argument("--config", default=argparse.SUPPRESS, help="Configuration file (default: config.env).")
        if command == "init":
            continue
        if command == "check":
            sub.add_argument("--stage", choices=CHECK_STAGES, default="all", help="all checks build/questions/evaluate; parse checks OCR separately.")
            sub.add_argument("--probe-apis", action="store_true", help="Explicitly make small paid API capability requests; OCR is not probed.")
            sub.add_argument("--models", nargs="+", help="Target IDs to validate/probe; default: configured target selection.")
            continue
        select = sub.add_mutually_exclusive_group()
        select.add_argument("--paper-id", help="Paper ID under the configured rawPaper/data root.")
        select.add_argument("--all", action="store_true", help="Process all eligible papers.")
        source = sub.add_mutually_exclusive_group()
        source.add_argument("--pdf", help="Input PDF for parse or run; includes OCR.")
        source.add_argument("--input", help="Parsed Markdown file or directory for build or run.")
        sub.add_argument("--models", nargs="+", help="Target model IDs on the configured target service; default: EVAL_TARGET_MODEL.")
        sub.add_argument("--workers", type=int, default=1, help="Concurrent papers; stages/models within each paper remain sequential.")
        sub.add_argument("--force", action="store_true", help="Rebuild selected stages and restart evaluation instead of resuming.")
        sub.add_argument("--dry-run", action="store_true", help="Print the stage plan without requests or output files.")
    return result


@contextmanager
def _target_scope(model: str):
    names = ("EVAL_TARGET_MODEL", "EVAL_TARGET_API_KEY", "EVAL_TARGET_BASE_URL")
    previous = {name: os.environ.get(name) for name in names}
    configured = target_environment(model)
    try:
        for name in names:
            if name in configured:
                os.environ[name] = configured[name]
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _init(config: str | None) -> None:
    destination = Path(config or os.getenv("PAPERGRAPH_CONFIG", "").strip() or "config.env").expanduser()
    if not destination.is_absolute():
        destination = PROJECT_ROOT / destination
    if destination.exists():
        raise FileExistsError(f"Configuration already exists: {destination}. Edit it in place.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(PROJECT_ROOT / "config.example.env", destination)
    print(f"Created {destination.resolve()}\nFill the API sections, then run: python run.py check --stage all")


def _check(args) -> int:
    stages = CHECK_STAGES[args.stage]
    has_models = bool(args.models or os.getenv("EVAL_BATCH_MODELS", "").strip() or os.getenv("EVAL_TARGET_MODEL", "").strip())
    models = select_models(args.models) if "evaluate" in stages and has_models else []
    construction = [stage for stage in stages if stage != "evaluate"]
    errors = validate_config(construction) if construction else []
    for model in models:
        with _target_scope(model):
            errors.extend(f"{model}: {error}" for error in validate_config(["evaluate"]))
    if "evaluate" in stages and not models:
        errors.extend(validate_config(["evaluate"]))
    if errors:
        print("Configuration check failed:\n" + "\n".join(f"  - {error}" for error in errors), file=sys.stderr)
        return 2
    print("Configuration OK (local validation; no provider requests made).")
    if args.probe_apis:
        from src.api_probe import probe_apis
        messages = probe_apis(construction) if construction else []
        for model in models:
            with _target_scope(model):
                messages.extend(f"{model}: {message}" for message in probe_apis(["evaluate"]))
        for message in messages:
            print(message)
        return 1 if any("FAILED" in message or "Configuration failed:" in message for message in messages) else 0
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "init":
            _init(args.config)
            return 0
        if args.config:
            os.environ["PAPERGRAPH_CONFIG"] = str((PROJECT_ROOT / args.config).resolve())
        config_path = load_environment(PROJECT_ROOT)
        if args.command == "check":
            return _check(args)
        if args.workers < 1:
            raise ValueError("--workers must be positive.")
        if args.all and (args.pdf or args.input):
            raise ValueError("--all cannot be combined with --pdf or --input.")
        if args.models and args.command not in {"run", "evaluate"}:
            raise ValueError("--models is available for run and evaluate only.")
        jobs = plan_jobs(args)
        stages = list(dict.fromkeys(stage for job in jobs for stage in job.stages))
        models = select_models(args.models) if "evaluate" in stages else []
        if not args.dry_run:
            construction = [stage for stage in stages if stage != "evaluate"]
            errors = validate_config(construction) if construction else []
            for model in models:
                with _target_scope(model):
                    errors.extend(f"{model}: {error}" for error in validate_config(["evaluate"]))
            if errors:
                raise ValueError("Configuration check failed:\n" + "\n".join(f"  - {error}" for error in errors))
        print(f"[pipeline] config={config_path} papers={len(jobs)} models={len(models)} workers={args.workers}")
        execute_jobs(jobs, models, workers=args.workers, force=args.force, dry_run=args.dry_run)
        return 0
    except KeyboardInterrupt:
        print("Interrupted. Re-run the same command to resume checkpoints.", file=sys.stderr)
        return 130
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
