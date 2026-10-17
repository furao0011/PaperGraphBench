"""Build fixed benchmark packages once, then evaluate independent repetitions."""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "papergraph_demo"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from measurement_helpers import assert_frozen, copy_sources, frozen_hashes, run_stage, source_revision, write_json
from src.config import load_environment
from src.pipeline_cli import main as pipeline_main
from src.pipeline_plans import select_models


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", help="Private configuration file (default: config.env).")
    result.add_argument("--paper-id", help="Measure just one rawPaper directory; omit to measure all papers.")
    result.add_argument("--models", nargs="+", help="Target model IDs; default: configured target selection.")
    result.add_argument("--workers", type=int, default=1, help="Concurrent papers (default: 1).")
    result.add_argument("--repeats", type=int, default=3, help="Independent evaluations of the same packages (default: 3).")
    result.add_argument("--output-root", default="outputs/measurement", help="Parent directory for a new, unique session.")
    result.add_argument("--dry-run", action="store_true", help="Print stages and destinations without API calls or writes.")
    return result


def _path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def _sources(paper_id: str | None) -> list[Path]:
    root = _path(os.environ["RAW_PAPER_ROOT"])
    if paper_id and (Path(paper_id).name != paper_id or paper_id in {".", ".."}):
        raise ValueError("--paper-id must be a single directory name, not a path.")
    candidates = [root / paper_id] if paper_id else sorted(root.iterdir()) if root.is_dir() else []
    sources = [path for path in candidates if path.is_dir() and any(path.glob("*.md")) and not path.name.endswith(".tmp")]
    if not sources:
        raise ValueError(f"No parsed Markdown papers found in {root}; provide rawPaper/<paper-id>/*.md first.")
    return sources


def _measure(args) -> int:
    if args.workers < 1 or args.repeats < 1:
        raise ValueError("--workers and --repeats must both be positive.")
    config = load_environment(ROOT, args.config)
    sources = _sources(args.paper_id)
    configured = bool(args.models or os.getenv("EVAL_TARGET_MODEL", "").strip() or os.getenv("EVAL_BATCH_MODELS", "").strip())
    models = select_models(args.models) if configured else []
    session = _path(args.output_root) / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8])
    selectors = ["--paper-id", args.paper_id] if args.paper_id else ["--all"]
    common = [*selectors, "--workers", str(args.workers), "--force"]
    if config.is_file():
        common += ["--config", str(config)]
    stages = [("build", ["build", *common]), ("questions", ["questions", *common])]
    for index in range(1, args.repeats + 1):
        stages.append((f"round-{index:02d}", ["evaluate", *common, *(["--models", *models] if models else [])]))
    print(f"[measurement] papers={len(sources)} models={','.join(models) or '(configure target)'} repeats={args.repeats}")
    print(f"[measurement] session={session}\n[measurement] fixed packages={session / 'dataset'}")
    for label, command in stages:
        destination = session / label / "evaluation" if label.startswith("round-") else session / "dataset"
        print(f"[plan] {label} -> {destination}")
    if args.dry_run:
        return 0
    check = ["check", "--stage", "all"]
    if config.is_file():
        check += ["--config", str(config)]
    if models:
        check += ["--models", *models]
    if pipeline_main(check):
        raise ValueError("Before running, create config.env with start.ps1 init / ./start.sh init, then fill API keys, base URLs and model IDs.")
    models = select_models(args.models)
    revision = source_revision(ROOT)
    session.mkdir(parents=True, exist_ok=False)
    manifest = {"version": 1, "status": "running", "paper_ids": [source.name for source in sources],
                "paper_count": len(sources), "target_models": models, "repeats": args.repeats,
                "workers": args.workers, "source_revision": revision, "started_at": datetime.now(timezone.utc).isoformat(),
                "model_roles": {name: os.getenv(name, "") for name in ("LLM_MODEL", "VISION_MODEL", "EMBED_MODEL")}}
    write_json(session / "measurement.json", manifest)
    try:
        copy_sources(sources, session / "rawPaper")
        env = {**os.environ, "RAW_PAPER_ROOT": str(session / "rawPaper"), "PAPERGRAPH_DATA_ROOT": str(session / "dataset")}
        hashes = None
        for label, command in stages:
            if label.startswith("round-"):
                assert_frozen(session, hashes)
                env["EVAL_RESULT_ROOT"] = str(session / label / "evaluation")
            env["PAPERGRAPH_LOG_ROOT"] = str(session / "logs" / label)
            print(f"[measurement] starting {label}; log={session / 'main.log'}", flush=True)
            run_stage(command, env, session / "main.log", project_root=ROOT)
            if label == "questions":
                hashes = frozen_hashes(session)
                write_json(session / "frozen_inputs.json", {"algorithm": "sha256", "files": hashes})
            elif label.startswith("round-"):
                assert_frozen(session, hashes)
        manifest["status"] = "completed"
        write_json(session / "measurement.json", manifest)
        print(f"[measurement] completed; results={session}")
        return 0
    except BaseException:
        manifest["status"] = "failed"
        write_json(session / "measurement.json", manifest)
        raise


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    previous = os.environ.copy()
    try:
        return _measure(args)
    except KeyboardInterrupt:
        print("Measurement interrupted; session marked failed. Run again to create a fresh session.", file=sys.stderr)
        return 130
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    finally:
        os.environ.clear()
        os.environ.update(previous)


if __name__ == "__main__":
    raise SystemExit(main())
