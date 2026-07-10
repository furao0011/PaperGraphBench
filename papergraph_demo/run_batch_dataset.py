from __future__ import annotations

import argparse
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
DEFAULT_RAW_PAPER_ROOT = PROJECT_ROOT / "rawPaper"


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    args = _parse_args()
    if args.skip_graph and args.skip_questions:
        raise ValueError("At least one dataset stage must be enabled.")
    if args.workers <= 0:
        raise ValueError("--workers must be positive.")

    raw_root = Path(args.raw_paper_root or os.getenv("RAW_PAPER_ROOT") or DEFAULT_RAW_PAPER_ROOT)
    paper_ids = _resolve_paper_ids(args.paper_ids, raw_root)
    if not paper_ids:
        raise RuntimeError(f"No OCR paper directories found under {raw_root}.")

    print(
        f"[batch-dataset] start | papers={len(paper_ids)} workers={args.workers} "
        f"raw_root={raw_root}",
        flush=True,
    )
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_paper = {
            executor.submit(
                _run_paper,
                paper_id,
                raw_root,
                args.force,
                args.dry_run,
                args.skip_graph,
                args.skip_questions,
            ): paper_id
            for paper_id in paper_ids
        }
        for future in as_completed(future_to_paper):
            paper_id = future_to_paper[future]
            try:
                code = future.result()
            except Exception as exc:
                code = 1
                print(
                    f"[batch-dataset] failed | paper_id={paper_id} "
                    f"error={type(exc).__name__}: {exc}",
                    flush=True,
                )
            if code:
                failures.append((paper_id, code))
                if not args.continue_on_failure:
                    for pending in future_to_paper:
                        pending.cancel()
                    raise SystemExit(code)

    if failures:
        raise SystemExit(f"Batch dataset build finished with {len(failures)} failed papers.")
    print("[batch-dataset] finished", flush=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build graphs and graph-guided question packages for multiple OCR papers."
    )
    parser.add_argument(
        "--paper-ids",
        nargs="*",
        help="Optional paper_id subset. Defaults to every directory under RAW_PAPER_ROOT.",
    )
    parser.add_argument(
        "--raw-paper-root",
        default="",
        help="OCR paper root. Defaults to RAW_PAPER_ROOT or ./rawPaper.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.getenv("DATASET_BATCH_WORKERS", "2") or "2"),
        help="Maximum concurrent papers. Stages remain sequential inside each paper.",
    )
    parser.add_argument("--skip-graph", action="store_true", help="Only run question generation.")
    parser.add_argument("--skip-questions", action="store_true", help="Only run graph construction.")
    parser.add_argument("--force", action="store_true", help="Restart enabled stages even if final outputs exist.")
    parser.add_argument("--dry-run", action="store_true", help="Print scheduled stages without executing them.")
    parser.add_argument("--continue-on-failure", action="store_true", help="Continue other papers after one fails.")
    return parser.parse_args()


def _resolve_paper_ids(raw_values: list[str] | None, raw_root: Path) -> list[str]:
    requested = _split_values(raw_values or [])
    if requested:
        return [_assert_raw_paper(raw_root, paper_id) for paper_id in requested]
    if not raw_root.exists():
        return []
    return [
        _assert_raw_paper(raw_root, path.name)
        for path in sorted(raw_root.iterdir(), key=lambda item: item.name)
        if path.is_dir()
    ]


def _assert_raw_paper(raw_root: Path, paper_id: str) -> str:
    layout = PaperArtifactLayout(BASE_DIR, paper_id)
    raw_dir = raw_root / layout.paper_id
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"OCR paper directory not found: {raw_dir}")
    return layout.paper_id


def _run_paper(
    paper_id: str,
    raw_root: Path,
    force: bool,
    dry_run: bool,
    skip_graph: bool,
    skip_questions: bool,
) -> int:
    layout = PaperArtifactLayout(BASE_DIR, paper_id)
    raw_dir = raw_root / layout.paper_id
    graph_path = layout.final("master_graph")
    question_path = layout.final("question_templates")

    if not skip_graph:
        if graph_path.exists() and not force:
            print(f"[batch-dataset] skip graph | paper_id={paper_id}", flush=True)
        elif dry_run:
            print(f"[batch-dataset] would build graph | paper_id={paper_id}", flush=True)
        else:
            code = _run_stage("run_build_graph.py", paper_id, raw_dir, force)
            if code:
                return code
            print(f"[batch-dataset] graph completed | paper_id={paper_id}", flush=True)

    if not skip_questions:
        if not graph_path.exists() and not dry_run:
            raise FileNotFoundError(
                f"Question generation requires graph output for {paper_id}: {graph_path}"
            )
        if question_path.exists() and not force:
            print(f"[batch-dataset] skip questions | paper_id={paper_id}", flush=True)
        elif dry_run:
            print(f"[batch-dataset] would generate questions | paper_id={paper_id}", flush=True)
        else:
            code = _run_stage("run_generate_questions.py", paper_id, raw_dir, force)
            if code:
                return code
            print(f"[batch-dataset] questions completed | paper_id={paper_id}", flush=True)
    return 0


def _run_stage(script_name: str, paper_id: str, raw_dir: Path, force: bool) -> int:
    env = os.environ.copy()
    env["PAPER_ID"] = paper_id
    env["PAPER_INPUT_DIR"] = str(raw_dir)
    env["PAPERGRAPH_RESTART"] = "true" if force else "false"
    env["PAPERGRAPH_RESUME"] = "false" if force else "true"
    env["BUILD_GRAPH_RESTART"] = "true" if force else "false"
    env["BUILD_GRAPH_RESUME"] = "false" if force else "true"
    env["QUESTION_RESTART"] = "true" if force else "false"
    env["QUESTION_RESUME"] = "false" if force else "true"

    for shared_override in (
        "PAPER_INPUT_FILE",
        "PAPERGRAPH_GRAPH_PATH",
        "QUESTION_CACHE_PATH",
        "CHALLENGE_LOOP_CACHE_PATH",
    ):
        env.pop(shared_override, None)

    print(f"[batch-dataset] run | paper_id={paper_id} script={script_name}", flush=True)
    completed = subprocess.run(
        [sys.executable, str(BASE_DIR / script_name)],
        cwd=str(PROJECT_ROOT),
        env=env,
    )
    return completed.returncode


def _split_values(values: list[str]) -> list[str]:
    result = []
    seen = set()
    for value in values:
        for item in re.split(r"[\n,]+", value):
            item = item.strip()
            if not item or item in seen:
                continue
            result.append(item)
            seen.add(item)
    return result


if __name__ == "__main__":
    main()