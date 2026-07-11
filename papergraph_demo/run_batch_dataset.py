from __future__ import annotations

import argparse
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
DEFAULT_LOG_ROOT = PROJECT_ROOT / "logs" / "main_dataset"


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    args = _parse_args()
    if args.skip_graph and args.skip_questions:
        raise ValueError("At least one dataset stage must be enabled.")
    if args.workers <= 0:
        raise ValueError("--workers must be positive.")

    raw_root = Path(args.raw_paper_root or os.getenv("RAW_PAPER_ROOT") or DEFAULT_RAW_PAPER_ROOT).resolve()
    log_root = Path(args.log_dir or os.getenv("DATASET_BATCH_LOG_DIR") or DEFAULT_LOG_ROOT).resolve()
    paper_ids = _resolve_paper_ids(args.paper_ids, raw_root)
    if not paper_ids:
        raise RuntimeError(f"No OCR paper directories found under {raw_root}.")

    stage_total = int(not args.skip_graph) + int(not args.skip_questions)
    tasks = [BatchTask(paper_id=paper_id, total=stage_total) for paper_id in paper_ids]
    failures: list[tuple[str, Exception]] = []
    with PaperBatchProgress("main-dataset", tasks) as progress:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_paper = {
                executor.submit(
                    _run_paper,
                    progress=progress,
                    paper_id=paper_id,
                    raw_root=raw_root,
                    log_root=log_root,
                    force=args.force,
                    dry_run=args.dry_run,
                    skip_graph=args.skip_graph,
                    skip_questions=args.skip_questions,
                ): paper_id
                for paper_id in paper_ids
            }
            for future in as_completed(future_to_paper):
                paper_id = future_to_paper[future]
                try:
                    future.result()
                except Exception as exc:
                    failures.append((paper_id, exc))
                    if not args.continue_on_failure:
                        for pending in future_to_paper:
                            pending.cancel()
                        raise

    if failures:
        raise RuntimeError(f"Batch dataset build finished with {len(failures)} failed papers.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build graphs and graph-guided question packages for multiple OCR papers."
    )
    parser.add_argument(
        "--paper-ids",
        nargs="*",
        help="Optional paper_id subset. Defaults to every complete OCR directory under RAW_PAPER_ROOT.",
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
    parser.add_argument("--log-dir", default="", help="Per-paper subprocess log root.")
    parser.add_argument("--skip-graph", action="store_true", help="Only run question generation.")
    parser.add_argument("--skip-questions", action="store_true", help="Only run graph construction.")
    parser.add_argument("--force", action="store_true", help="Restart enabled stages even if final outputs exist.")
    parser.add_argument("--dry-run", action="store_true", help="Show resolved paper/stage jobs without executing them.")
    parser.add_argument("--continue-on-failure", action="store_true", help="Continue other papers after one fails.")
    return parser.parse_args()


def _resolve_paper_ids(raw_values: list[str] | None, raw_root: Path) -> list[str]:
    requested = _split_values(raw_values or [])
    if requested:
        return [_assert_raw_paper(raw_root, paper_id) for paper_id in requested]
    if not raw_root.exists():
        return []
    paper_ids = []
    for path in sorted(raw_root.iterdir(), key=lambda item: item.name):
        if not path.is_dir() or path.name.endswith(".tmp"):
            continue
        paper_ids.append(_assert_raw_paper(raw_root, path.name))
    return paper_ids


def _assert_raw_paper(raw_root: Path, paper_id: str) -> str:
    layout = PaperArtifactLayout(BASE_DIR, paper_id)
    raw_dir = (raw_root / layout.paper_id).resolve()
    raw_dir.relative_to(raw_root.resolve())
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"OCR paper directory not found: {raw_dir}")
    manifest_path = raw_dir / "papergraph_ocr_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"OCR paper directory has no completion manifest: {manifest_path}")
    if not any(raw_dir.glob("doc_*.md")):
        raise FileNotFoundError(f"OCR paper directory has no doc_*.md files: {raw_dir}")
    return layout.paper_id


def _run_paper(
    *,
    progress: PaperBatchProgress,
    paper_id: str,
    raw_root: Path,
    log_root: Path,
    force: bool,
    dry_run: bool,
    skip_graph: bool,
    skip_questions: bool,
) -> None:
    layout = PaperArtifactLayout(BASE_DIR, paper_id)
    raw_dir = raw_root / layout.paper_id
    graph_path = layout.final("master_graph")
    question_path = layout.final("question_templates")
    progress.start(paper_id, "starting")

    try:
        if not skip_graph:
            if graph_path.exists() and not force:
                progress.update(paper_id, status="graph skipped; questions queued", advance=1, emit=True)
            elif dry_run:
                progress.update(paper_id, status="would build graph", advance=1, emit=True)
            else:
                progress.update(paper_id, status="building graph", emit=True)
                _run_stage("run_build_graph.py", paper_id, raw_dir, log_root, force)
                if not graph_path.is_file():
                    raise RuntimeError(f"Graph process exited successfully but output is missing: {graph_path}")
                progress.update(paper_id, status="graph complete; questions queued", advance=1, emit=True)

        if not skip_questions:
            if not graph_path.exists() and not dry_run:
                raise FileNotFoundError(
                    f"Question generation requires graph output for {paper_id}: {graph_path}"
                )
            if question_path.exists() and not force:
                progress.update(paper_id, status="questions skipped", advance=1, emit=True)
            elif dry_run:
                progress.update(paper_id, status="would generate questions", advance=1, emit=True)
            else:
                progress.update(paper_id, status="generating questions", emit=True)
                _run_stage("run_generate_questions.py", paper_id, raw_dir, log_root, force)
                if not question_path.is_file():
                    raise RuntimeError(
                        f"Question process exited successfully but output is missing: {question_path}"
                    )
                progress.update(paper_id, status="questions complete", advance=1, emit=True)
        progress.finish(paper_id, "completed")
    except Exception as exc:
        progress.fail(paper_id, exc)
        raise


def _run_stage(
    script_name: str,
    paper_id: str,
    raw_dir: Path,
    log_root: Path,
    force: bool,
) -> None:
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
        "PAPERGRAPH_QUESTION_PATH",
        "QUESTION_CACHE_PATH",
        "CHALLENGE_LOOP_CACHE_PATH",
    ):
        env.pop(shared_override, None)

    stage_name = Path(script_name).stem.removeprefix("run_")
    log_path = log_root / _safe_name(paper_id) / f"{stage_name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(
            f"\n[{datetime.now().isoformat(timespec='seconds')}] "
            f"paper_id={paper_id} script={script_name} force={force}\n"
        )
        log_file.flush()
        completed = subprocess.run(
            [sys.executable, str(BASE_DIR / script_name)],
            cwd=str(PROJECT_ROOT),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
    if completed.returncode:
        raise RuntimeError(
            f"{script_name} failed for {paper_id} with exit code {completed.returncode}; log={log_path}"
        )


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


def _safe_name(value: str) -> str:
    safe = re.sub(r"[^\w._-]+", "_", value.strip())
    return safe.strip("._-") or "unknown"


if __name__ == "__main__":
    main()
