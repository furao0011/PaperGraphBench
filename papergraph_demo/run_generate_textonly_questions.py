from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from src.artifact_layout import PaperArtifactLayout
from src.batch_progress import BatchTask, PaperBatchProgress
from src.config import load_settings
from src.json_io import write_json_atomic
from src.model_client import ModelConfig, OpenAICompatClient
from src.multimodal_explainer import build_vision_client
from src.progress import log, span
from src.textonly_benchmark import load_paper_clean_text, load_textonly_multimodal_assets
from src.textonly_challenge_pipeline import build_filtered_textonly_package


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent


def main() -> None:
    settings = load_settings(PROJECT_ROOT)
    args = _parse_args()
    paper_ids = _resolve_paper_ids(args.paper_ids)
    if not paper_ids:
        raise RuntimeError("No paper_clean_text.md files found under data/.")

    client = OpenAICompatClient(ModelConfig(settings.api_key, settings.base_url, settings.llm_model))
    if not client.is_ready():
        raise RuntimeError("Text-only question generation requires API_KEY, BASE_URL, and LLM_MODEL.")
    vision_client = None
    if args.multimodal_challenge_count:
        vision_client = build_vision_client(
            embed_api_key=settings.embed_api_key,
            vision_api_key=settings.vision_api_key,
            vision_base_url=settings.vision_base_url,
            vision_model=settings.vision_model,
        )

    if args.workers <= 0:
        raise ValueError("--workers must be positive.")
    if args.schema_attempts <= 0:
        raise ValueError("--schema-attempts must be positive.")
    failures = []
    tasks = [BatchTask(paper_id=paper_id, total=1) for paper_id in paper_ids]
    with PaperBatchProgress("textonly-build", tasks) as progress:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_paper = {
                executor.submit(
                    _run_generation_job,
                    progress=progress,
                    paper_id=paper_id,
                    client=client,
                    vision_client=vision_client,
                    macro_count=args.macro_count,
                    text_plan_count=args.text_plan_count,
                    multimodal_plan_count=args.multimodal_plan_count,
                    challenge_count=args.challenge_count,
                    multimodal_challenge_count=args.multimodal_challenge_count,
                    multimodal_asset_limit=args.multimodal_asset_limit,
                    temperature=args.temperature,
                    generation_attempts=args.schema_attempts,
                    force=args.force,
                    dry_run=args.dry_run,
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
        raise RuntimeError(f"Text-only generation finished with {len(failures)} failed papers.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate no-graph text-only benchmark question packages.")
    parser.add_argument("--paper-ids", nargs="*", help="Optional paper_id subset. Defaults to every data/<paper_id> with paper_clean_text.md.")
    parser.add_argument("--macro-count", type=int, default=int(os.getenv("TEXTONLY_MACRO_COUNT", "8") or "8"))
    parser.add_argument("--text-plan-count", type=int, default=int(os.getenv("TEXTONLY_CHALLENGE_PLAN_POOL", "40") or "40"))
    parser.add_argument("--multimodal-plan-count", type=int, default=int(os.getenv("TEXTONLY_MULTIMODAL_CHALLENGE_PLAN_POOL", "40") or "40"))
    parser.add_argument("--challenge-count", type=int, default=int(os.getenv("TEXTONLY_CHALLENGE_COUNT", "10") or "10"))
    parser.add_argument("--multimodal-challenge-count", type=int, default=int(os.getenv("TEXTONLY_MULTIMODAL_CHALLENGE_COUNT", "10") or "10"))
    parser.add_argument("--multimodal-asset-limit", type=int, default=int(os.getenv("TEXTONLY_MULTIMODAL_ASSET_LIMIT", "40") or "40"))
    parser.add_argument("--temperature", type=float, default=float(os.getenv("TEXTONLY_GENERATION_TEMPERATURE", "0.2") or "0.2"))
    parser.add_argument(
        "--schema-attempts",
        type=int,
        default=int(os.getenv("TEXTONLY_GENERATION_SCHEMA_ATTEMPTS", "3") or "3"),
        help="Maximum generation attempts when a candidate payload fails strict schema validation.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.getenv("TEXTONLY_GENERATION_WORKERS", "2") or "2"),
        help="Maximum concurrent paper generations.",
    )
    parser.add_argument("--force", action="store_true", help="Regenerate even if textonly_question_templates.json exists.")
    parser.add_argument("--dry-run", action="store_true", help="Print scheduled generations without calling models.")
    parser.add_argument("--continue-on-failure", action="store_true", help="Continue other papers after one generation fails.")
    return parser.parse_args()


def _run_generation_job(
    progress: PaperBatchProgress,
    paper_id: str,
    client: OpenAICompatClient,
    vision_client: OpenAICompatClient | None,
    macro_count: int,
    text_plan_count: int,
    multimodal_plan_count: int,
    challenge_count: int,
    multimodal_challenge_count: int,
    multimodal_asset_limit: int,
    temperature: float,
    generation_attempts: int,
    force: bool,
    dry_run: bool,
) -> str:
    progress.start(paper_id, "running")
    try:
        outcome = _generate_for_paper(
            paper_id=paper_id,
            client=client,
            vision_client=vision_client,
            macro_count=macro_count,
            text_plan_count=text_plan_count,
            multimodal_plan_count=multimodal_plan_count,
            challenge_count=challenge_count,
            multimodal_challenge_count=multimodal_challenge_count,
            multimodal_asset_limit=multimodal_asset_limit,
            temperature=temperature,
            generation_attempts=generation_attempts,
            force=force,
            dry_run=dry_run,
        )
    except Exception as exc:
        progress.fail(paper_id, exc)
        raise
    progress.finish(paper_id, outcome)
    return outcome


def _generate_for_paper(
    paper_id: str,
    client: OpenAICompatClient,
    vision_client: OpenAICompatClient | None,
    macro_count: int,
    text_plan_count: int,
    multimodal_plan_count: int,
    challenge_count: int,
    multimodal_challenge_count: int,
    multimodal_asset_limit: int,
    temperature: float,
    generation_attempts: int,
    force: bool,
    dry_run: bool,
) -> str:
    layout = PaperArtifactLayout(PROJECT_ROOT, paper_id)
    output_path = layout.root / "textonly_question_templates.json"
    if output_path.exists() and not force:
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        if existing.get("challenge_pipeline", {}).get("mode") != "plan_solver_filter":
            raise RuntimeError(f"Legacy no-graph package requires --force rebuild: {output_path}")
        return "skipped: filtered output exists"
    if dry_run:
        return "dry-run"
    if text_plan_count < challenge_count:
        raise ValueError("text plan pool must be at least the accepted text challenge target.")
    if multimodal_plan_count < multimodal_challenge_count:
        raise ValueError("multimodal plan pool must be at least the accepted multimodal challenge target.")
    paper_text = load_paper_clean_text(PROJECT_ROOT, paper_id, limit_env="TEXTONLY_GENERATION_PAPER_CHAR_LIMIT")
    multimodal_assets = load_textonly_multimodal_assets(PROJECT_ROOT, paper_id, multimodal_asset_limit)
    if multimodal_challenge_count > 0 and not multimodal_assets:
        raise RuntimeError(
            f"TEXTONLY_MULTIMODAL_CHALLENGE_COUNT={multimodal_challenge_count} requires usable multimodal assets for {paper_id}."
        )
    log(
        "text-only question generation input ready",
        paper_id=paper_id,
        paper_chars=len(paper_text),
        macro_count=macro_count,
        text_plan_count=text_plan_count,
        multimodal_plan_count=multimodal_plan_count,
        challenge_count=challenge_count,
        multimodal_challenge_count=multimodal_challenge_count,
        multimodal_assets=len(multimodal_assets),
        generation_attempts=generation_attempts,
    )
    with span("generate and filter no-graph question package", paper_id=paper_id):
        package = build_filtered_textonly_package(
            paper_id=paper_id,
            paper_text=paper_text,
            multimodal_assets=multimodal_assets,
            text_client=client,
            vision_client=vision_client,
            cache_dir=layout.root / "cache" / "textonly",
            macro_count=macro_count,
            text_plan_count=text_plan_count,
            multimodal_plan_count=multimodal_plan_count,
            text_accept_count=challenge_count,
            multimodal_accept_count=multimodal_challenge_count,
            temperature=temperature,
            restart=force,
            generation_attempts=generation_attempts,
        )
    _write_json(output_path, package)
    return "completed"


def _resolve_paper_ids(raw_paper_ids: list[str] | None) -> list[str]:
    if raw_paper_ids:
        return [PaperArtifactLayout(PROJECT_ROOT, paper_id).paper_id for paper_id in raw_paper_ids]
    paper_ids = []
    data_root = PROJECT_ROOT / "data"
    if not data_root.exists():
        return paper_ids
    for path in sorted(data_root.iterdir(), key=lambda item: item.name):
        if path.is_dir() and (path / "paper_clean_text.md").exists():
            paper_ids.append(PaperArtifactLayout(PROJECT_ROOT, path.name).paper_id)
    return paper_ids


def _write_json(path: Path, payload: dict | list) -> None:
    write_json_atomic(path, payload)


if __name__ == "__main__":
    main()
