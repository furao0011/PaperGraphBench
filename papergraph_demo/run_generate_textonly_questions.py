from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from src.artifact_layout import PaperArtifactLayout
from src.config import load_settings
from src.model_client import ModelConfig, OpenAICompatClient
from src.progress import log, span
from src.textonly_benchmark import generate_textonly_question_package, load_paper_clean_text, load_textonly_multimodal_assets


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent


def main() -> None:
    settings = load_settings(PROJECT_ROOT)
    args = _parse_args()
    paper_ids = _resolve_paper_ids(args.paper_ids)
    if not paper_ids:
        raise RuntimeError("No paper_clean_text.md files found under papergraph_demo/data.")

    client = OpenAICompatClient(ModelConfig(settings.api_key, settings.base_url, settings.llm_model))
    if not client.is_ready():
        raise RuntimeError("Text-only question generation requires API_KEY, BASE_URL, and LLM_MODEL.")

    if args.workers <= 0:
        raise ValueError("--workers must be positive.")
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_paper = {
            executor.submit(
                _generate_for_paper,
                paper_id=paper_id,
                client=client,
                macro_count=args.macro_count,
                challenge_count=args.challenge_count,
                multimodal_challenge_count=args.multimodal_challenge_count,
                multimodal_asset_limit=args.multimodal_asset_limit,
                temperature=args.temperature,
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
                print(
                    f"[textonly-generate] failed | paper_id={paper_id} "
                    f"error={type(exc).__name__}: {exc}",
                    flush=True,
                )
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
    parser.add_argument("--challenge-count", type=int, default=int(os.getenv("TEXTONLY_CHALLENGE_COUNT", "10") or "10"))
    parser.add_argument("--multimodal-challenge-count", type=int, default=int(os.getenv("TEXTONLY_MULTIMODAL_CHALLENGE_COUNT", "10") or "10"))
    parser.add_argument("--multimodal-asset-limit", type=int, default=int(os.getenv("TEXTONLY_MULTIMODAL_ASSET_LIMIT", "40") or "40"))
    parser.add_argument("--temperature", type=float, default=float(os.getenv("TEXTONLY_GENERATION_TEMPERATURE", "0.2") or "0.2"))
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


def _generate_for_paper(
    paper_id: str,
    client: OpenAICompatClient,
    macro_count: int,
    challenge_count: int,
    multimodal_challenge_count: int,
    multimodal_asset_limit: int,
    temperature: float,
    force: bool,
    dry_run: bool,
) -> None:
    layout = PaperArtifactLayout(BASE_DIR, paper_id)
    output_path = layout.root / "textonly_question_templates.json"
    if output_path.exists() and not force:
        print(f"[textonly-generate] skip existing | paper_id={paper_id} path={output_path}", flush=True)
        return
    if dry_run:
        print(f"[textonly-generate] would run | paper_id={paper_id} output={output_path}", flush=True)
        return
    paper_text = load_paper_clean_text(BASE_DIR, paper_id, limit_env="TEXTONLY_GENERATION_PAPER_CHAR_LIMIT")
    multimodal_assets = load_textonly_multimodal_assets(BASE_DIR, paper_id, multimodal_asset_limit)
    if multimodal_challenge_count > 0 and not multimodal_assets:
        raise RuntimeError(
            f"TEXTONLY_MULTIMODAL_CHALLENGE_COUNT={multimodal_challenge_count} requires usable multimodal assets for {paper_id}."
        )
    log(
        "text-only question generation input ready",
        paper_id=paper_id,
        paper_chars=len(paper_text),
        macro_count=macro_count,
        challenge_count=challenge_count,
        multimodal_challenge_count=multimodal_challenge_count,
        multimodal_assets=len(multimodal_assets),
    )
    with span("generate text-only question package", paper_id=paper_id):
        package = generate_textonly_question_package(
            paper_id=paper_id,
            paper_text=paper_text,
            client=client,
            macro_count=macro_count,
            challenge_count=challenge_count,
            multimodal_challenge_count=multimodal_challenge_count,
            multimodal_assets=multimodal_assets,
            temperature=temperature,
        )
    _write_json(output_path, package)
    print(f"[textonly-generate] written | paper_id={paper_id} path={output_path}", flush=True)


def _resolve_paper_ids(raw_paper_ids: list[str] | None) -> list[str]:
    if raw_paper_ids:
        return [PaperArtifactLayout(BASE_DIR, paper_id).paper_id for paper_id in raw_paper_ids]
    paper_ids = []
    data_root = BASE_DIR / "data"
    if not data_root.exists():
        return paper_ids
    for path in sorted(data_root.iterdir(), key=lambda item: item.name):
        if path.is_dir() and (path / "paper_clean_text.md").exists():
            paper_ids.append(PaperArtifactLayout(BASE_DIR, path.name).paper_id)
    return paper_ids


def _write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


if __name__ == "__main__":
    main()
