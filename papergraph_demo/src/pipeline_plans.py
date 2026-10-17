"""Resolve public CLI selections into explicit, paper-scoped stage plans."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from src.artifact_layout import PaperArtifactLayout, safe_path_component


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = {
    "parse_pdf": "run_parse_pdf.py",
    "build_graph": "run_build_graph.py",
    "generate_questions": "run_generate_questions.py",
    "evaluate": "run_evaluation.py",
}


@dataclass(frozen=True)
class PaperJob:
    paper_id: str
    stages: tuple[str, ...]
    source: Path | None = None
    pdf: Path | None = None


def project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def paper_layout(paper_id: str) -> PaperArtifactLayout:
    return PaperArtifactLayout(PROJECT_ROOT / "papergraph_demo", paper_id)


def result_dir(model: str, paper_id: str) -> Path:
    root = project_path(os.environ["EVAL_RESULT_ROOT"])
    return root / safe_path_component(model, "model") / safe_path_component(paper_id, "paper_id")


def select_models(requested: list[str] | None) -> list[str]:
    configured = os.getenv("EVAL_BATCH_MODELS", "").strip()
    values = requested or ([configured] if configured else [os.getenv("EVAL_TARGET_MODEL", "")])
    models = list(dict.fromkeys(item.strip() for value in values for item in value.replace(",", " ").split() if item.strip()))
    if not models:
        raise ValueError("Set EVAL_TARGET_MODEL in config.env, or pass --models explicitly.")
    return models


def _paper_id(value: str) -> str:
    result = safe_path_component(value, "paper_id")
    if result != value:
        raise ValueError(f"Invalid paper_id {value!r}; use letters, digits, underscores, dots or hyphens.")
    return result


def _has_markdown(path: Path) -> bool:
    return path.is_dir() and any(path.glob("*.md"))


def plan_jobs(args) -> list[PaperJob]:
    stages = {
        "run": ("build_graph", "generate_questions", "evaluate"),
        "build": ("build_graph",),
        "questions": ("generate_questions",),
        "evaluate": ("evaluate",),
        "parse": ("parse_pdf",),
    }[args.command]
    raw_root = project_path(os.environ["RAW_PAPER_ROOT"])
    if args.pdf:
        if args.command not in {"run", "parse"}:
            raise ValueError("--pdf is available for run and parse only.")
        pdf = project_path(args.pdf)
        if not pdf.is_file() or pdf.suffix.lower() != ".pdf":
            raise FileNotFoundError(f"PDF not found: {pdf}")
        paper_id = _paper_id(args.paper_id or safe_path_component(pdf.stem, "PDF filename"))
        pipeline = stages if args.command == "parse" else ("parse_pdf", *stages)
        return [PaperJob(paper_id, pipeline, raw_root / paper_id, pdf)]

    if args.command == "parse":
        if not args.all:
            raise ValueError("Use parse --pdf path/to/paper.pdf, or parse --all.")
        input_root = project_path(os.getenv("PAPERGRAPH_PDF_INPUT_DIR", "pdfInput"))
        pdfs = sorted(input_root.glob("*.pdf")) if input_root.is_dir() else []
        if not pdfs:
            raise FileNotFoundError(f"No PDF files in {input_root}.")
        ids = [safe_path_component(pdf.stem, "PDF filename") for pdf in pdfs]
        if len(ids) != len(set(ids)):
            raise ValueError("PDF filenames produce duplicate paper IDs; rename them before parsing.")
        return [PaperJob(pid, stages, raw_root / pid, pdf) for pid, pdf in zip(ids, pdfs)]

    if args.input:
        if args.command not in {"run", "build"}:
            raise ValueError("--input is available for run and build only.")
        source = project_path(args.input)
        if not (_has_markdown(source) or (source.is_file() and source.suffix.lower() == ".md")):
            raise ValueError(f"Expected an OCR/Markdown directory or a .md file: {source}")
        source_id = source.name if source.is_dir() else source.stem
        return [PaperJob(_paper_id(args.paper_id or source_id), stages, source)]

    if args.all:
        source_root = raw_root if args.command in {"run", "build"} else paper_layout("_selection").data_root
        dirs = sorted(source_root.iterdir()) if source_root.is_dir() else []
        if args.command in {"run", "build"}:
            dirs = [path for path in dirs if _has_markdown(path) and not path.name.endswith(".tmp")]
        else:
            dirs = [path for path in dirs if path.is_dir() and (path / "master_graph.json").is_file()]
        ids = [_paper_id(path.name) for path in dirs]
        if not ids:
            raise FileNotFoundError(f"No eligible papers in {source_root}.")
    else:
        configured_id = args.paper_id or os.getenv("PAPER_ID", "").strip()
        if not configured_id:
            raise ValueError("Select --paper-id ID, --input PATH, --pdf PATH, or --all.")
        ids = [_paper_id(configured_id)]

    jobs = [PaperJob(pid, stages, raw_root / pid) for pid in ids]
    for job in jobs:
        if "build_graph" in stages and not _has_markdown(job.source):
            raise FileNotFoundError(f"No parsed Markdown found in {job.source}; use --pdf or --input.")
        if args.command in {"questions", "evaluate"}:
            require_json(paper_layout(job.paper_id).final("master_graph"), job.paper_id)
        if args.command == "evaluate":
            require_json(paper_layout(job.paper_id).final("question_templates"), job.paper_id)
    return jobs


def require_json(path: Path, paper_id: str) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Required artifact not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError) as exc:
        raise ValueError(f"Invalid artifact JSON: {path}") from exc
    if not isinstance(payload, dict) or payload.get("paper_id") != paper_id:
        raise ValueError(f"Artifact must contain paper_id={paper_id!r}: {path}")
    return payload
