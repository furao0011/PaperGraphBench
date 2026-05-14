from __future__ import annotations

import os
from pathlib import Path

from src.paper_parser import load_paper_text, load_paper_text_from_dir


def load_full_paper_text(
    graph: dict,
    base_dir: Path,
    limit_env: str = "EVAL_PAPER_CHAR_LIMIT",
    prefer_evaluation_context: bool | None = None,
) -> str:
    if prefer_evaluation_context is None:
        prefer_evaluation_context = _env_bool("EVAL_USE_MULTIMODAL_PAPER_CONTEXT", False)
    evaluation_path = str(graph.get("evaluation_paper_text_path") or "").strip()
    raw_path = evaluation_path if prefer_evaluation_context else ""
    if not raw_path:
        raw_path = str(graph.get("paper_text_path", "")).strip()
    if not raw_path:
        raise FileNotFoundError(
            "master_graph.json must include evaluation_paper_text_path or paper_text_path for full-paper evaluation context."
        )

    paper_path = _resolve_paper_path(Path(raw_path), base_dir)
    if raw_path == evaluation_path and paper_path.is_file():
        text = paper_path.read_text(encoding="utf-8")
    elif paper_path.is_dir():
        text = load_paper_text_from_dir(paper_path)
    elif paper_path.is_file():
        text = load_paper_text(paper_path)
    else:
        raise FileNotFoundError(f"Paper text path is neither a file nor a directory: {paper_path}")

    limit = _env_nonnegative_int(limit_env, 0)
    return text[:limit] if limit > 0 else text


def _resolve_paper_path(path: Path, base_dir: Path) -> Path:
    if path.is_absolute():
        return path
    candidates = [
        base_dir / path,
        base_dir.parent / path,
        Path.cwd() / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _env_nonnegative_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a non-negative integer, got {raw!r}.") from exc
    if value < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {value}.")
    return value


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}
