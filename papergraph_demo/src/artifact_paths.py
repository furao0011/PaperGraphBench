"""Explicit storage roots for portable graph and OCR asset references.

New artifacts use ``storage.path_base=repository``. Legacy ``data/...``
references belong to the configured dataset root, not the working directory.
Publisher-specific absolute OCR paths are relocated only through their declared
``rawPaper/<paper_id>`` suffix; arbitrary missing paths are never searched for.
"""
from __future__ import annotations

import os
from pathlib import Path, PureWindowsPath


def repository_root(base_dir: Path) -> Path:
    base = Path(base_dir).resolve()
    return base.parent if base.name == "papergraph_demo" else base


def _configured_root(name: str, default: str, base_dir: Path) -> Path:
    root = repository_root(base_dir)
    value = os.getenv(name, "").strip()
    path = Path(value).expanduser() if value else Path(default)
    return (path if path.is_absolute() else root / path).resolve()


def configured_data_root(base_dir: Path) -> Path:
    return _configured_root("PAPERGRAPH_DATA_ROOT", "data", base_dir)


def configured_raw_root(base_dir: Path) -> Path:
    return _configured_root("RAW_PAPER_ROOT", "rawPaper", base_dir)


def _path(raw_path: str | Path) -> Path:
    text = str(raw_path).strip()
    if not text:
        raise ValueError("Artifact path must not be empty.")
    return Path(text.replace("\\", "/"))


def resolve_graph_path(raw_path: str | Path, base_dir: Path, graph: dict | None = None) -> Path:
    path = _path(raw_path)
    if path.is_absolute():
        return path.resolve()
    if PureWindowsPath(str(raw_path)).is_absolute():
        raise ValueError(f"Artifact uses an absolute path from another platform: {raw_path}")
    root = repository_root(base_dir)
    storage = (graph or {}).get("storage")
    if storage is not None:
        if not isinstance(storage, dict):
            raise ValueError("graph.storage must be an object.")
        path_base = storage.get("path_base")
        if path_base == "repository":
            return (root / path).resolve()
        if path_base == "papergraph_demo":
            return (root / "papergraph_demo" / path).resolve()
        raise ValueError(f"Unsupported graph.storage.path_base: {path_base!r}")
    if path.parts[0] == "data":
        return configured_data_root(base_dir).joinpath(*path.parts[1:]).resolve()
    if path.parts[0] in {"papergraph_demo", "rawPaper"}:
        return (root / path).resolve()
    raise ValueError(
        f"Ambiguous legacy artifact path {raw_path!r}; declare graph.storage.path_base."
    )


def resolve_asset_path(raw_path: str | Path, base_dir: Path, graph: dict | None = None) -> Path:
    path = _path(raw_path)
    # Published packages contain the author's Windows absolute rawPaper paths.
    # The named OCR root and paper id provide an unambiguous relocation contract.
    parts = path.parts
    markers = [i for i, part in enumerate(parts) if part == "rawPaper"]
    if markers:
        if len(markers) != 1 or len(parts) <= markers[0] + 2:
            raise ValueError(f"Invalid rawPaper asset reference: {raw_path}")
        suffix = parts[markers[0] + 1 :]
        paper_id = str((graph or {}).get("paper_id") or "")
        if paper_id and suffix[0] != paper_id:
            raise ValueError(f"Asset paper id {suffix[0]!r} differs from graph paper id {paper_id!r}.")
        if any(part == ".." for part in suffix):
            raise ValueError(f"Asset reference escapes its OCR paper directory: {raw_path}")
        return configured_raw_root(base_dir).joinpath(*suffix).resolve()
    return resolve_graph_path(raw_path, base_dir, graph)
