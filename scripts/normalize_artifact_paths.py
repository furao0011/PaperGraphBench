"""Check or migrate filesystem references in published JSON paper packages.

Only path-valued fields are changed. Paper text, captions, IDs, evidence,
questions, and diagnostic counts are never rewritten. No configuration files
or credentials are loaded, and the script makes no network requests.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path, PureWindowsPath
import re
import sys


REPOSITORY = Path(__file__).resolve().parents[1]
ROOT_NAMES = {"data", "rawPaper"}
PATH_KEYS = {"path", "paths", "file", "files", "directory", "directories"}
PATH_SUFFIXES = ("_path", "_paths", "_file", "_files", "_directory", "_directories")


def is_path_key(key: str) -> bool:
    return key in PATH_KEYS or key.endswith(PATH_SUFFIXES)


def normalize_path(value: str, paper_id: str) -> str:
    """Convert explicit data/rawPaper references without guessing their origin."""
    if not value:
        return value
    text = value.replace("\\", "/")
    if text.startswith(("https://", "http://", "data:")):
        return value
    windows = PureWindowsPath(value)
    absolute = text.startswith("/") or windows.is_absolute()
    if windows.drive and not windows.is_absolute():
        raise ValueError("Drive-relative filesystem references are not portable.")
    if text.startswith("file:"):
        raise ValueError("Filesystem URI paths require explicit migration.")
    parts = text.split("/")
    if ".." in parts:
        raise ValueError("Filesystem reference contains a parent-directory traversal.")
    if absolute:
        markers = [i for i, part in enumerate(parts) if part in ROOT_NAMES]
        if len(markers) != 1:
            raise ValueError("Absolute filesystem reference has no unique data/rawPaper root.")
        parts = parts[markers[0]:]
        if len(parts) < 2 or parts[1] != paper_id:
            raise ValueError("Absolute filesystem reference belongs to another paper.")
    elif parts[0] in ROOT_NAMES:
        # Shared data/graphs definitions are not paper-specific references.
        if len(parts) < 2 or (parts[1] != paper_id and parts[:2] != ["data", "graphs"]):
            raise ValueError("Repository filesystem reference belongs to another paper.")
    else:
        # Plain filenames and semantic strings are not reinterpreted as paths.
        return value
    if any(part in {"", "."} for part in parts):
        raise ValueError("Filesystem reference contains an empty or dot component.")
    return "/".join(parts)


def normalize_document(document: dict, paper_id: str, *, master_graph: bool = False) -> tuple[dict, list[str]]:
    if not isinstance(document, dict):
        raise ValueError("A public JSON artifact must be an object.")
    if document.get("paper_id", paper_id) != paper_id:
        raise ValueError("Artifact paper_id differs from its dataset directory.")
    result = copy.deepcopy(document)
    changes: list[str] = []

    def walk(node: object, pointer: str = "", path_field: bool = False) -> object:
        if isinstance(node, str) and path_field:
            try:
                normalized = normalize_path(node, paper_id)
            except ValueError as exc:
                raise ValueError(f"{pointer}: {exc}") from exc
            if normalized != node:
                changes.append(pointer)
            return normalized
        if isinstance(node, list):
            return [walk(value, f"{pointer}/{index}", path_field) for index, value in enumerate(node)]
        if isinstance(node, dict):
            return {key: walk(value, f"{pointer}/{key}", is_path_key(key)) for key, value in node.items()}
        return node

    result = walk(result)
    if master_graph:
        storage = result.setdefault("storage", {})
        if not isinstance(storage, dict):
            raise ValueError("master_graph.storage must be an object.")
        if storage.get("path_base", "repository") != "repository":
            raise ValueError("master_graph.storage.path_base is not repository; migrate its layout explicitly.")
        if "path_base" not in storage:
            storage["path_base"] = "repository"
            changes.append("/storage/path_base")
    return result, changes


def planned_changes(data_root: Path, paper_ids: list[str] | None = None) -> list[tuple[Path, dict, list[str]]]:
    if not data_root.is_dir():
        raise ValueError("Dataset root is not a directory.")
    names = paper_ids or sorted(path.name for path in data_root.iterdir() if path.is_dir() and not path.name.startswith("."))
    planned = []
    for paper_id in names:
        if not paper_id or paper_id in {".", ".."} or re.search(r"[\\/]", paper_id):
            raise ValueError("paper_id must be a single directory name.")
        directory = data_root / paper_id
        if not directory.is_dir():
            raise ValueError(f"Paper directory is absent: {paper_id}")
        for path in sorted(directory.rglob("*.json")):
            # This is a published-artifact migration, never a private-cache scan.
            if any(part.startswith(".") or part == "cache" for part in path.relative_to(directory).parts):
                continue
            try:
                document = json.loads(path.read_text(encoding="utf-8"))
                normalized, changes = normalize_document(document, paper_id, master_graph=path.name == "master_graph.json")
            except (ValueError, UnicodeError) as exc:
                raise ValueError(f"{path.relative_to(data_root)}: {exc}") from exc
            if changes:
                planned.append((path, normalized, changes))
    return planned


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=REPOSITORY / "data")
    parser.add_argument("--paper-id", action="append", help="Limit migration to a paper; repeat to select several.")
    parser.add_argument("--write", action="store_true", help="Apply changes; otherwise check only (exit 1 if changes are needed).")
    args = parser.parse_args(argv)
    try:
        # Validate every selected artifact before writing any of them.
        planned = planned_changes(args.data_root.resolve(), args.paper_id)
        for path, document, changes in planned:
            if args.write:
                path.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            action = "normalized" if args.write else "needs normalization"
            print(f"{path.relative_to(args.data_root.resolve())}: {action} ({len(changes)} path/metadata fields)")
        print(f"{len(planned)} artifact(s) {'updated' if args.write else 'need normalization'}.")
        return 0 if args.write or not planned else 1
    except (OSError, ValueError) as exc:
        print(f"Path migration failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
