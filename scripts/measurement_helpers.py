"""File freezing and child execution for the formal measurement entry point."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def input_files(root: Path) -> list[Path]:
    """Enumerate published inputs, never checkpoints, hidden files or bytecode."""
    files = []
    for directory, names, filenames in os.walk(root, followlinks=False):
        names[:] = sorted(name for name in names if name not in {"cache", "__pycache__"} and not name.startswith("."))
        for name in sorted(filenames):
            path = Path(directory) / name
            if not name.startswith(".") and path.suffix != ".env" and not path.is_symlink() and path.is_file():
                files.append(path)
    return files


def copy_sources(sources: list[Path], destination: Path) -> None:
    for source in sources:
        for path in input_files(source):
            target = destination / source.name / path.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)


def frozen_hashes(session: Path) -> dict[str, str]:
    result = {}
    for name in ("dataset", "rawPaper"):
        for path in input_files(session / name):
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            result[path.relative_to(session).as_posix()] = digest.hexdigest()
    return result


def assert_frozen(session: Path, expected: dict[str, str]) -> None:
    actual = frozen_hashes(session)
    if actual != expected:
        changed = sorted(key for key in set(expected) | set(actual) if expected.get(key) != actual.get(key))
        raise RuntimeError("Frozen measurement inputs changed: " + ", ".join(changed[:10]))


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def source_revision(project_root: Path) -> dict:
    if not (project_root / ".git").exists():
        return {"commit": None, "status": "unavailable: source archive without Git metadata"}
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=project_root, capture_output=True, text=True, check=True)
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=project_root,
                               capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError("Cannot read repository revision; fix Git metadata before measuring this checkout.") from exc
    return {"commit": result.stdout.strip(), "tracked_changes": bool(dirty.stdout.strip())}


def run_stage(argv: list[str], env: dict[str, str], log: Path, *, project_root: Path) -> None:
    """Stream the current child console into one session log; fail immediately."""
    command = [sys.executable, str(project_root / "run.py"), *argv]
    env = {**env, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
    with log.open("a", encoding="utf-8") as stream:
        stream.write("\n[measurement] " + " ".join(argv) + "\n")
        stream.flush()
        with subprocess.Popen(command, cwd=project_root, env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace") as child:
            try:
                for line in child.stdout:
                    print(line, end="", flush=True)
                    stream.write(line)
                    stream.flush()
            except BaseException:
                child.terminate()
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
                raise
            code = child.wait()
    if code:
        raise RuntimeError(f"Stage {argv[0]} exited {code}; inspect {log} and this session's logs/ directory.")
