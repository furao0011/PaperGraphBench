from __future__ import annotations

import os
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from typing import Iterator


def enabled() -> bool:
    return os.getenv("PAPERGRAPH_PROGRESS", "false").lower() in {"1", "true", "yes", "on"}


def log(message: str, **fields: object) -> None:
    if not enabled():
        return
    ts = datetime.now().strftime("%H:%M:%S")
    suffix = ""
    if fields:
        pairs = " ".join(f"{key}={value}" for key, value in fields.items())
        suffix = f" | {pairs}"
    print(f"[papergraph {ts}] {message}{suffix}", file=sys.stderr, flush=True)


@contextmanager
def span(message: str, **fields: object) -> Iterator[None]:
    if not enabled():
        yield
        return
    start = time.perf_counter()
    log(f"{message} started", **fields)
    try:
        yield
    except Exception as exc:
        elapsed = f"{time.perf_counter() - start:.1f}s"
        log(f"{message} failed", elapsed=elapsed, error=f"{type(exc).__name__}: {exc}")
        raise
    else:
        elapsed = f"{time.perf_counter() - start:.1f}s"
        log(f"{message} finished", elapsed=elapsed)
