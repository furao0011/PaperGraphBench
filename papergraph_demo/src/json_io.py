from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path


def write_json_atomic(path: Path, payload: dict | list, *, attempts: int = 8, delay_s: float = 0.2) -> None:
    data = json.dumps(payload, ensure_ascii=False, indent=2)
    last_error: OSError | None = None
    for attempt in range(attempts):
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.parent / f".pgjson-{os.getpid()}-{uuid.uuid4().hex[:12]}.tmp"
        try:
            tmp.write_text(data, encoding="utf-8")
            os.replace(tmp, path)
            return
        except (PermissionError, FileNotFoundError) as exc:
            last_error = exc
            try:
                tmp.unlink()
            except OSError:
                pass
            time.sleep(delay_s * (attempt + 1))
    if last_error is not None:
        raise last_error
    raise RuntimeError(f"Failed to write JSON atomically: {path}")
