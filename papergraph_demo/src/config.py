from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Settings:
    api_key: str = ""
    base_url: str = ""
    llm_model: str = ""
    embed_base_url: str = ""
    embed_model: str = ""
    working_dir: str = "./working"
    embed_dim: int = 1024
    embed_max_tokens: int = 8192
    use_online_kc_extract: bool = False


def load_dotenv(dotenv_path: Path) -> None:
    if not dotenv_path.exists():
        return
    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        os.environ.setdefault(key, value)


def _to_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def load_settings(project_root: Path) -> Settings:
    load_dotenv(project_root / ".env")
    def _to_int(name: str, default: int) -> int:
        raw = os.getenv(name)
        if raw is None or raw.strip() == "":
            return default
        try:
            return int(raw)
        except ValueError:
            return default
    return Settings(
        api_key=os.getenv("API_KEY", ""),
        base_url=os.getenv("BASE_URL", ""),
        llm_model=os.getenv("LLM_MODEL", ""),
        embed_base_url=os.getenv("EMBED_BASE_URL", ""),
        embed_model=os.getenv("EMBED_MODEL", ""),
        working_dir=os.getenv("WORKING_DIR", "./working"),
        embed_dim=_to_int("EMBED_DIM", 1024),
        embed_max_tokens=_to_int("EMBED_MAX_TOKENS", 8192),
        use_online_kc_extract=_to_bool(os.getenv("USE_ONLINE_KC_EXTRACT"), default=False),
    )
