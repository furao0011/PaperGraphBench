from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


DEFAULT_PATHS = {
    "PAPERGRAPH_DATA_ROOT": "data",
    "RAW_PAPER_ROOT": "rawPaper",
    "EVAL_RESULT_ROOT": "outputs/evaluation",
}

DEFAULT_FLAGS = {
    "USE_ONLINE_KC_EXTRACT": "true",
    "MULTIMODAL_ENABLED": "true",
    "MULTIMODAL_KC_ENABLED": "true",
    "USE_ONLINE_EVAL": "true",
    "PAPERGRAPH_PROGRESS": "true",
    "ALLOW_OFFLINE_FALLBACK": "false",
    "ALLOW_MOCK_EVAL": "false",
    "CLAIM_VERIFY_ENABLED": "false",
}


@dataclass
class Settings:
    api_key: str = ""
    embed_api_key: str = ""
    base_url: str = ""
    llm_model: str = ""
    embed_base_url: str = ""
    embed_model: str = ""
    vision_api_key: str = ""
    vision_base_url: str = ""
    vision_model: str = ""
    working_dir: str = "./working"
    embed_dim: int = 1024
    embed_max_tokens: int = 8192
    use_online_kc_extract: bool = True
    data_root: Path | None = None
    raw_paper_root: Path | None = None
    eval_result_root: Path | None = None


def _config_path(project_root: Path, config_path: Path | str | None = None) -> Path:
    configured = config_path or os.getenv("PAPERGRAPH_CONFIG", "").strip() or "config.env"
    path = Path(configured).expanduser()
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def _read_config(path: Path, *, required: bool) -> None:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"PAPERGRAPH_CONFIG file does not exist: {path}")
        return
    if not path.is_file():
        raise ValueError(f"Configuration path is not a file: {path}")
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"Invalid configuration line {line_number} in {path.name}: expected KEY=VALUE")
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or not key.replace("_", "").isalnum() or key[0].isdigit():
            raise ValueError(f"Invalid configuration key at line {line_number} in {path.name}")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ.setdefault(key, value)


def resolve_project_path(project_root: Path, value: str | Path) -> Path:
    """Interpret output roots relative to the repository root."""
    root = project_root.expanduser().resolve()
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def load_environment(project_root: Path, config_path: Path | str | None = None) -> Path:
    """Load only config.env (or PAPERGRAPH_CONFIG); existing environment wins.

    The default file is optional so deployments can use process variables.
    This function never implicitly reads .env.
    """
    root = project_root.expanduser().resolve()
    path = _config_path(root, config_path)
    explicit = config_path is not None or bool(os.getenv("PAPERGRAPH_CONFIG", "").strip())
    _read_config(path, required=explicit)
    for name, value in DEFAULT_FLAGS.items():
        os.environ.setdefault(name, value)
    for name, value in DEFAULT_PATHS.items():
        configured = os.getenv(name, "").strip() or value
        os.environ[name] = str(resolve_project_path(root, configured))
    return path


def _to_bool(value: str | None, default: bool = False) -> bool:
    if value is None or not value.strip():
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def load_settings(project_root: Path) -> Settings:
    load_environment(project_root)

    def _to_int(name: str, default: int) -> int:
        raw = os.getenv(name)
        if raw is None or raw.strip() == "":
            return default
        try:
            return int(raw)
        except ValueError as exc:
            raise ValueError(f"{name} must be an integer") from exc

    return Settings(
        api_key=os.getenv("API_KEY", ""),
        embed_api_key=os.getenv("EMBED_API_KEY", ""),
        base_url=os.getenv("BASE_URL", ""),
        llm_model=os.getenv("LLM_MODEL", ""),
        embed_base_url=os.getenv("EMBED_BASE_URL", ""),
        embed_model=os.getenv("EMBED_MODEL", ""),
        vision_api_key=os.getenv("VISION_API_KEY", ""),
        vision_base_url=os.getenv("VISION_BASE_URL", ""),
        vision_model=os.getenv("VISION_MODEL", ""),
        working_dir=os.getenv("WORKING_DIR", "./working"),
        embed_dim=_to_int("EMBED_DIM", 1024),
        embed_max_tokens=_to_int("EMBED_MAX_TOKENS", 8192),
        use_online_kc_extract=_to_bool(os.getenv("USE_ONLINE_KC_EXTRACT"), default=True),
        data_root=Path(os.environ["PAPERGRAPH_DATA_ROOT"]),
        raw_paper_root=Path(os.environ["RAW_PAPER_ROOT"]),
        eval_result_root=Path(os.environ["EVAL_RESULT_ROOT"]),
    )
