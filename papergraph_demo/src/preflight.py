from __future__ import annotations

import os
import math
from pathlib import Path
from urllib.parse import urlsplit

from src.config import load_environment


_STAGE_ALIASES = {
    "ocr": "parse_pdf",
    "pdf": "parse_pdf",
    "graph": "build_graph",
    "build": "build_graph",
    "questions": "generate_questions",
    "question": "generate_questions",
    "eval": "evaluate",
    "evaluation": "evaluate",
}
_STAGES = {"parse_pdf", "build_graph", "generate_questions", "evaluate"}
_BOOL_KEYS = {
    "USE_ONLINE_KC_EXTRACT",
    "MULTIMODAL_ENABLED",
    "MULTIMODAL_KC_ENABLED",
    "USE_ONLINE_EVAL",
    "PAPERGRAPH_PROGRESS",
    "ALLOW_OFFLINE_FALLBACK",
    "ALLOW_MOCK_EVAL",
    "CLAIM_VERIFY_ENABLED",
    "PAPERGRAPH_RESUME",
    "PAPERGRAPH_RESTART",
    "BUILD_GRAPH_RESUME",
    "BUILD_GRAPH_RESTART",
    "QUESTION_RESUME",
    "QUESTION_RESTART",
    "EVAL_RESUME",
    "EVAL_RESTART",
}
_INT_MINIMUMS = {
    "EMBED_DIM": 1,
    "EMBED_MAX_TOKENS": 1,
    "EMBED_BATCH_SIZE": 1,
    "OCR_BATCH_WORKERS": 1,
    "DATASET_BATCH_WORKERS": 1,
    "EVAL_BATCH_WORKERS": 1,
    "PADDLE_OCR_TIMEOUT_S": 1,
    "PADDLE_OCR_MAX_RETRIES": 0,
    "PADDLE_OCR_DOWNLOAD_MAX_RETRIES": 0,
    "LLM_TIMEOUT_S": 1,
    "LLM_MAX_RETRIES": 0,
    "PAPERGRAPH_LLM_TIMEOUT_S": 1,
    "PAPERGRAPH_LLM_MAX_RETRIES": 0,
    "VISION_TIMEOUT_S": 1,
    "VISION_MAX_RETRIES": 0,
    "EVAL_MAX_TURNS": 0,
    "CHALLENGE_SOLVER_COUNT": 1,
    "MULTIMODAL_CHALLENGE_SOLVER_COUNT": 1,
    "THREAD_CHALLENGE_SOLVER_COUNT": 1,
}
_FLOAT_MINIMUMS = {
    "PADDLE_OCR_RETRY_SLEEP_S": 0.0,
    "LLM_RETRY_SLEEP_S": 0.0,
    "PAPERGRAPH_LLM_RETRY_SLEEP_S": 0.0,
    "EVAL_TARGET_TEMPERATURE": 0.0,
}
_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}
_API_BASE_URL_KEYS = {"BASE_URL", "VISION_BASE_URL", "EMBED_BASE_URL", "EVAL_TARGET_BASE_URL"}


def _enabled(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in _TRUE


def validate_config(stages: list[str], project_root: Path | None = None) -> list[str]:
    """Return all local configuration errors without making network requests.

    Stages: parse_pdf, build_graph, generate_questions, evaluate. Short aliases
    ocr, graph, questions, eval are accepted. No secret values are reported.
    """
    root = project_root or Path(__file__).resolve().parents[2]
    errors: list[str] = []
    try:
        load_environment(root)
    except (OSError, ValueError) as exc:
        errors.append(str(exc))

    selected = {_STAGE_ALIASES.get(stage.strip().lower(), stage.strip().lower()) for stage in stages}
    if not selected:
        errors.append("At least one preflight stage is required.")
    for stage in sorted(selected - _STAGES):
        errors.append(f"Unknown preflight stage: {stage}")
    selected &= _STAGES

    for name in sorted(_BOOL_KEYS):
        raw = os.getenv(name, "").strip().lower()
        if raw and raw not in _TRUE | _FALSE:
            errors.append(f"{name} must be true or false.")
    for name, minimum in _INT_MINIMUMS.items():
        raw = os.getenv(name, "").strip()
        if not raw:
            continue
        try:
            value = int(raw)
        except ValueError:
            errors.append(f"{name} must be an integer >= {minimum}.")
            continue
        if value < minimum or (name == "EMBED_BATCH_SIZE" and value > 10):
            bound = "between 1 and 10" if name == "EMBED_BATCH_SIZE" else f">= {minimum}"
            errors.append(f"{name} must be {bound}.")
    for name, minimum in _FLOAT_MINIMUMS.items():
        raw = os.getenv(name, "").strip()
        if not raw:
            continue
        try:
            value = float(raw)
        except ValueError:
            errors.append(f"{name} must be a number >= {minimum}.")
            continue
        if not math.isfinite(value) or value < minimum:
            errors.append(f"{name} must be >= {minimum}.")

    required: set[str] = set()
    if "parse_pdf" in selected:
        required.update({"PADDLE_OCR_API_URL", "PADDLE_OCR_TOKEN"})
    if selected & {"build_graph", "generate_questions", "evaluate"}:
        required.update({"API_KEY", "BASE_URL", "LLM_MODEL"})
    if "build_graph" in selected or ("evaluate" in selected and _enabled("CLAIM_VERIFY_ENABLED")):
        required.update({"EMBED_API_KEY", "EMBED_BASE_URL", "EMBED_MODEL"})
    if selected & {"build_graph", "generate_questions"} and _enabled("MULTIMODAL_ENABLED"):
        required.update({"VISION_API_KEY", "VISION_BASE_URL", "VISION_MODEL"})
    if "evaluate" in selected:
        required.update({"EVAL_TARGET_API_KEY", "EVAL_TARGET_BASE_URL", "EVAL_TARGET_MODEL"})
    for name in sorted(required):
        if not os.getenv(name, "").strip():
            errors.append(f"Missing required setting: {name}.")
    for name in sorted((_API_BASE_URL_KEYS | {"PADDLE_OCR_API_URL"}) & required):
        raw = os.getenv(name, "").strip()
        if not raw:
            continue
        url = urlsplit(raw)
        if url.scheme.lower() not in {"http", "https"} or not url.netloc or url.query or url.fragment:
            errors.append(f"{name} must be an HTTP(S) URL without a query or fragment.")
            continue
        if name in _API_BASE_URL_KEYS:
            parts = [part.lower() for part in url.path.split("/") if part]
            if parts[-2:] == ["chat", "completions"] or parts[-1:] == ["embeddings"]:
                errors.append(f"{name} must be an API base URL, not a /chat/completions or /embeddings endpoint.")
            if any(left == right and left.startswith("v") and left[1:].isdigit() for left, right in zip(parts, parts[1:])):
                errors.append(f"{name} contains a duplicated API version path segment.")

    if selected & {"build_graph", "generate_questions", "evaluate"}:
        if _enabled("ALLOW_OFFLINE_FALLBACK"):
            errors.append("ALLOW_OFFLINE_FALLBACK must be false for the formal pipeline.")
        if _enabled("ALLOW_MOCK_EVAL"):
            errors.append("ALLOW_MOCK_EVAL must be false for the formal pipeline.")
    if "build_graph" in selected and not _enabled("USE_ONLINE_KC_EXTRACT"):
        errors.append("USE_ONLINE_KC_EXTRACT must be true for online graph construction.")
    if selected & {"build_graph", "generate_questions"} and not _enabled("MULTIMODAL_ENABLED"):
        errors.append("MULTIMODAL_ENABLED must be true for the formal multimodal pipeline.")
    if "build_graph" in selected and not _enabled("MULTIMODAL_KC_ENABLED"):
        errors.append("MULTIMODAL_KC_ENABLED must be true for the formal multimodal pipeline.")
    if "evaluate" in selected and not _enabled("USE_ONLINE_EVAL"):
        errors.append("USE_ONLINE_EVAL must be true for formal evaluation.")
    return errors
