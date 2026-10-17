from __future__ import annotations

import math
import os
import re
import struct
import tempfile
import zlib
from pathlib import Path

from src.model_client import ModelConfig, OpenAICompatClient
from src.preflight import validate_config


_TIMEOUT_S = 20
_ALIASES = {
    "ocr": "parse_pdf",
    "pdf": "parse_pdf",
    "graph": "build_graph",
    "build": "build_graph",
    "questions": "generate_questions",
    "question": "generate_questions",
    "eval": "evaluate",
    "evaluation": "evaluate",
}


def _redact(message: str) -> str:
    values = {
        value
        for name, value in os.environ.items()
        if value and (name == "API_KEY" or name.endswith("_API_KEY") or name.endswith("_TOKEN"))
    }
    for value in sorted(values, key=len, reverse=True):
        message = message.replace(value, "[REDACTED]")
    return re.sub(r"(?i)(Bearer\s+|token\s+)[^\s,;]+", r"\1[REDACTED]", message)


def _client(key: str, base: str, model: str, *, embedding: bool = False) -> OpenAICompatClient:
    config = ModelConfig(
        api_key=key,
        base_url=base,
        llm_model=model,
        embed_api_key=key if embedding else "",
        embed_base_url=base if embedding else "",
        embed_model=model if embedding else "",
        timeout_s=_TIMEOUT_S,
        max_retries=0,
    )
    client = OpenAICompatClient(config)
    # The production client normally honors timeout/retry environment settings.
    # A probe is intentionally bounded to one request per endpoint.
    client.cfg.timeout_s = _TIMEOUT_S
    client.cfg.max_retries = 0
    return client


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    body = kind + payload
    return struct.pack(">I", len(payload)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)


def _tiny_png() -> bytes:
    """A valid 64x64 white PNG assembled with the standard library."""
    width = height = 64
    pixels = (b"\x00" + b"\xff\xff\xff" * width) * height
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + _png_chunk(b"IHDR", header) + _png_chunk(b"IDAT", zlib.compress(pixels)) + _png_chunk(b"IEND", b"")


def _probe_json(client: OpenAICompatClient, *, image_path: str | None = None) -> None:
    system = 'Return only a JSON object with one field: {"ok": true}.'
    user = "This is a minimal PaperGraphBench API connectivity and JSON-format probe."
    if image_path:
        result = client.chat_json_with_images(system, user, [image_path], timeout_s=_TIMEOUT_S)
    else:
        result = client.chat_json(system, user, timeout_s=_TIMEOUT_S)
    if not isinstance(result, dict) or result.get("ok") is not True:
        raise ValueError("Response did not contain the expected JSON object with ok=true.")


def _probe_embedding(client: OpenAICompatClient) -> None:
    vectors = client.embed_texts(["PaperGraphBench connectivity probe"], timeout_s=_TIMEOUT_S)
    if len(vectors) != 1 or not vectors[0] or not all(
        isinstance(item, (int, float)) and math.isfinite(float(item)) for item in vectors[0]
    ):
        raise ValueError("Embedding response did not contain one finite numeric vector.")


def _probe_target(client: OpenAICompatClient) -> None:
    answer = client.chat_text(
        "Answer with a short plain-text acknowledgement.",
        "PaperGraphBench connectivity probe.",
        timeout_s=_TIMEOUT_S,
    )
    if not answer.strip():
        raise ValueError("Target chat response was empty.")


def probe_apis(stages: list[str]) -> list[str]:
    """Explicit, low-cost API probe; never called by normal startup.

    A caller must opt in (for example via --probe-apis). OCR is deliberately
    not sent a fabricated PDF; it receives configuration validation only.
    """
    errors = validate_config(stages)
    if errors:
        return [_redact(f"Configuration failed: {error}") for error in errors]

    selected = {_ALIASES.get(stage.strip().lower(), stage.strip().lower()) for stage in stages}
    messages: list[str] = []

    def attempt(label: str, callback) -> None:
        try:
            callback()
        except Exception as exc:
            messages.append(_redact(f"{label}: FAILED ({type(exc).__name__}): {exc}"))
        else:
            messages.append(f"{label}: OK")

    if "parse_pdf" in selected:
        messages.append("OCR: configuration OK; not probed (requires a real PDF).")
    if selected & {"build_graph", "generate_questions", "evaluate"}:
        judge = _client(os.environ["API_KEY"], os.environ["BASE_URL"], os.environ["LLM_MODEL"])
        attempt("Judge/chat JSON", lambda: _probe_json(judge))
    if "build_graph" in selected or ("evaluate" in selected and os.getenv("CLAIM_VERIFY_ENABLED", "").lower() in {"1", "true", "yes", "on"}):
        embed = _client(
            os.environ["EMBED_API_KEY"],
            os.environ["EMBED_BASE_URL"],
            os.environ["EMBED_MODEL"],
            embedding=True,
        )
        attempt("Embedding", lambda: _probe_embedding(embed))
    if selected & {"build_graph", "generate_questions"}:
        vision = _client(os.environ["VISION_API_KEY"], os.environ["VISION_BASE_URL"], os.environ["VISION_MODEL"])

        def probe_vision() -> None:
            with tempfile.TemporaryDirectory(prefix="papergraph-api-probe-") as directory:
                image_path = Path(directory) / "tiny.png"
                image_path.write_bytes(_tiny_png())
                _probe_json(vision, image_path=str(image_path))

        attempt("Vision/chat JSON+image", probe_vision)
    if "evaluate" in selected:
        target = _client(
            os.environ["EVAL_TARGET_API_KEY"],
            os.environ["EVAL_TARGET_BASE_URL"],
            os.environ["EVAL_TARGET_MODEL"],
        )
        attempt("Target/chat text", lambda: _probe_target(target))
    return messages
