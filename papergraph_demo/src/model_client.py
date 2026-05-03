from __future__ import annotations

import json
import os
import time
import urllib.request
import urllib.error
from dataclasses import dataclass
from typing import Any

from src.progress import log


@dataclass
class ModelConfig:
    api_key: str
    base_url: str
    llm_model: str
    embed_api_key: str = ""
    embed_base_url: str = ""
    embed_model: str = ""
    timeout_s: int = 300
    max_retries: int = 2
    retry_sleep_s: float = 5.0


class OpenAICompatClient:
    def __init__(self, cfg: ModelConfig) -> None:
        self.cfg = ModelConfig(
            api_key=cfg.api_key,
            base_url=cfg.base_url,
            llm_model=cfg.llm_model,
            embed_api_key=cfg.embed_api_key or os.getenv("EMBED_API_KEY", ""),
            embed_base_url=cfg.embed_base_url or os.getenv("EMBED_BASE_URL", ""),
            embed_model=cfg.embed_model or os.getenv("EMBED_MODEL", ""),
            timeout_s=_env_int("PAPERGRAPH_LLM_TIMEOUT_S", _env_int("LLM_TIMEOUT_S", cfg.timeout_s)),
            max_retries=_env_nonnegative_int(
                "PAPERGRAPH_LLM_MAX_RETRIES",
                _env_nonnegative_int("LLM_MAX_RETRIES", cfg.max_retries),
            ),
            retry_sleep_s=_env_float(
                "PAPERGRAPH_LLM_RETRY_SLEEP_S",
                _env_float("LLM_RETRY_SLEEP_S", cfg.retry_sleep_s),
            ),
        )

    def is_ready(self) -> bool:
        return bool(self.cfg.api_key and self.cfg.base_url and self.cfg.llm_model)

    def embeddings_ready(self) -> bool:
        return bool(self.cfg.embed_api_key and self.cfg.embed_base_url and self.cfg.embed_model)

    def chat_json(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
        timeout_s: int | None = None,
    ) -> dict:
        if not self.is_ready():
            raise RuntimeError("Model client is not configured. Check API_KEY/BASE_URL/LLM_MODEL.")

        url = self.cfg.base_url.rstrip("/") + "/chat/completions"
        payload = {
            "model": self.cfg.llm_model,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "response_format": {"type": "json_object"},
        }
        body = self._post_json(url, payload, timeout_s)
        result = json.loads(body)
        content = result["choices"][0]["message"]["content"]
        return json.loads(content)

    def chat_text(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
        timeout_s: int | None = None,
    ) -> str:
        if not self.is_ready():
            raise RuntimeError("Model client is not configured. Check API_KEY/BASE_URL/LLM_MODEL.")

        url = self.cfg.base_url.rstrip("/") + "/chat/completions"
        payload = {
            "model": self.cfg.llm_model,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        body = self._post_json(url, payload, timeout_s)
        result = json.loads(body)
        return result["choices"][0]["message"]["content"]

    def embed_texts(self, texts: list[str], timeout_s: int | None = None) -> list[list[float]]:
        if not self.embeddings_ready():
            raise RuntimeError("Embedding client is not configured. Check EMBED_API_KEY/EMBED_BASE_URL/EMBED_MODEL.")
        if not texts:
            return []
        batch_size = _env_int("EMBED_BATCH_SIZE", 10)
        if batch_size > 10:
            raise ValueError("EMBED_BATCH_SIZE must be <= 10 for DashScope text-embedding-v4.")
        embeddings: list[list[float]] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            embeddings.extend(self._embed_batch(batch, timeout_s))
        return embeddings

    def _embed_batch(self, texts: list[str], timeout_s: int | None = None) -> list[list[float]]:
        url = self.cfg.embed_base_url.rstrip("/") + "/embeddings"
        payload = {
            "model": self.cfg.embed_model,
            "input": texts[0] if len(texts) == 1 else texts,
        }
        body = self._post_json(url, payload, timeout_s, api_key=self.cfg.embed_api_key)
        result = json.loads(body)
        data = result.get("data", [])
        if len(data) != len(texts):
            raise RuntimeError(f"Embedding response count mismatch: expected {len(texts)}, got {len(data)}.")
        ordered = sorted(data, key=lambda item: item.get("index", 0))
        embeddings = [item.get("embedding") for item in ordered]
        if not all(isinstance(vec, list) and vec for vec in embeddings):
            raise RuntimeError("Embedding response contains empty or invalid vectors.")
        return embeddings

    def _post_json(
        self,
        url: str,
        payload: dict[str, Any],
        timeout_s: int | None,
        api_key: str | None = None,
    ) -> str:
        timeout = timeout_s if timeout_s is not None else self.cfg.timeout_s
        attempts = max(1, self.cfg.max_retries + 1)
        last_exc: Exception | None = None
        data = json.dumps(payload).encode("utf-8")
        auth_key = api_key or self.cfg.api_key
        for attempt in range(1, attempts + 1):
            req = urllib.request.Request(
                url=url,
                data=data,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {auth_key}",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    return resp.read().decode("utf-8")
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                last_exc = RuntimeError(f"HTTP {exc.code} {exc.reason}: {detail}")
                if attempt >= attempts:
                    break
                log(
                    "model request retry",
                    attempt=attempt,
                    attempts=attempts,
                    timeout_s=timeout,
                    error=str(last_exc),
                )
                time.sleep(max(0.0, self.cfg.retry_sleep_s))
            except Exception as exc:
                last_exc = exc
                if attempt >= attempts:
                    break
                log(
                    "model request retry",
                    attempt=attempt,
                    attempts=attempts,
                    timeout_s=timeout,
                    error=f"{type(exc).__name__}: {exc}",
                )
                time.sleep(max(0.0, self.cfg.retry_sleep_s))
        assert last_exc is not None
        raise last_exc


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def _env_nonnegative_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value >= 0 else default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if value >= 0 else default
