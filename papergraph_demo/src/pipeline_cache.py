"""Content-addressed completion stamps for the public pipeline.

Stamps contain hashes only, never a dump of process configuration or credentials.
Missing stamps mean an output has not been checked by this pipeline version.
Corrupt stamps are errors rather than permission to reuse an old output.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

from src.artifact_paths import configured_raw_root, resolve_asset_path, resolve_graph_path
from src.json_io import write_json_atomic
from src.pipeline_plans import PROJECT_ROOT, PaperJob, paper_layout, require_json


_VERSION = 1
_PREFIXES = {
    "parse_pdf": ("PADDLE_OCR_", "PAPERGRAPH_OCR_"),
    "build_graph": ("ACTIVE_", "KC_", "EXTRACTION_", "EDGE_", "GRAPH_", "UNIT_", "ONLINE_", "MACRO_", "RUBRIC_", "REASONING_", "MULTIMODAL_", "THREAD_", "USE_", "EMBED_", "VISION_", "LLM_", "PAPERGRAPH_", "EVAL_USE_"),
    "generate_questions": ("CHALLENGE_", "THREAD_", "QUESTION_", "MULTIMODAL_", "LLM_", "VISION_", "PAPERGRAPH_", "EVAL_USE_"),
    "evaluate": ("EVAL_", "CLAIM_", "THREAD_", "CHALLENGE_", "MAX_", "REVIEW_", "NO_GAMEOVER_", "MULTIMODAL_", "LLM_", "PAPERGRAPH_", "EMBED_", "USE_"),
}
_SECRET = re.compile(r"(?:^|_)(?:API_KEY|TOKEN|SECRET|PASSWORD|CREDENTIALS?|AUTHORIZATION)(?:_|$)")
_RUNTIME = re.compile(r"(?:_RESUME|_RESTART|_WORKERS|_TIMEOUT_S|_MAX_RETRIES|_RETRY_SLEEP_S|_PATH|_DIR|_ROOT|_FILE)$")
_RUN_KEYS = {"PAPERGRAPH_PROGRESS", "PAPERGRAPH_CONFIG", "PAPERGRAPH_OCR_OVERWRITE", "EVAL_BATCH_MODELS", "PAPER_ID"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"Pipeline signature input does not exist: {path}")
    files = []
    for directory, names, filenames in os.walk(path, followlinks=False):
        names[:] = sorted(name for name in names if name != "cache" and not name.startswith("."))
        for name in sorted(filenames):
            if not name.startswith("."):
                file = Path(directory) / name
                if file.is_file():
                    files.append(file)
    return files


def _configuration(stage: str, model: str | None) -> dict[str, str]:
    selected = {}
    for name, value in os.environ.items():
        if _SECRET.search(name) or _RUNTIME.search(name) or name in _RUN_KEYS:
            continue
        if name.startswith(("EVAL_TARGET_API_KEY_", "EVAL_TARGET_BASE_URL_")):
            continue
        if name.startswith(_PREFIXES[stage]) or (stage != "parse_pdf" and name in {"BASE_URL", "ALLOW_OFFLINE_FALLBACK", "ALLOW_MOCK_EVAL"}) or (stage == "generate_questions" and name == "EVAL_PAPER_CHAR_LIMIT"):
            selected[name] = value
    if stage == "evaluate":
        if not model:
            raise ValueError("An evaluation signature requires a target model.")
        suffix = re.sub(r"[^A-Za-z0-9]+", "_", model).strip("_").upper()
        selected["EVAL_TARGET_MODEL"] = model
        selected["EVAL_TARGET_BASE_URL"] = os.getenv(f"EVAL_TARGET_BASE_URL_{suffix}", "").strip() or os.getenv("EVAL_TARGET_BASE_URL", "")
    return selected


def _image_references(value: Any):
    if isinstance(value, dict):
        if value.get("type") == "image" and value.get("path"):
            yield value["path"]
        for name, child in value.items():
            if name == "image_paths":
                if not isinstance(child, list):
                    raise ValueError("image_paths must be a list when hashing pipeline inputs.")
                yield from child
            elif name == "resolved_image_path" and child:
                yield child
            else:
                yield from _image_references(child)
    elif isinstance(value, list):
        for child in value:
            yield from _image_references(child)


def build_stage_signature(job: PaperJob, stage: str, model: str | None = None) -> str:
    if stage not in _PREFIXES:
        raise ValueError(f"Unknown pipeline stage: {stage}")
    base_dir = PROJECT_ROOT / "papergraph_demo"
    inputs: set[Path] = set()

    def include(path: Path) -> None:
        inputs.update(file.resolve() for file in _files(path))

    if stage == "parse_pdf":
        if job.pdf is None:
            raise ValueError("An OCR signature requires the source PDF.")
        include(job.pdf)
    elif stage == "build_graph":
        source = job.source if job.source is not None else configured_raw_root(base_dir) / job.paper_id
        include(source)
        if source.is_file():
            from src.paper_block_parser import load_paper_bundle_from_file

            for reference in _image_references(load_paper_bundle_from_file(source)):
                include(Path(reference))
    else:
        layout = paper_layout(job.paper_id)
        graph_path = layout.final("master_graph")
        graph = require_json(graph_path, job.paper_id)
        include(graph_path)
        if not graph.get("paper_text_path"):
            raise ValueError("Graph must declare paper_text_path for pipeline signatures.")
        for name in ("paper_text_path", "evaluation_paper_text_path", "multimodal_assets_path", "multimodal_asset_explanations_path"):
            raw_path = graph.get(name)
            if not raw_path:
                continue
            path = resolve_graph_path(raw_path, base_dir, graph)
            include(path)
            if name in {"multimodal_assets_path", "multimodal_asset_explanations_path"}:
                payload = require_json(path, job.paper_id)
                for reference in _image_references(payload):
                    include(resolve_asset_path(reference, base_dir, graph))
        diagnostics = graph.get("diagnostics") or {}
        source = diagnostics.get("master_graph_kc_source") or (diagnostics.get("graph_signature") or {}).get("master_graph_kc_source")
        if stage == "evaluate" and graph.get("kc_bank_path") and source != "bank":
            include(resolve_graph_path(graph["kc_bank_path"], base_dir, graph))
        for reference in _image_references(graph):
            include(resolve_asset_path(reference, base_dir, graph))
        if stage == "evaluate":
            question_path = layout.final("question_templates")
            questions = require_json(question_path, job.paper_id)
            include(question_path)
            for reference in _image_references(questions):
                include(resolve_asset_path(reference, base_dir, graph))

    content = []
    for path in sorted(inputs, key=lambda item: item.as_posix()):
        try:
            label = path.relative_to(PROJECT_ROOT.resolve()).as_posix()
        except ValueError:
            label = path.as_posix()
        content.append({"path": label, "sha256": _sha256(path)})
    manifest = {"version": _VERSION, "stage": stage, "paper_id": job.paper_id, "model": model, "inputs": content, "configuration": _configuration(stage, model)}
    return hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


def _stamp_path(output_path: Path) -> Path:
    return output_path.parent / "cache" / "pipeline" / f"{output_path.name}.json"


def prepare_stage_execution(output_path: Path, signature: str) -> bool:
    """Record an input attempt and return whether algorithm caches must restart.

    Only an interrupted attempt with the same input signature may resume. This
    stamp never authorizes output reuse; that remains a completion/hash check.
    """
    if not isinstance(signature, str) or not signature:
        raise ValueError("A pipeline attempt requires a nonempty signature.")
    attempt_path = _stamp_path(output_path).with_name(f"{output_path.name}.attempt.json")
    restart = True
    if attempt_path.exists():
        if not attempt_path.is_file():
            raise ValueError(f"Pipeline attempt stamp is not a file: {attempt_path}")
        try:
            attempt = json.loads(attempt_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ValueError(f"Invalid pipeline attempt stamp: {attempt_path}") from exc
        if not isinstance(attempt, dict) or not isinstance(attempt.get("signature"), str) or not attempt["signature"]:
            raise ValueError(f"Invalid pipeline attempt stamp schema: {attempt_path}")
        restart = attempt["signature"] != signature
    write_json_atomic(attempt_path, {"signature": signature})
    return restart


def stage_is_current(output_path: Path, signature: str) -> bool:
    stamp_path = _stamp_path(output_path)
    if not stamp_path.exists():
        return False
    if not stamp_path.is_file():
        raise ValueError(f"Pipeline completion stamp is not a file: {stamp_path}")
    try:
        stamp = json.loads(stamp_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"Invalid pipeline completion stamp: {stamp_path}") from exc
    if not isinstance(stamp, dict) or stamp.get("version") != _VERSION or not isinstance(stamp.get("signature"), str) or not stamp["signature"] or not re.fullmatch(r"[0-9a-f]{64}", str(stamp.get("output_sha256", ""))):
        raise ValueError(f"Invalid pipeline completion stamp schema: {stamp_path}")
    return stamp["signature"] == signature and output_path.is_file() and stamp["output_sha256"] == _sha256(output_path)


def record_stage_completion(output_path: Path, signature: str) -> None:
    if not isinstance(signature, str) or not signature:
        raise ValueError("A pipeline completion stamp requires a nonempty signature.")
    if not output_path.is_file():
        raise FileNotFoundError(f"Cannot stamp missing pipeline output: {output_path}")
    write_json_atomic(_stamp_path(output_path), {"version": _VERSION, "signature": signature, "output_sha256": _sha256(output_path)})
