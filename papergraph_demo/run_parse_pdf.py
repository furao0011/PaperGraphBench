from __future__ import annotations

import base64
import json
import os
import re
import shutil
import ssl
import time
import urllib.error
import urllib.request
from pathlib import Path

from src.artifact_layout import paper_data_root, safe_paper_id
from src.config import load_dotenv
from src.progress import log, span


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
DEFAULT_PDF_INPUT_DIR = PROJECT_ROOT / "pdfInput"
DEFAULT_RAW_PAPER_ROOT = PROJECT_ROOT / "rawPaper"


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    api_url = _required_env("PADDLE_OCR_API_URL")
    token = _required_env("PADDLE_OCR_TOKEN")
    timeout_s = int(os.getenv("PADDLE_OCR_TIMEOUT_S", "600") or "600")
    overwrite = _env_bool("PAPERGRAPH_OCR_OVERWRITE", False)
    ocr_retries = _env_positive_int("PADDLE_OCR_MAX_RETRIES", 3)
    download_retries = _env_positive_int("PADDLE_OCR_DOWNLOAD_MAX_RETRIES", ocr_retries)
    retry_sleep_s = _env_float("PADDLE_OCR_RETRY_SLEEP_S", 3.0)

    pdf_paths = _input_pdf_paths()
    raw_root = _env_path("RAW_PAPER_ROOT", DEFAULT_RAW_PAPER_ROOT).resolve()
    raw_root.mkdir(parents=True, exist_ok=True)
    fixed_paper_id = safe_paper_id(os.getenv("PAPER_ID", ""))
    if fixed_paper_id and len(pdf_paths) > 1:
        raise ValueError("PAPER_ID can only be set when parsing a single PDF. Leave PAPER_ID empty for batch OCR.")

    log(
        "PDF OCR parse configuration loaded",
        pdfs=len(pdf_paths),
        raw_root=raw_root,
        overwrite=overwrite,
        ocr_retries=ocr_retries,
        download_retries=download_retries,
    )
    for pdf_path in pdf_paths:
        paper_id = fixed_paper_id or safe_paper_id(pdf_path.stem)
        if not paper_id:
            raise ValueError(f"Cannot derive paper_id from PDF filename: {pdf_path.name}. Set PAPER_ID for a single PDF.")
        output_dir = (raw_root / paper_id).resolve()
        with span("parse PDF with OCR", pdf=pdf_path.name, paper_id=paper_id):
            staging_dir = _prepare_staging_output_dir(output_dir, raw_root, overwrite)
            try:
                result = _call_paddle_ocr(api_url, token, pdf_path, timeout_s, ocr_retries, retry_sleep_s)
                _write_ocr_result(
                    result,
                    pdf_path,
                    paper_id,
                    staging_dir,
                    output_dir,
                    timeout_s,
                    download_retries,
                    retry_sleep_s,
                )
                _commit_output_dir(staging_dir, output_dir, raw_root, overwrite)
            except Exception:
                _cleanup_staging_dir(staging_dir, raw_root)
                raise
        print(f"OCR markdown written: {output_dir}")


def _input_pdf_paths() -> list[Path]:
    input_file = os.getenv("PAPERGRAPH_PDF_INPUT_FILE", "").strip()
    if input_file:
        path = Path(input_file).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"PAPERGRAPH_PDF_INPUT_FILE is not a file: {path}")
        if path.suffix.lower() != ".pdf":
            raise ValueError(f"PAPERGRAPH_PDF_INPUT_FILE must be a PDF: {path}")
        return [path]

    input_dir = _env_path("PAPERGRAPH_PDF_INPUT_DIR", DEFAULT_PDF_INPUT_DIR).expanduser().resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"PAPERGRAPH_PDF_INPUT_DIR is not a directory: {input_dir}")
    pdfs = sorted(input_dir.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDF files found in: {input_dir}")
    return pdfs


def _call_paddle_ocr(
    api_url: str,
    token: str,
    pdf_path: Path,
    timeout_s: int,
    max_attempts: int,
    retry_sleep_s: float,
) -> dict:
    file_data = base64.b64encode(pdf_path.read_bytes()).decode("ascii")
    payload = {
        "file": file_data,
        "fileType": int(os.getenv("PADDLE_OCR_FILE_TYPE", "0") or "0"),
        "useDocOrientationClassify": _env_bool("PADDLE_OCR_USE_DOC_ORIENTATION_CLASSIFY", False),
        "useDocUnwarping": _env_bool("PADDLE_OCR_USE_DOC_UNWARPING", False),
        "useChartRecognition": _env_bool("PADDLE_OCR_USE_CHART_RECOGNITION", False),
    }
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        api_url,
        data=body,
        headers={
            "Authorization": f"token {token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with span("call Paddle OCR API", pdf=pdf_path.name, bytes=pdf_path.stat().st_size):
        response_body = _open_url_with_retry(
            request,
            timeout_s=timeout_s,
            max_attempts=max_attempts,
            retry_sleep_s=retry_sleep_s,
            label="Paddle OCR request",
        ).decode("utf-8")
    parsed = json.loads(response_body)
    result = parsed.get("result")
    if not isinstance(result, dict) or not isinstance(result.get("layoutParsingResults"), list):
        raise ValueError("Paddle OCR response does not contain result.layoutParsingResults.")
    return result


def _write_ocr_result(
    result: dict,
    pdf_path: Path,
    paper_id: str,
    output_dir: Path,
    final_output_dir: Path,
    timeout_s: int,
    download_retries: int,
    retry_sleep_s: float,
) -> None:
    layout_results = result["layoutParsingResults"]
    image_count = 0
    layout_image_count = 0
    with span("write OCR markdown and images", docs=len(layout_results), output_dir=output_dir):
        for idx, item in enumerate(layout_results):
            markdown = item.get("markdown")
            if not isinstance(markdown, dict):
                raise ValueError(f"layoutParsingResults[{idx}] has no markdown object.")
            text = markdown.get("text")
            if not isinstance(text, str):
                raise ValueError(f"layoutParsingResults[{idx}].markdown.text is not a string.")
            (output_dir / f"doc_{idx}.md").write_text(text, encoding="utf-8")

            images = markdown.get("images") or {}
            if not isinstance(images, dict):
                raise ValueError(f"layoutParsingResults[{idx}].markdown.images is not an object.")
            log("writing OCR page assets", doc_index=idx, markdown_images=len(images))
            for relative_path, image_url in images.items():
                target = (output_dir / str(relative_path)).resolve()
                _assert_inside(target, output_dir)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(_download_bytes(str(image_url), timeout_s, download_retries, retry_sleep_s))
                image_count += 1

            output_images = item.get("outputImages") or {}
            if not isinstance(output_images, dict):
                raise ValueError(f"layoutParsingResults[{idx}].outputImages is not an object.")
            log("writing OCR layout images", doc_index=idx, layout_images=len(output_images))
            for image_name, image_url in output_images.items():
                filename = f"{_safe_filename(str(image_name))}_{idx}.jpg"
                target = output_dir / filename
                target.write_bytes(_download_bytes(str(image_url), timeout_s, download_retries, retry_sleep_s))
                layout_image_count += 1

    manifest = {
        "paper_id": paper_id,
        "source_pdf": str(pdf_path),
        "raw_paper_dir": str(final_output_dir),
        "doc_count": len(layout_results),
        "markdown_image_count": image_count,
        "layout_image_count": layout_image_count,
        "build_graph_env": {
            "PAPER_INPUT_DIR": str(final_output_dir),
            "PAPER_ID": paper_id,
        },
        "papergraph_data_dir": str(paper_data_root(BASE_DIR, paper_id)),
    }
    (output_dir / "papergraph_ocr_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    log(
        "OCR markdown/images staged",
        docs=len(layout_results),
        markdown_images=image_count,
        layout_images=layout_image_count,
    )


def _download_bytes(url: str, timeout_s: int, max_attempts: int, retry_sleep_s: float) -> bytes:
    if not url:
        raise ValueError("Image URL is empty.")
    return _open_url_with_retry(
        url,
        timeout_s=timeout_s,
        max_attempts=max_attempts,
        retry_sleep_s=retry_sleep_s,
        label="OCR image download",
    )


def _open_url_with_retry(
    request: urllib.request.Request | str,
    timeout_s: int,
    max_attempts: int,
    retry_sleep_s: float,
    label: str,
) -> bytes:
    last_exc: Exception | None = None
    attempts = max(1, max_attempts)
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout_s) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            last_exc = exc
            if attempt >= attempts or not _retryable_http_status(exc.code):
                break
        except (urllib.error.URLError, TimeoutError, ssl.SSLError, ConnectionError) as exc:
            last_exc = exc
            if attempt >= attempts:
                break
        sleep_s = max(0.0, retry_sleep_s) * (2 ** max(0, attempt - 1))
        log(
            "OCR network retry",
            operation=label,
            attempt=attempt,
            attempts=attempts,
            sleep_s=f"{sleep_s:.1f}",
            error=f"{type(last_exc).__name__}: {last_exc}",
        )
        if sleep_s > 0:
            time.sleep(sleep_s)
    raise RuntimeError(f"{label} failed after {attempts} attempt(s): {type(last_exc).__name__}: {last_exc}") from last_exc


def _retryable_http_status(status_code: int) -> bool:
    return status_code in {408, 429} or status_code >= 500


def _prepare_staging_output_dir(output_dir: Path, raw_root: Path, overwrite: bool) -> Path:
    _assert_inside(output_dir, raw_root)
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"OCR output already exists: {output_dir}. Set PAPERGRAPH_OCR_OVERWRITE=true to replace it."
            )
    staging_dir = output_dir.with_name(output_dir.name + ".tmp")
    _assert_inside(staging_dir, raw_root)
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=False)
    return staging_dir


def _commit_output_dir(staging_dir: Path, output_dir: Path, raw_root: Path, overwrite: bool) -> None:
    _assert_inside(staging_dir, raw_root)
    _assert_inside(output_dir, raw_root)
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"OCR output already exists: {output_dir}. Set PAPERGRAPH_OCR_OVERWRITE=true to replace it."
            )
        shutil.rmtree(output_dir)
    staging_dir.replace(output_dir)
    log("OCR output committed", output_dir=output_dir)


def _cleanup_staging_dir(staging_dir: Path, raw_root: Path) -> None:
    _assert_inside(staging_dir, raw_root)
    if staging_dir.exists():
        shutil.rmtree(staging_dir)


def _assert_inside(path: Path, root: Path) -> None:
    path.resolve().relative_to(root.resolve())


def _required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} is required.")
    return value


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_path(name: str, default: Path) -> Path:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return Path(raw.strip())


def _env_positive_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return max(1, default)
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer, got {raw!r}.") from exc
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value}.")
    return value


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}.") from exc


def _safe_filename(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return safe.strip("._-") or "image"


if __name__ == "__main__":
    main()
