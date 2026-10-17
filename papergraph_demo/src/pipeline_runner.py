"""Execute paper stage plans with isolated logs and checked outputs."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from src.pipeline_artifacts import completed_evaluation, validate_graph_inputs, validate_question_inputs
from src.pipeline_cache import build_stage_signature, prepare_stage_execution, record_stage_completion, stage_is_current
from src.pipeline_plans import PROJECT_ROOT, SCRIPTS, PaperJob, paper_layout, require_json, result_dir


def target_environment(model: str) -> dict[str, str]:
    env = os.environ.copy()
    env["EVAL_TARGET_MODEL"] = model
    suffix = re.sub(r"[^A-Za-z0-9]+", "_", model).strip("_").upper()
    for name in ("EVAL_TARGET_API_KEY", "EVAL_TARGET_BASE_URL"):
        override = env.get(f"{name}_{suffix}", "").strip()
        if override:
            env[name] = override
    return env


def stage_output(job: PaperJob, stage: str, model: str | None) -> Path:
    if stage == "parse_pdf":
        return Path(os.environ["RAW_PAPER_ROOT"]) / job.paper_id / "papergraph_ocr_manifest.json"
    if stage == "evaluate":
        return result_dir(model, job.paper_id) / "evaluation_report.json"
    return paper_layout(job.paper_id).final("master_graph" if stage == "build_graph" else "question_templates")


def validate_output(job: PaperJob, stage: str, model: str | None) -> bool:
    output = stage_output(job, stage, model)
    if not output.is_file():
        return False
    require_json(output, job.paper_id)
    if stage == "build_graph":
        validate_graph_inputs(job.paper_id)
        return True
    if stage == "generate_questions":
        validate_question_inputs(job.paper_id)
        return True
    if stage == "parse_pdf":
        from run_parse_pdf import _assert_completed_ocr_output

        _assert_completed_ocr_output(output.parent, job.pdf, job.paper_id)
        return True
    return completed_evaluation(output, job.paper_id, model)


def _command(job: PaperJob, stage: str) -> list[str]:
    command = [sys.executable, str(PROJECT_ROOT / "papergraph_demo" / SCRIPTS[stage])]
    if stage == "parse_pdf":
        command.extend([
            "--input-file", str(job.pdf), "--paper-id", job.paper_id,
            "--raw-paper-root", os.environ["RAW_PAPER_ROOT"],
        ])
    return command


def _stage_environment(job: PaperJob, model: str | None, force: bool) -> dict[str, str]:
    env = target_environment(model) if model else os.environ.copy()
    for name in (
        "PAPER_INPUT_DIR", "PAPER_INPUT_FILE", "PAPERGRAPH_GRAPH_PATH", "PAPERGRAPH_QUESTION_PATH",
        "BUILD_GRAPH_CHECKPOINT_PATH", "QUESTION_CACHE_PATH", "CHALLENGE_LOOP_CACHE_PATH",
        "EVAL_CHECKPOINT_PATH", "EVAL_ARTIFACT_DIR", "PAPERGRAPH_PDF_INPUT_FILE",
    ):
        env.pop(name, None)
    env["PAPER_ID"] = job.paper_id
    if job.source:
        name = "PAPER_INPUT_FILE" if job.source.is_file() else "PAPER_INPUT_DIR"
        env[name] = str(job.source)
    for prefix in ("PAPERGRAPH", "BUILD_GRAPH", "QUESTION", "EVAL"):
        env[f"{prefix}_RESTART"] = "true" if force else "false"
        env[f"{prefix}_RESUME"] = "false" if force else "true"
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def execute_jobs(jobs: list[PaperJob], models: list[str], *, workers: int, force: bool, dry_run: bool) -> None:
    if dry_run:
        for job in jobs:
            for stage in job.stages:
                for model in models if stage == "evaluate" else [None]:
                    print(f"[plan] {job.paper_id} | {stage} | model={model or '-'} | output={stage_output(job, stage, model)}")
        return

    from src.batch_progress import BatchTask, PaperBatchProgress

    tasks = [BatchTask(job.paper_id, sum(len(models) if stage == "evaluate" else 1 for stage in job.stages)) for job in jobs]
    failures: list[str] = []
    with PaperBatchProgress("pipeline", tasks) as progress:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_run_paper, job, models, force, progress): job.paper_id for job in jobs}
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as exc:
                    failures.append(f"{futures[future]}: {exc}")
    if failures:
        raise RuntimeError("Pipeline failed:\n" + "\n".join(failures))


def _run_paper(job: PaperJob, models: list[str], force: bool, progress) -> None:
    progress.start(job.paper_id, "starting")
    try:
        for stage in job.stages:
            for model in models if stage == "evaluate" else [None]:
                label = stage + (f" ({model})" if model else "")
                if stage in {"generate_questions", "evaluate"}:
                    graph = validate_graph_inputs(job.paper_id)
                    if stage == "evaluate":
                        validate_question_inputs(job.paper_id, graph)
                output = stage_output(job, stage, model)
                signature = build_stage_signature(job, stage, model)
                if not force and stage_is_current(output, signature) and validate_output(job, stage, model):
                    progress.update(job.paper_id, status=f"{label}: unchanged completed output reused", advance=1, emit=True)
                    continue
                restart = prepare_stage_execution(output, signature) or force
                command = _command(job, stage)
                if restart and stage == "parse_pdf":
                    command.append("--overwrite")
                log_name = "evaluate_" + re.sub(r"[^\w.-]+", "_", model) if model else stage
                log_path = PROJECT_ROOT / "outputs" / "logs" / job.paper_id / f"{log_name}.log"
                log_path.parent.mkdir(parents=True, exist_ok=True)
                progress.update(job.paper_id, status=f"{label}: running; log={log_path}", emit=True)
                with log_path.open("a", encoding="utf-8") as stream:
                    stream.write(f"\n[{datetime.now().isoformat(timespec='seconds')}] stage={stage} model={model or '-'}\n")
                    stream.flush()
                    previous_mtime = output.stat().st_mtime_ns if output.exists() else None
                    result = subprocess.run(command, cwd=PROJECT_ROOT, env=_stage_environment(job, model, restart), stdout=stream, stderr=subprocess.STDOUT)
                if result.returncode:
                    raise RuntimeError(f"{label} exited {result.returncode}; see {log_path}")
                if not validate_output(job, stage, model):
                    raise RuntimeError(f"{label} exited without completed output: {stage_output(job, stage, model)}; see {log_path}")
                if stage != "parse_pdf" and output.stat().st_mtime_ns == previous_mtime:
                    raise RuntimeError(f"{label} exited without updating its output; see {log_path}")
                record_stage_completion(output, signature)
                progress.update(job.paper_id, status=f"{label}: complete", advance=1, emit=True)
        progress.finish(job.paper_id, "completed")
    except Exception as exc:
        progress.fail(job.paper_id, exc)
        raise
