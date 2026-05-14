from __future__ import annotations

import json
import os
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from src.model_client import OpenAICompatClient
from src.progress import log, span
from src.prompt_loader import load_prompt, render_prompt


FAILURE_MODES = {
    "overclaim",
    "wrong_relation",
    "false_premise",
    "thread_overclaim",
    "thread_wrong_bridge",
    "thread_premise_mutation",
    "other",
    "none",
}


def challenge_solver_configs(default_model: str, provider: str = "common_api") -> list[dict]:
    return _solver_configs(default_model, provider=provider)


def run_single_challenge_question_trials(
    question: dict,
    client: OpenAICompatClient,
    paper_text: str,
    solver_client: OpenAICompatClient | None = None,
) -> dict:
    if not client or not client.is_ready():
        raise RuntimeError("Challenge filtering requires a configured online model client.")
    if not isinstance(paper_text, str) or not paper_text.strip():
        raise ValueError("Challenge trial requires non-empty full paper text.")
    solver_runtime = solver_client or client
    provider = "vision_api" if _requires_multimodal_input(question) else "thread_api" if _is_thread_challenge(question) else "common_api"
    solvers = _solver_configs(solver_runtime.cfg.llm_model, provider=provider)
    return _normalize_trial_bundle(
        question,
        _run_question_trials(question, solvers, client, paper_text, solver_client=solver_runtime),
        solvers,
    )


def question_with_filter_metadata(question: dict, bundle: dict) -> dict:
    return _question_with_filter_metadata(question, bundle)


def filter_challenge_questions(
    raw_questions: dict,
    client: OpenAICompatClient,
    cache_path: Path,
    paper_text: str,
    resume: bool = False,
    restart: bool = False,
) -> dict:
    if not client or not client.is_ready():
        raise RuntimeError("Challenge filtering requires a configured online model client.")
    questions = raw_questions.get("challenge_questions_raw", [])
    if not isinstance(questions, list) or not questions:
        raise ValueError("Challenge filtering requires non-empty challenge_questions_raw.")
    if not isinstance(paper_text, str) or not paper_text.strip():
        raise ValueError("Challenge filtering requires non-empty full paper text.")

    solvers = _solver_configs(client.cfg.llm_model)
    signature = _raw_question_signature(raw_questions, solvers, paper_text)
    cache = _load_cache(cache_path) if resume and not restart else {}
    if cache.get("raw_question_signature") != signature:
        cache = {
            "paper_id": raw_questions.get("paper_id", "unknown"),
            "raw_question_signature": signature,
            "trials_by_question_id": {},
        }
        _write_json(cache_path, cache)
    cache.setdefault("trials_by_question_id", {})

    workers = min(_env_positive_int("CHALLENGE_FILTER_WORKERS", 3), len(questions))
    completed: dict[str, dict] = {}
    futures = {}
    errors: list[str] = []

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for question in questions:
            question_id = str(question.get("question_id", "")).strip()
            if not question_id:
                raise ValueError("Every raw challenge question must include question_id.")
            cached = cache["trials_by_question_id"].get(question_id)
            if cached:
                try:
                    completed[question_id] = _normalize_trial_bundle(question, cached, solvers)
                    log("challenge filter cache hit", question_id=question_id)
                    continue
                except Exception:
                    cache["trials_by_question_id"].pop(question_id, None)
                    _write_json(cache_path, cache)
            futures[ex.submit(_run_question_trials, question, solvers, client, paper_text)] = question

        for fut in as_completed(futures):
            question = futures[fut]
            question_id = question.get("question_id")
            try:
                bundle = fut.result()
                bundle = _normalize_trial_bundle(question, bundle, solvers)
                cache["trials_by_question_id"][question_id] = bundle
                _write_json(cache_path, cache)
                completed[question_id] = bundle
                log("challenge question filtered", question_id=question_id, wrong_count=bundle["wrong_count"])
            except Exception as exc:
                errors.append(f"{question_id}: {type(exc).__name__}: {exc}")
                log("challenge filtering error", question_id=question_id, error=f"{type(exc).__name__}: {exc}")

    if errors:
        raise RuntimeError("Challenge filtering failed: " + "; ".join(errors[:5]))

    trial_bundles = []
    filtered = []
    human_review = []
    rejected = []
    for question in questions:
        question_id = question["question_id"]
        bundle = completed.get(question_id)
        if not bundle:
            raise RuntimeError(f"Missing challenge filtering bundle for question_id={question_id}.")
        trial_bundles.append(bundle)
        item = _question_with_filter_metadata(question, bundle)
        if bundle["wrong_count"] == 0:
            item["filter_reason"] = "too_easy"
            rejected.append(item)
        else:
            if bundle["wrong_count"] == bundle["solver_count"]:
                item["needs_human_review"] = True
                item["all_solvers_failed"] = True
                human_review.append(item)
            filtered.append(item)

    return {
        "paper_id": raw_questions.get("paper_id", "unknown"),
        "schema_version": "v2",
        "source_raw_question_signature": signature,
        "solver_configs": solvers,
        "solver_trials": trial_bundles,
        "challenge_questions_filtered": filtered,
        "challenge_questions_need_human_review": human_review,
        "challenge_questions_rejected": rejected,
        "summary": {
            "raw_question_count": len(questions),
            "solver_count": len(solvers),
            "filtered_count": len(filtered),
            "human_review_count": len(human_review),
            "rejected_count": len(rejected),
        },
    }


def _solver_configs(default_model: str, provider: str = "common_api") -> list[dict]:
    if provider == "vision_api":
        count = _env_positive_int(
            "MULTIMODAL_CHALLENGE_SOLVER_COUNT",
            _env_positive_int("CHALLENGE_SOLVER_COUNT", 3),
        )
        models = _text_list(os.getenv("MULTIMODAL_CHALLENGE_SOLVER_MODELS", ""))
        temperature = _env_float(
            "MULTIMODAL_CHALLENGE_SOLVER_TEMPERATURE",
            _env_float("CHALLENGE_SOLVER_TEMPERATURE", 1.5),
        )
        timeout_s = _env_positive_int(
            "MULTIMODAL_CHALLENGE_SOLVER_TIMEOUT_S",
            _env_positive_int("VISION_TIMEOUT_S", 180),
        )
    elif provider == "thread_api":
        count = _env_positive_int(
            "THREAD_CHALLENGE_SOLVER_COUNT",
            _env_positive_int("CHALLENGE_SOLVER_COUNT", 3),
        )
        models = _text_list(os.getenv("THREAD_CHALLENGE_SOLVER_MODELS", ""))
        temperature = _env_float(
            "THREAD_CHALLENGE_SOLVER_TEMPERATURE",
            _env_float("CHALLENGE_SOLVER_TEMPERATURE", 1.5),
        )
        timeout_s = _env_nonnegative_int(
            "THREAD_CHALLENGE_SOLVER_TIMEOUT_S",
            _env_nonnegative_int("CHALLENGE_SOLVER_TIMEOUT_S", 0),
        )
    else:
        count = _env_positive_int("CHALLENGE_SOLVER_COUNT", 3)
        models = _text_list(os.getenv("CHALLENGE_SOLVER_MODELS", ""))
        temperature = _env_float("CHALLENGE_SOLVER_TEMPERATURE", 1.5)
        timeout_s = _env_nonnegative_int("CHALLENGE_SOLVER_TIMEOUT_S", 0)
    if not models:
        models = [default_model for _ in range(count)]
    if len(models) != count:
        raise ValueError(
            f"Solver model list must provide exactly {count} values for provider={provider}."
        )
    return [
        {
            "solver_id": f"solver_{idx}",
            "provider": provider,
            "model": models[idx - 1],
            "temperature": temperature,
            "timeout_s": timeout_s,
        }
        for idx in range(1, count + 1)
    ]


def _run_question_trials(
    question: dict,
    solvers: list[dict],
    client: OpenAICompatClient,
    paper_text: str,
    solver_client: OpenAICompatClient | None = None,
) -> dict:
    trials_by_id = {}
    judge_tpl = load_prompt("judge_challenge_answer.txt")
    solver_runtime = solver_client or client
    if any(solver.get("provider") == "vision_api" for solver in solvers):
        worker_default = _env_positive_int("CHALLENGE_SOLVER_WORKERS", len(solvers))
        workers = _env_positive_int("MULTIMODAL_CHALLENGE_SOLVER_WORKERS", worker_default)
    elif any(solver.get("provider") == "thread_api" for solver in solvers):
        worker_default = _env_positive_int("CHALLENGE_SOLVER_WORKERS", len(solvers))
        workers = _env_positive_int("THREAD_CHALLENGE_SOLVER_WORKERS", worker_default)
    else:
        workers = _env_positive_int("CHALLENGE_SOLVER_WORKERS", len(solvers))
    workers = min(workers, len(solvers))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {
            ex.submit(_run_single_solver_trial, question, solver, client, paper_text, judge_tpl, solver_runtime): solver
            for solver in solvers
        }
        for fut in as_completed(futures):
            trial = fut.result()
            trials_by_id[trial["solver_id"]] = trial
    return {
        "question_id": question["question_id"],
        "solver_trials": [trials_by_id[solver["solver_id"]] for solver in solvers],
    }


def _run_single_solver_trial(
    question: dict,
    solver: dict,
    judge_client: OpenAICompatClient,
    paper_text: str,
    judge_tpl: str,
    solver_client: OpenAICompatClient,
) -> dict:
    answer = _solve_question(question, solver, solver_client, paper_text)
    judge_result = _judge_solver_answer(question, solver, answer, judge_client, judge_tpl)
    return {
        "solver_id": solver["solver_id"],
        "provider": solver["provider"],
        "model": solver["model"],
        "temperature": solver["temperature"],
        "answer": answer,
        "judge_result": judge_result,
    }


def _solve_question(question: dict, solver: dict, client: OpenAICompatClient, paper_text: str) -> str:
    prompt = _build_solver_eval_prompt(paper_text, question)
    image_paths = _question_image_paths(question)
    with span("challenge solver answer", question_id=question["question_id"], solver_id=solver["solver_id"]):
        system_prompt = (
            "Answer the paper-evaluation question based only on the provided original paper, attached figure/table assets, and dialogue context."
        )
        if _requires_multimodal_input(question) and image_paths:
            answer = client.chat_text_with_images(
                system_prompt=system_prompt,
                user_prompt=prompt,
                image_paths=image_paths,
                temperature=float(solver["temperature"]),
                model=solver["model"],
                timeout_s=int(solver.get("timeout_s") or 0) or None,
            )
        else:
            answer = client.chat_text(
                system_prompt=system_prompt,
                user_prompt=prompt,
                temperature=float(solver["temperature"]),
                model=solver["model"],
                timeout_s=int(solver.get("timeout_s") or 0) or None,
            )
    answer = answer.strip()
    if not answer:
        raise ValueError(f"Solver {solver['solver_id']} returned an empty answer for {question['question_id']}.")
    return answer


def _judge_solver_answer(
    question: dict,
    solver: dict,
    answer: str,
    client: OpenAICompatClient,
    judge_tpl: str,
) -> dict:
    payload = {
        "question_id": question["question_id"],
        "challenge_type": question.get("challenge_type"),
        "question_text": question.get("question_text"),
        "answer": answer,
        "expected_behavior": question.get("expected_behavior"),
        "target_failure_mode": question.get("target_failure_mode"),
        "evidence": question.get("evidence", []),
        "solver": solver,
    }
    with span("judge challenge answer", question_id=question["question_id"], solver_id=solver["solver_id"]):
        result = client.chat_json(
            system_prompt="You judge challenge-question answers for paper evaluation. Return JSON only.",
            user_prompt=render_prompt(
                judge_tpl,
                challenge_trial_json=json.dumps(payload, ensure_ascii=False, indent=2),
            ),
            temperature=_env_float("CHALLENGE_JUDGE_TEMPERATURE", 0.1),
        )
    return _normalize_judge_result(question, solver, result)


def _normalize_trial_bundle(question: dict, bundle: dict, solvers: list[dict]) -> dict:
    if not isinstance(bundle, dict):
        raise ValueError(f"Trial bundle for {question['question_id']} must be an object.")
    trials = bundle.get("solver_trials", [])
    if not isinstance(trials, list):
        raise ValueError(f"Trial bundle for {question['question_id']} must contain solver_trials list.")
    expected_ids = [solver["solver_id"] for solver in solvers]
    seen_ids = [trial.get("solver_id") for trial in trials]
    if seen_ids != expected_ids:
        raise ValueError(
            f"Trial bundle for {question['question_id']} solver order mismatch: expected {expected_ids}, got {seen_ids}."
        )
    normalized_trials = []
    wrong_count = 0
    matched_target_failure_count = 0
    for trial in trials:
        solver = next(item for item in solvers if item["solver_id"] == trial.get("solver_id"))
        answer = str(trial.get("answer", "")).strip()
        if not answer:
            raise ValueError(f"Trial {question['question_id']}:{solver['solver_id']} has empty answer.")
        judge_result = _normalize_judge_result(question, solver, trial.get("judge_result", {}))
        if judge_result["is_wrong"]:
            wrong_count += 1
        if judge_result["matched_target_failure"]:
            matched_target_failure_count += 1
        normalized_trials.append(
            {
                "solver_id": solver["solver_id"],
                "provider": solver["provider"],
                "model": _non_empty_text(
                    trial.get("model"),
                    f"Trial {question['question_id']}:{solver['solver_id']} missing model.",
                ),
                "temperature": float(trial.get("temperature", solver["temperature"])),
                "answer": answer,
                "judge_result": judge_result,
            }
        )
    return {
        "question_id": question["question_id"],
        "solver_trials": normalized_trials,
        "wrong_count": wrong_count,
        "matched_target_failure_count": matched_target_failure_count,
        "solver_count": len(solvers),
    }


def _normalize_judge_result(question: dict, solver: dict, result: dict) -> dict:
    if not isinstance(result, dict):
        raise ValueError(f"Judge result for {question['question_id']}:{solver['solver_id']} must be an object.")
    if "is_wrong" not in result or "matched_target_failure" not in result:
        raise ValueError(
            f"Judge result for {question['question_id']}:{solver['solver_id']} must include is_wrong and matched_target_failure."
        )
    failure_mode = str(result.get("failure_mode", "none")).strip()
    if failure_mode not in FAILURE_MODES:
        raise ValueError(
            f"Judge result for {question['question_id']}:{solver['solver_id']} has invalid failure_mode={failure_mode!r}."
        )
    return {
        "is_wrong": bool(result["is_wrong"]),
        "matched_target_failure": bool(result["matched_target_failure"]),
        "failure_mode": failure_mode,
        "confidence": _bounded_float(result.get("confidence", 0.0)),
        "reason": str(result.get("reason", "")).strip(),
    }


def _question_with_filter_metadata(question: dict, bundle: dict) -> dict:
    item = dict(question)
    item["solver_trial_summary"] = {
        "wrong_count": bundle["wrong_count"],
        "matched_target_failure_count": bundle["matched_target_failure_count"],
        "solver_count": bundle["solver_count"],
        "thread_context_used": bool(
            isinstance(question.get("synthetic_thread_history"), dict)
            and question.get("synthetic_thread_history", {}).get("thread_context_used")
        ),
    }
    item["needs_human_review"] = False
    item["all_solvers_failed"] = False
    return item


def _raw_question_signature(raw_questions: dict, solvers: list[dict], paper_text: str) -> dict:
    questions = raw_questions.get("challenge_questions_raw", [])
    return {
        "paper_id": raw_questions.get("paper_id", "unknown"),
        "schema_version": raw_questions.get("schema_version"),
        "source_challenge_plan_signature": raw_questions.get("source_challenge_plan_signature"),
        "solver_configs": solvers,
        "paper_text_sha256": hashlib.sha256(paper_text.encode("utf-8")).hexdigest(),
        "question_ids": [question.get("question_id") for question in questions],
        "question_sources": [
            [
                question.get("question_id"),
                question.get("source_plan_id"),
                question.get("target_failure_mode"),
                question.get("question_text"),
            ]
            for question in questions
        ],
    }


def _build_solver_eval_prompt(paper_text: str, question: dict) -> str:
    asset_context = _asset_context_for_prompt(question)
    synthetic_history = question.get("synthetic_thread_history")
    if not isinstance(synthetic_history, dict):
        synthetic_history = {}
    history_text = str(synthetic_history.get("history_text") or "").strip() or "No previous turns."
    return (
        "```original paper\n"
        f"{paper_text}\n"
        "```\n\n"
        f"{asset_context}"
        "[dialogue history]\n"
        f"{history_text}\n\n"
        "[current question]\n"
        f"{question['question_text']}"
    )


def _requires_multimodal_input(question: dict) -> bool:
    return bool(question.get("requires_multimodal_input") or question.get("asset_references"))


def _is_thread_challenge(question: dict) -> bool:
    return (
        question.get("question_type") == "thread_challenge_question"
        or question.get("challenge_scope") == "thread"
        or str(question.get("challenge_type") or "").startswith("thread_")
    )


def _question_image_paths(question: dict) -> list[str]:
    paths = []
    for ref in question.get("asset_references", []):
        for attachment in ref.get("attachments", []):
            if attachment.get("type") == "image" and str(attachment.get("path", "")).strip():
                paths.append(str(attachment["path"]).strip())
    return paths


def _asset_context_for_prompt(question: dict) -> str:
    refs = question.get("asset_references", [])
    if not refs:
        return ""
    lines = ["[attached multimodal assets]"]
    for ref in refs:
        lines.append(f"- asset_id: {ref.get('asset_id')}")
        lines.append(f"  asset_type: {ref.get('asset_type')}")
        if str(ref.get("caption") or "").strip():
            lines.append(f"  caption: {ref.get('caption')}")
        if str(ref.get("summary") or "").strip():
            lines.append(f"  summary: {ref.get('summary')}")
        evidence_bases = ref.get("evidence_bases", [])
        if evidence_bases:
            lines.append("  evidence_bases:")
            for basis in evidence_bases:
                lines.append(f"    - {basis}")
        for attachment in ref.get("attachments", []):
            if attachment.get("type") == "table_latex":
                lines.append("  table_latex:")
                lines.append("```latex")
                lines.append(str(attachment.get("content") or ""))
                lines.append("```")
            elif attachment.get("type") == "image":
                lines.append(f"  image_path: {attachment.get('path')}")
    return "\n".join(lines) + "\n\n"


def _text_list(raw: str) -> list[str]:
    values = []
    for item in raw.split(","):
        text = item.strip()
        if not text:
            continue
        values.append(text)
    return values


def _non_empty_text(value: object, error: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(error)
    return text


def _bounded_float(value: object) -> float:
    try:
        raw = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Expected a float in [0, 2], got {value!r}.") from exc
    if raw < 0 or raw > 2:
        raise ValueError(f"Expected a float in [0, 2], got {raw}.")
    return round(raw, 4)


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return _bounded_float(raw)


def _env_positive_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer, got {raw!r}.") from exc
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value}.")
    return value


def _env_nonnegative_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a non-negative integer, got {raw!r}.") from exc
    if value < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {value}.")
    return value


def _load_cache(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Cannot read challenge filter cache {path}: {type(exc).__name__}: {exc}") from exc


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)
