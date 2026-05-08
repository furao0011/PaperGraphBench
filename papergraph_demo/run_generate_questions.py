import json
import os
from pathlib import Path

from src.challenge_filter import filter_challenge_questions
from src.challenge_plan_builder import build_challenge_plans
from src.challenge_question_generator import generate_raw_challenge_questions
from src.config import load_settings
from src.model_client import ModelConfig, OpenAICompatClient
from src.paper_context import load_full_paper_text
from src.progress import log, span
from src.question_generator import generate_questions_cached


BASE_DIR = Path(__file__).resolve().parent
GRAPH_PATH = BASE_DIR / "data" / "graphs" / "master_graph.json"
QUESTION_PATH = BASE_DIR / "data" / "questions" / "question_templates.json"
QUESTION_CACHE_PATH = BASE_DIR / "data" / "questions" / "question_generation_cache.json"
CHALLENGE_PLAN_PATH = BASE_DIR / "data" / "questions" / "challenge_plans.json"
CHALLENGE_QUESTION_RAW_PATH = BASE_DIR / "data" / "questions" / "challenge_questions_raw.json"
CHALLENGE_QUESTION_CACHE_PATH = BASE_DIR / "data" / "questions" / "challenge_question_generation_cache.json"
CHALLENGE_FILTER_CACHE_PATH = BASE_DIR / "data" / "questions" / "challenge_filter_cache.json"
CHALLENGE_SOLVER_TRIALS_PATH = BASE_DIR / "data" / "questions" / "challenge_solver_trials.json"
CHALLENGE_QUESTION_FILTERED_PATH = BASE_DIR / "data" / "questions" / "challenge_questions_filtered.json"
CHALLENGE_QUESTION_HUMAN_REVIEW_PATH = BASE_DIR / "data" / "questions" / "challenge_questions_need_human_review.json"
CHALLENGE_QUESTION_REJECTED_PATH = BASE_DIR / "data" / "questions" / "challenge_questions_rejected.json"


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def main() -> None:
    if not GRAPH_PATH.exists():
        raise FileNotFoundError(f"Master graph not found: {GRAPH_PATH}")

    settings = load_settings(BASE_DIR.parent)
    allow_offline_fallback = _env_bool("ALLOW_OFFLINE_FALLBACK")
    resume = _env_bool("PAPERGRAPH_RESUME") or _env_bool("QUESTION_RESUME")
    restart = _env_bool("PAPERGRAPH_RESTART") or _env_bool("QUESTION_RESTART")
    cache_path = Path(os.getenv("QUESTION_CACHE_PATH", str(QUESTION_CACHE_PATH)))
    challenge_question_cache_path = Path(
        os.getenv("CHALLENGE_QUESTION_CACHE_PATH", str(CHALLENGE_QUESTION_CACHE_PATH))
    )
    challenge_filter_cache_path = Path(os.getenv("CHALLENGE_FILTER_CACHE_PATH", str(CHALLENGE_FILTER_CACHE_PATH)))
    client = OpenAICompatClient(
        ModelConfig(settings.api_key, settings.base_url, settings.llm_model)
    )
    if not client.is_ready() and not allow_offline_fallback:
        raise RuntimeError("Online question generation requires API_KEY, BASE_URL, and LLM_MODEL. Set ALLOW_OFFLINE_FALLBACK=true only for local debugging.")
    log("loading master graph", path=GRAPH_PATH)
    graph = json.loads(GRAPH_PATH.read_text(encoding="utf-8"))
    with span("load paper text for challenge filtering"):
        paper_text = load_full_paper_text(graph, BASE_DIR)
    log(
        "master graph loaded",
        kcs=len(graph.get("kc_nodes", [])),
        macros=len(graph.get("macro_nodes", [])),
        paths=len(graph.get("reasoning_paths", [])),
        paper_chars=len(paper_text),
    )
    challenge_plans = build_challenge_plans(graph)
    CHALLENGE_PLAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    CHALLENGE_PLAN_PATH.write_text(json.dumps(challenge_plans, ensure_ascii=False, indent=2), encoding="utf-8")
    log(
        "challenge plans generated",
        path=CHALLENGE_PLAN_PATH,
        plans=challenge_plans.get("summary", {}).get("plan_count", 0),
    )
    with span("generate raw challenge questions"):
        raw_challenge_questions = generate_raw_challenge_questions(
            challenge_plans=challenge_plans,
            client=client,
            cache_path=challenge_question_cache_path,
            resume=resume,
            restart=restart,
        )
    CHALLENGE_QUESTION_RAW_PATH.write_text(
        json.dumps(raw_challenge_questions, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    log(
        "raw challenge questions generated",
        path=CHALLENGE_QUESTION_RAW_PATH,
        questions=raw_challenge_questions.get("summary", {}).get("raw_question_count", 0),
    )
    with span("filter challenge questions"):
        challenge_filter_result = filter_challenge_questions(
            raw_questions=raw_challenge_questions,
            client=client,
            cache_path=challenge_filter_cache_path,
            paper_text=paper_text,
            resume=resume,
            restart=restart,
        )
    _write_json(
        CHALLENGE_SOLVER_TRIALS_PATH,
        {
            "paper_id": challenge_filter_result["paper_id"],
            "schema_version": challenge_filter_result["schema_version"],
            "source_raw_question_signature": challenge_filter_result["source_raw_question_signature"],
            "solver_configs": challenge_filter_result["solver_configs"],
            "solver_trials": challenge_filter_result["solver_trials"],
            "summary": challenge_filter_result["summary"],
        },
    )
    _write_json(
        CHALLENGE_QUESTION_FILTERED_PATH,
        {
            "paper_id": challenge_filter_result["paper_id"],
            "schema_version": challenge_filter_result["schema_version"],
            "challenge_questions_filtered": challenge_filter_result["challenge_questions_filtered"],
            "summary": challenge_filter_result["summary"],
        },
    )
    _write_json(
        CHALLENGE_QUESTION_HUMAN_REVIEW_PATH,
        {
            "paper_id": challenge_filter_result["paper_id"],
            "schema_version": challenge_filter_result["schema_version"],
            "challenge_questions_need_human_review": challenge_filter_result["challenge_questions_need_human_review"],
            "summary": challenge_filter_result["summary"],
        },
    )
    _write_json(
        CHALLENGE_QUESTION_REJECTED_PATH,
        {
            "paper_id": challenge_filter_result["paper_id"],
            "schema_version": challenge_filter_result["schema_version"],
            "challenge_questions_rejected": challenge_filter_result["challenge_questions_rejected"],
            "summary": challenge_filter_result["summary"],
        },
    )
    log(
        "challenge questions filtered",
        filtered=challenge_filter_result["summary"]["filtered_count"],
        human_review=challenge_filter_result["summary"]["human_review_count"],
        rejected=challenge_filter_result["summary"]["rejected_count"],
    )
    try:
        with span("generate questions"):
            bundle = generate_questions_cached(
                graph,
                client,
                cache_path=cache_path,
                resume=resume,
                restart=restart,
                allow_offline_fallback=allow_offline_fallback,
            )
    except KeyboardInterrupt:
        log("question generation interrupted; cache saved", cache=cache_path)
        print(f"Question generation interrupted. Cache saved: {cache_path}")
        return
    log(
        "questions generated",
        macro_main=len(bundle.get("macro_main_questions", [])),
        thread_seeds=len(bundle.get("thread_question_seeds", [])),
    )
    payload = {
        "paper_id": graph.get("paper_id", "unknown"),
        "schema_version": "v1",
        "challenge_plans_path": "data/questions/challenge_plans.json",
        "challenge_plan_summary": challenge_plans.get("summary", {}),
        "challenge_questions_raw_path": "data/questions/challenge_questions_raw.json",
        "challenge_question_raw_summary": raw_challenge_questions.get("summary", {}),
        "challenge_questions_filtered_path": "data/questions/challenge_questions_filtered.json",
        "challenge_solver_trials_path": "data/questions/challenge_solver_trials.json",
        "challenge_filter_summary": challenge_filter_result.get("summary", {}),
        "challenge_questions": challenge_filter_result["challenge_questions_filtered"],
        "challenge_scheduler_config": {
            "macro_level_enabled": True,
            "thread_level_enabled": True,
        },
        "macro_main_questions": bundle["macro_main_questions"],
        "thread_question_seeds": bundle["thread_question_seeds"],
        "review_question_seeds": bundle["review_question_seeds"],
        # Compatibility aliases for the current v0 evaluation runner.
        "main_questions": bundle["main_questions"],
        "multi_hop_questions": bundle["multi_hop_questions"],
        "reserved_followup_templates": bundle["reserved_followup_templates"],
    }

    QUESTION_PATH.parent.mkdir(parents=True, exist_ok=True)
    QUESTION_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    log("question templates written", path=QUESTION_PATH, cache=cache_path)
    print(f"Challenge plans generated: {CHALLENGE_PLAN_PATH}")
    print(f"Raw challenge questions generated: {CHALLENGE_QUESTION_RAW_PATH}")
    print(f"Filtered challenge questions generated: {CHALLENGE_QUESTION_FILTERED_PATH}")
    print(f"Question templates generated: {QUESTION_PATH}")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


if __name__ == "__main__":
    main()
