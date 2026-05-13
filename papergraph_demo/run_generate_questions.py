import json
import os
from pathlib import Path

from src.artifact_layout import PaperArtifactLayout
from src.challenge_loop import build_challenge_questions_loop
from src.challenge_plan_builder import build_challenge_plans
from src.config import load_settings
from src.model_client import ModelConfig, OpenAICompatClient
from src.multimodal_explainer import build_vision_client
from src.multimodal_question_assets import (
    attach_asset_references_to_challenge_plans,
    attach_asset_references_to_questions,
    load_multimodal_asset_index,
)
from src.paper_context import load_full_paper_text
from src.progress import log, span
from src.question_generator import generate_questions_cached
from src.thread_challenge_plan_builder import build_thread_challenge_plans


BASE_DIR = Path(__file__).resolve().parent
GRAPH_PATH = BASE_DIR / "data" / "graphs" / "master_graph.json"
QUESTION_PATH = BASE_DIR / "data" / "questions" / "question_templates.json"
QUESTION_CACHE_PATH = BASE_DIR / "data" / "questions" / "question_generation_cache.json"
CHALLENGE_PLAN_PATH = BASE_DIR / "data" / "questions" / "challenge_plans.json"
CHALLENGE_QUESTION_RAW_PATH = BASE_DIR / "data" / "questions" / "challenge_questions_raw.json"
CHALLENGE_LOOP_CACHE_PATH = BASE_DIR / "data" / "questions" / "challenge_loop_cache.json"
CHALLENGE_SOLVER_TRIALS_PATH = BASE_DIR / "data" / "questions" / "challenge_solver_trials.json"
CHALLENGE_QUESTION_FILTERED_PATH = BASE_DIR / "data" / "questions" / "challenge_questions_filtered.json"
CHALLENGE_QUESTION_HUMAN_REVIEW_PATH = BASE_DIR / "data" / "questions" / "challenge_questions_need_human_review.json"
CHALLENGE_QUESTION_REJECTED_PATH = BASE_DIR / "data" / "questions" / "challenge_questions_rejected.json"
THREAD_CHALLENGE_PLAN_PATH = BASE_DIR / "data" / "questions" / "thread_challenge_plans.json"
THREAD_CHALLENGE_QUESTION_RAW_PATH = BASE_DIR / "data" / "questions" / "thread_challenge_questions_raw.json"
THREAD_CHALLENGE_LOOP_CACHE_PATH = BASE_DIR / "data" / "questions" / "thread_challenge_loop_cache.json"
THREAD_CHALLENGE_SOLVER_TRIALS_PATH = BASE_DIR / "data" / "questions" / "thread_challenge_solver_trials.json"
THREAD_CHALLENGE_QUESTION_FILTERED_PATH = BASE_DIR / "data" / "questions" / "thread_challenge_questions_filtered.json"
THREAD_CHALLENGE_QUESTION_HUMAN_REVIEW_PATH = BASE_DIR / "data" / "questions" / "thread_challenge_questions_need_human_review.json"
THREAD_CHALLENGE_QUESTION_REJECTED_PATH = BASE_DIR / "data" / "questions" / "thread_challenge_questions_rejected.json"


def _apply_paper_layout_from_env() -> PaperArtifactLayout | None:
    paper_id = os.getenv("PAPER_ID", "").strip()
    if not paper_id:
        return None
    layout = PaperArtifactLayout(BASE_DIR, paper_id)
    _configure_layout_paths(layout)
    return layout


def _configure_layout_paths(layout: PaperArtifactLayout) -> None:
    global GRAPH_PATH, QUESTION_PATH, QUESTION_CACHE_PATH, CHALLENGE_PLAN_PATH
    global CHALLENGE_QUESTION_RAW_PATH, CHALLENGE_LOOP_CACHE_PATH, CHALLENGE_SOLVER_TRIALS_PATH
    global CHALLENGE_QUESTION_FILTERED_PATH, CHALLENGE_QUESTION_HUMAN_REVIEW_PATH
    global CHALLENGE_QUESTION_REJECTED_PATH
    global THREAD_CHALLENGE_PLAN_PATH, THREAD_CHALLENGE_QUESTION_RAW_PATH, THREAD_CHALLENGE_LOOP_CACHE_PATH
    global THREAD_CHALLENGE_SOLVER_TRIALS_PATH, THREAD_CHALLENGE_QUESTION_FILTERED_PATH
    global THREAD_CHALLENGE_QUESTION_HUMAN_REVIEW_PATH, THREAD_CHALLENGE_QUESTION_REJECTED_PATH

    GRAPH_PATH = layout.final("master_graph")
    QUESTION_PATH = layout.final("question_templates")
    QUESTION_CACHE_PATH = layout.cache_file("questions", "question_generation_cache")
    CHALLENGE_PLAN_PATH = layout.cache_file("questions", "challenge_plans")
    CHALLENGE_QUESTION_RAW_PATH = layout.cache_file("questions", "challenge_questions_raw")
    CHALLENGE_LOOP_CACHE_PATH = layout.cache_file("questions", "challenge_loop_cache")
    CHALLENGE_SOLVER_TRIALS_PATH = layout.cache_file("questions", "challenge_solver_trials")
    CHALLENGE_QUESTION_FILTERED_PATH = layout.cache_file("questions", "challenge_questions_filtered")
    CHALLENGE_QUESTION_HUMAN_REVIEW_PATH = layout.cache_file("questions", "challenge_questions_need_human_review")
    CHALLENGE_QUESTION_REJECTED_PATH = layout.cache_file("questions", "challenge_questions_rejected")
    THREAD_CHALLENGE_PLAN_PATH = layout.cache_file("questions", "thread_challenge_plans")
    THREAD_CHALLENGE_QUESTION_RAW_PATH = layout.cache_file("questions", "thread_challenge_questions_raw")
    THREAD_CHALLENGE_LOOP_CACHE_PATH = layout.cache_file("questions", "thread_challenge_loop_cache")
    THREAD_CHALLENGE_SOLVER_TRIALS_PATH = layout.cache_file("questions", "thread_challenge_solver_trials")
    THREAD_CHALLENGE_QUESTION_FILTERED_PATH = layout.cache_file("questions", "thread_challenge_questions_filtered")
    THREAD_CHALLENGE_QUESTION_HUMAN_REVIEW_PATH = layout.cache_file("questions", "thread_challenge_questions_need_human_review")
    THREAD_CHALLENGE_QUESTION_REJECTED_PATH = layout.cache_file("questions", "thread_challenge_questions_rejected")


def _rel(path: Path) -> str:
    return path.resolve().relative_to(BASE_DIR.resolve()).as_posix()


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_path(name: str, default: Path) -> Path:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return Path(raw.strip())


def _build_multimodal_challenge_questions_with_quotas(
    multimodal_challenge_plans: dict,
    client: OpenAICompatClient,
    paper_text: str,
    cache_path: Path,
    resume: bool,
    restart: bool,
    target_count: int,
    figure_min: int,
    solver_client: OpenAICompatClient,
) -> dict:
    figure_plans = _challenge_plan_asset_subset(multimodal_challenge_plans, "figure")
    if len(figure_plans.get("challenge_plans", [])) < figure_min:
        raise RuntimeError(
            f"Multimodal challenge quota requires at least {figure_min} figure plans, "
            f"but only {len(figure_plans.get('challenge_plans', []))} are available."
        )
    figure_result = build_challenge_questions_loop(
        challenge_plans=figure_plans,
        client=client,
        paper_text=paper_text,
        cache_path=_pool_cache_path(cache_path, "multimodal_figure"),
        resume=resume,
        restart=restart,
        target_count=figure_min,
        solver_client=solver_client,
        question_id_prefix="CHQF",
    )
    accepted_figure_count = _accepted_asset_type_count(figure_result, "figure")
    if accepted_figure_count < figure_min:
        raise RuntimeError(
            f"Multimodal challenge figure quota unmet: required {figure_min}, accepted {accepted_figure_count}."
        )

    remaining_target = max(0, target_count - len(figure_result.get("challenge_questions_filtered", [])))
    if remaining_target == 0:
        return _merge_named_challenge_loop_results(
            multimodal_challenge_plans,
            {"figure": figure_result},
        )

    other_plans = _challenge_plan_excluding_plan_ids(
        multimodal_challenge_plans,
        {
            question.get("source_plan_id")
            for question in figure_result.get("challenge_questions_filtered", [])
            if question.get("source_plan_id")
        },
    )
    if not other_plans.get("challenge_plans"):
        raise RuntimeError("Multimodal challenge quota requires more accepted questions, but no remaining plans are available.")
    other_result = build_challenge_questions_loop(
        challenge_plans=other_plans,
        client=client,
        paper_text=paper_text,
        cache_path=_pool_cache_path(cache_path, "multimodal_remaining"),
        resume=resume,
        restart=restart,
        target_count=remaining_target,
        solver_client=solver_client,
        question_id_prefix="CHQM",
    )
    return _merge_named_challenge_loop_results(
        multimodal_challenge_plans,
        {
            "figure": figure_result,
            "remaining": other_result,
        },
    )


def _validate_multimodal_challenge_quotas(result: dict, target_count: int, figure_min: int) -> None:
    accepted = result.get("challenge_questions_filtered", [])
    if len(accepted) < target_count:
        raise RuntimeError(
            f"Multimodal challenge quota unmet: required {target_count} accepted questions, got {len(accepted)}."
        )
    figure_count = _accepted_asset_type_count(result, "figure")
    if figure_count < figure_min:
        raise RuntimeError(
            f"Multimodal challenge figure quota unmet: required {figure_min}, accepted {figure_count}."
        )


def main() -> None:
    settings = load_settings(BASE_DIR.parent)
    layout = _apply_paper_layout_from_env()
    graph_override = os.getenv("PAPERGRAPH_GRAPH_PATH", "").strip()
    if graph_override:
        global GRAPH_PATH
        GRAPH_PATH = Path(graph_override)
    if not GRAPH_PATH.exists():
        raise FileNotFoundError(f"Master graph not found: {GRAPH_PATH}")

    allow_offline_fallback = _env_bool("ALLOW_OFFLINE_FALLBACK")
    resume = _env_bool("PAPERGRAPH_RESUME") or _env_bool("QUESTION_RESUME")
    restart = _env_bool("PAPERGRAPH_RESTART") or _env_bool("QUESTION_RESTART")
    cache_path = _env_path("QUESTION_CACHE_PATH", QUESTION_CACHE_PATH)
    challenge_loop_cache_path = _env_path("CHALLENGE_LOOP_CACHE_PATH", CHALLENGE_LOOP_CACHE_PATH)
    client = OpenAICompatClient(
        ModelConfig(settings.api_key, settings.base_url, settings.llm_model)
    )
    if not client.is_ready() and not allow_offline_fallback:
        raise RuntimeError("Online question generation requires API_KEY, BASE_URL, and LLM_MODEL. Set ALLOW_OFFLINE_FALLBACK=true only for local debugging.")
    log("loading master graph", path=GRAPH_PATH)
    graph = json.loads(GRAPH_PATH.read_text(encoding="utf-8"))
    asset_index = load_multimodal_asset_index(graph, BASE_DIR)
    by_kc = {
        kc["kc_id"]: kc
        for kc in graph.get("kc_nodes", [])
        if kc.get("kc_id")
    }
    with span("load clean Storybench text for challenge solver trials"):
        paper_text = load_full_paper_text(graph, BASE_DIR, prefer_evaluation_context=False)
    log(
        "master graph loaded",
        kcs=len(graph.get("kc_nodes", [])),
        macros=len(graph.get("macro_nodes", [])),
        paths=len(graph.get("reasoning_paths", [])),
        paper_chars=len(paper_text),
    )
    challenge_plans = attach_asset_references_to_challenge_plans(
        build_challenge_plans(graph),
        by_kc,
        asset_index,
    )
    CHALLENGE_PLAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    CHALLENGE_PLAN_PATH.write_text(json.dumps(challenge_plans, ensure_ascii=False, indent=2), encoding="utf-8")
    log(
        "challenge plans generated",
        path=CHALLENGE_PLAN_PATH,
        plans=challenge_plans.get("summary", {}).get("plan_count", 0),
    )
    text_challenge_plans = _challenge_plan_subset(challenge_plans, "text")
    multimodal_challenge_plans = _challenge_plan_subset(challenge_plans, "multimodal")
    with span("build text challenge questions by loop"):
        text_loop_result = build_challenge_questions_loop(
            challenge_plans=text_challenge_plans,
            client=client,
            paper_text=paper_text,
            cache_path=_pool_cache_path(challenge_loop_cache_path, "text"),
            resume=resume,
            restart=restart,
        )
    multimodal_loop_result = _empty_challenge_loop_result(multimodal_challenge_plans)
    if multimodal_challenge_plans.get("challenge_plans"):
        vision_client = build_vision_client(
            embed_api_key=settings.embed_api_key,
            vision_api_key=settings.vision_api_key,
            vision_base_url=settings.vision_base_url,
            vision_model=settings.vision_model,
        )
        multimodal_target = _env_positive_int(
            "MULTIMODAL_CHALLENGE_ACCEPT_TARGET",
            min(5, len(multimodal_challenge_plans["challenge_plans"])),
        )
        figure_min = _env_positive_int("MULTIMODAL_CHALLENGE_FIGURE_ACCEPT_MIN", 2)
        with span("build multimodal challenge questions by loop"):
            multimodal_loop_result = _build_multimodal_challenge_questions_with_quotas(
                multimodal_challenge_plans=multimodal_challenge_plans,
                client=client,
                paper_text=paper_text,
                cache_path=challenge_loop_cache_path,
                resume=resume,
                restart=restart,
                target_count=multimodal_target,
                figure_min=figure_min,
                solver_client=vision_client,
            )
        _validate_multimodal_challenge_quotas(
            multimodal_loop_result,
            target_count=multimodal_target,
            figure_min=figure_min,
        )
    challenge_loop_result = _merge_challenge_loop_results(
        challenge_plans,
        text_loop_result,
        multimodal_loop_result,
    )
    challenge_loop_result["challenge_questions_raw"] = attach_asset_references_to_questions(
        challenge_loop_result["challenge_questions_raw"],
        by_kc,
        asset_index,
    )
    challenge_loop_result["challenge_questions_filtered"] = attach_asset_references_to_questions(
        challenge_loop_result["challenge_questions_filtered"],
        by_kc,
        asset_index,
    )
    challenge_loop_result["challenge_questions_need_human_review"] = attach_asset_references_to_questions(
        challenge_loop_result["challenge_questions_need_human_review"],
        by_kc,
        asset_index,
    )
    challenge_loop_result["challenge_questions_rejected"] = attach_asset_references_to_questions(
        challenge_loop_result["challenge_questions_rejected"],
        by_kc,
        asset_index,
    )
    raw_challenge_questions = {
        "paper_id": challenge_loop_result["paper_id"],
        "schema_version": challenge_loop_result["schema_version"],
        "source_challenge_plan_signature": challenge_loop_result["source_challenge_plan_signature"],
        "challenge_loop_signature": challenge_loop_result["challenge_loop_signature"],
        "challenge_questions_raw": challenge_loop_result["challenge_questions_raw"],
        "summary": {
            "raw_question_count": challenge_loop_result["summary"]["raw_question_count"],
            "by_type": challenge_loop_result["summary"].get("by_type", {}),
            "by_modality_pool": challenge_loop_result["summary"].get("by_modality_pool", {}),
        },
    }
    _write_json(CHALLENGE_QUESTION_RAW_PATH, raw_challenge_questions)
    _write_json(
        CHALLENGE_SOLVER_TRIALS_PATH,
        {
            "paper_id": challenge_loop_result["paper_id"],
            "schema_version": challenge_loop_result["schema_version"],
            "challenge_loop_signature": challenge_loop_result["challenge_loop_signature"],
            "solver_configs": challenge_loop_result["solver_configs"],
            "solver_trials": challenge_loop_result["solver_trials"],
            "summary": challenge_loop_result["summary"],
        },
    )
    _write_json(
        CHALLENGE_QUESTION_FILTERED_PATH,
        {
            "paper_id": challenge_loop_result["paper_id"],
            "schema_version": challenge_loop_result["schema_version"],
            "challenge_questions_filtered": challenge_loop_result["challenge_questions_filtered"],
            "summary": challenge_loop_result["summary"],
        },
    )
    _write_json(
        CHALLENGE_QUESTION_HUMAN_REVIEW_PATH,
        {
            "paper_id": challenge_loop_result["paper_id"],
            "schema_version": challenge_loop_result["schema_version"],
            "challenge_questions_need_human_review": challenge_loop_result["challenge_questions_need_human_review"],
            "summary": challenge_loop_result["summary"],
        },
    )
    _write_json(
        CHALLENGE_QUESTION_REJECTED_PATH,
        {
            "paper_id": challenge_loop_result["paper_id"],
            "schema_version": challenge_loop_result["schema_version"],
            "challenge_questions_rejected": challenge_loop_result["challenge_questions_rejected"],
            "blacklisted_plan_ids": challenge_loop_result["blacklisted_plan_ids"],
            "loop_events": challenge_loop_result["loop_events"],
            "summary": challenge_loop_result["summary"],
        },
    )
    log(
        "challenge loop complete",
        filtered=challenge_loop_result["summary"]["filtered_count"],
        human_review=challenge_loop_result["summary"]["human_review_count"],
        rejected=challenge_loop_result["summary"]["rejected_count"],
        stop_reason=challenge_loop_result["summary"]["stop_reason"],
    )
    thread_challenge_plans = {
        "paper_id": graph.get("paper_id", "unknown"),
        "schema_version": "v1",
        "challenge_scope": "thread",
        "challenge_plans": [],
        "summary": {"plan_count": 0},
    }
    thread_challenge_loop_result = _empty_challenge_loop_result(thread_challenge_plans)
    if _env_bool("THREAD_CHALLENGE_ENABLED", True):
        thread_challenge_plans = build_thread_challenge_plans(graph, asset_index=asset_index)
        thread_challenge_plans = attach_asset_references_to_challenge_plans(
            thread_challenge_plans,
            by_kc,
            asset_index,
        )
        _write_json(THREAD_CHALLENGE_PLAN_PATH, thread_challenge_plans)
        log(
            "thread challenge plans generated",
            path=THREAD_CHALLENGE_PLAN_PATH,
            plans=thread_challenge_plans.get("summary", {}).get("plan_count", 0),
        )
        if thread_challenge_plans.get("challenge_plans"):
            thread_target = _env_positive_int(
                "THREAD_CHALLENGE_ACCEPT_TARGET",
                min(8, len(thread_challenge_plans["challenge_plans"])),
            )
            with span("build thread challenge questions by loop"):
                thread_challenge_loop_result = build_challenge_questions_loop(
                    challenge_plans=thread_challenge_plans,
                    client=client,
                    paper_text=paper_text,
                    cache_path=THREAD_CHALLENGE_LOOP_CACHE_PATH,
                    resume=resume,
                    restart=restart,
                    target_count=thread_target,
                    question_id_prefix="TCQ",
                )
            thread_challenge_loop_result["challenge_questions_raw"] = attach_asset_references_to_questions(
                thread_challenge_loop_result["challenge_questions_raw"],
                by_kc,
                asset_index,
            )
            thread_challenge_loop_result["challenge_questions_filtered"] = attach_asset_references_to_questions(
                thread_challenge_loop_result["challenge_questions_filtered"],
                by_kc,
                asset_index,
            )
            thread_challenge_loop_result["challenge_questions_need_human_review"] = attach_asset_references_to_questions(
                thread_challenge_loop_result["challenge_questions_need_human_review"],
                by_kc,
                asset_index,
            )
            thread_challenge_loop_result["challenge_questions_rejected"] = attach_asset_references_to_questions(
                thread_challenge_loop_result["challenge_questions_rejected"],
                by_kc,
                asset_index,
            )
        _write_thread_challenge_outputs(thread_challenge_loop_result)
        log(
            "thread challenge loop complete",
            filtered=thread_challenge_loop_result["summary"]["filtered_count"],
            human_review=thread_challenge_loop_result["summary"]["human_review_count"],
            rejected=thread_challenge_loop_result["summary"]["rejected_count"],
            stop_reason=thread_challenge_loop_result["summary"]["stop_reason"],
        )
    else:
        _write_json(THREAD_CHALLENGE_PLAN_PATH, thread_challenge_plans)
        _write_thread_challenge_outputs(thread_challenge_loop_result)
        log("thread challenge generation skipped", env="THREAD_CHALLENGE_ENABLED")
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
            bundle["macro_main_questions"] = attach_asset_references_to_questions(
                bundle["macro_main_questions"],
                by_kc,
                asset_index,
            )
            bundle["thread_question_seeds"] = attach_asset_references_to_questions(
                bundle["thread_question_seeds"],
                by_kc,
                asset_index,
            )
            bundle["main_questions"] = attach_asset_references_to_questions(
                bundle["main_questions"],
                by_kc,
                asset_index,
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
        "challenge_plans_path": _rel(CHALLENGE_PLAN_PATH),
        "challenge_plan_summary": challenge_plans.get("summary", {}),
        "challenge_questions_raw_path": _rel(CHALLENGE_QUESTION_RAW_PATH),
        "challenge_question_raw_summary": raw_challenge_questions.get("summary", {}),
        "challenge_questions_filtered_path": _rel(CHALLENGE_QUESTION_FILTERED_PATH),
        "challenge_solver_trials_path": _rel(CHALLENGE_SOLVER_TRIALS_PATH),
        "challenge_filter_summary": challenge_loop_result.get("summary", {}),
        "challenge_questions": challenge_loop_result["challenge_questions_filtered"],
        "thread_challenge_plans_path": _rel(THREAD_CHALLENGE_PLAN_PATH),
        "thread_challenge_plan_summary": thread_challenge_plans.get("summary", {}),
        "thread_challenge_questions_raw_path": _rel(THREAD_CHALLENGE_QUESTION_RAW_PATH),
        "thread_challenge_questions_filtered_path": _rel(THREAD_CHALLENGE_QUESTION_FILTERED_PATH),
        "thread_challenge_solver_trials_path": _rel(THREAD_CHALLENGE_SOLVER_TRIALS_PATH),
        "thread_challenge_filter_summary": thread_challenge_loop_result.get("summary", {}),
        "thread_challenge_questions": thread_challenge_loop_result["challenge_questions_filtered"],
        "challenge_scheduler_config": {
            "macro_level_enabled": True,
            "thread_level_enabled": True,
            "thread_challenge_enabled": _env_bool("THREAD_CHALLENGE_ENABLED", True),
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
    print(f"Thread challenge plans generated: {THREAD_CHALLENGE_PLAN_PATH}")
    print(f"Filtered thread challenge questions generated: {THREAD_CHALLENGE_QUESTION_FILTERED_PATH}")
    print(f"Question templates generated: {QUESTION_PATH}")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _write_thread_challenge_outputs(loop_result: dict) -> None:
    _write_json(
        THREAD_CHALLENGE_QUESTION_RAW_PATH,
        {
            "paper_id": loop_result["paper_id"],
            "schema_version": loop_result["schema_version"],
            "source_challenge_plan_signature": loop_result["source_challenge_plan_signature"],
            "challenge_loop_signature": loop_result["challenge_loop_signature"],
            "thread_challenge_questions_raw": loop_result["challenge_questions_raw"],
            "summary": {
                "raw_question_count": loop_result["summary"]["raw_question_count"],
                "by_type": loop_result["summary"].get("by_type", {}),
                "by_modality_pool": loop_result["summary"].get("by_modality_pool", {}),
            },
        },
    )
    _write_json(
        THREAD_CHALLENGE_SOLVER_TRIALS_PATH,
        {
            "paper_id": loop_result["paper_id"],
            "schema_version": loop_result["schema_version"],
            "challenge_loop_signature": loop_result["challenge_loop_signature"],
            "solver_configs": loop_result["solver_configs"],
            "solver_trials": loop_result["solver_trials"],
            "summary": loop_result["summary"],
        },
    )
    _write_json(
        THREAD_CHALLENGE_QUESTION_FILTERED_PATH,
        {
            "paper_id": loop_result["paper_id"],
            "schema_version": loop_result["schema_version"],
            "thread_challenge_questions_filtered": loop_result["challenge_questions_filtered"],
            "summary": loop_result["summary"],
        },
    )
    _write_json(
        THREAD_CHALLENGE_QUESTION_HUMAN_REVIEW_PATH,
        {
            "paper_id": loop_result["paper_id"],
            "schema_version": loop_result["schema_version"],
            "thread_challenge_questions_need_human_review": loop_result["challenge_questions_need_human_review"],
            "summary": loop_result["summary"],
        },
    )
    _write_json(
        THREAD_CHALLENGE_QUESTION_REJECTED_PATH,
        {
            "paper_id": loop_result["paper_id"],
            "schema_version": loop_result["schema_version"],
            "thread_challenge_questions_rejected": loop_result["challenge_questions_rejected"],
            "blacklisted_plan_ids": loop_result["blacklisted_plan_ids"],
            "loop_events": loop_result["loop_events"],
            "summary": loop_result["summary"],
        },
    )


def _challenge_plan_subset(challenge_plans: dict, pool: str) -> dict:
    plans = [
        plan
        for plan in challenge_plans.get("challenge_plans", [])
        if str(plan.get("modality_pool") or plan.get("metadata", {}).get("modality_pool") or "text") == pool
    ]
    return {
        "paper_id": challenge_plans.get("paper_id", "unknown"),
        "schema_version": challenge_plans.get("schema_version", "v2"),
        "plan_builder": challenge_plans.get("plan_builder", ""),
        "source_graph_signature": challenge_plans.get("source_graph_signature"),
        "modality_pool": pool,
        "challenge_plans": plans,
        "summary": {
            "plan_count": len(plans),
            "by_type": _count_by_type(plans),
            "by_modality_pool": {pool: len(plans)},
        },
    }


def _challenge_plan_asset_subset(challenge_plans: dict, asset_type: str) -> dict:
    plans = [
        plan
        for plan in challenge_plans.get("challenge_plans", [])
        if asset_type in _challenge_plan_asset_types(plan)
    ]
    return _challenge_plan_subset_from_list(challenge_plans, plans, f"{challenge_plans.get('modality_pool', 'multimodal')}_{asset_type}")


def _challenge_plan_excluding_plan_ids(challenge_plans: dict, excluded_plan_ids: set[str]) -> dict:
    plans = [
        plan
        for plan in challenge_plans.get("challenge_plans", [])
        if plan.get("challenge_plan_id") not in excluded_plan_ids
    ]
    return _challenge_plan_subset_from_list(challenge_plans, plans, challenge_plans.get("modality_pool", "multimodal"))


def _challenge_plan_subset_from_list(challenge_plans: dict, plans: list[dict], pool: str) -> dict:
    return {
        "paper_id": challenge_plans.get("paper_id", "unknown"),
        "schema_version": challenge_plans.get("schema_version", "v2"),
        "plan_builder": challenge_plans.get("plan_builder", ""),
        "source_graph_signature": challenge_plans.get("source_graph_signature"),
        "modality_pool": pool,
        "challenge_plans": plans,
        "summary": {
            "plan_count": len(plans),
            "by_type": _count_by_type(plans),
            "by_modality_pool": _count_by_modality_pool(plans),
            "by_asset_type": _count_plan_asset_types(plans),
        },
    }


def _pool_cache_path(path: Path, pool: str) -> Path:
    return path.with_name(f"{path.stem}_{pool}{path.suffix}")


def _empty_challenge_loop_result(challenge_plans: dict) -> dict:
    return {
        "paper_id": challenge_plans.get("paper_id", "unknown"),
        "schema_version": "v2",
        "source_challenge_plan_signature": {
            "paper_id": challenge_plans.get("paper_id", "unknown"),
            "schema_version": challenge_plans.get("schema_version"),
            "source_graph_signature": challenge_plans.get("source_graph_signature"),
            "plan_ids": [],
            "plan_sources": [],
        },
        "challenge_loop_signature": {},
        "solver_configs": [],
        "plan_order": [],
        "blacklisted_plan_ids": [],
        "challenge_questions_raw": [],
        "solver_trials": [],
        "challenge_questions_filtered": [],
        "challenge_questions_need_human_review": [],
        "challenge_questions_rejected": [],
        "loop_events": [],
        "summary": {
            "plan_pool_count": 0,
            "target_count": 0,
            "max_attempts_per_plan": 0,
            "raw_question_count": 0,
            "solver_trial_count": 0,
            "filtered_count": 0,
            "human_review_count": 0,
            "rejected_count": 0,
            "blacklisted_plan_count": 0,
            "by_type": {},
            "by_modality_pool": {},
            "stop_reason": "no_plans",
        },
    }


def _merge_challenge_loop_results(challenge_plans: dict, text_result: dict, multimodal_result: dict) -> dict:
    raw_questions = text_result["challenge_questions_raw"] + multimodal_result["challenge_questions_raw"]
    filtered = text_result["challenge_questions_filtered"] + multimodal_result["challenge_questions_filtered"]
    human_review = text_result["challenge_questions_need_human_review"] + multimodal_result["challenge_questions_need_human_review"]
    rejected = text_result["challenge_questions_rejected"] + multimodal_result["challenge_questions_rejected"]
    solver_trials = text_result["solver_trials"] + multimodal_result["solver_trials"]
    return {
        "paper_id": challenge_plans.get("paper_id", "unknown"),
        "schema_version": "v2",
        "source_challenge_plan_signature": {
            "text": text_result.get("source_challenge_plan_signature"),
            "multimodal": multimodal_result.get("source_challenge_plan_signature"),
        },
        "challenge_loop_signature": {
            "text": text_result.get("challenge_loop_signature"),
            "multimodal": multimodal_result.get("challenge_loop_signature"),
        },
        "solver_configs": text_result.get("solver_configs") or multimodal_result.get("solver_configs") or [],
        "plan_order": text_result.get("plan_order", []) + multimodal_result.get("plan_order", []),
        "blacklisted_plan_ids": text_result.get("blacklisted_plan_ids", []) + multimodal_result.get("blacklisted_plan_ids", []),
        "challenge_questions_raw": raw_questions,
        "solver_trials": solver_trials,
        "challenge_questions_filtered": filtered,
        "challenge_questions_need_human_review": human_review,
        "challenge_questions_rejected": rejected,
        "loop_events": text_result.get("loop_events", []) + multimodal_result.get("loop_events", []),
        "summary": {
            "plan_pool_count": len(challenge_plans.get("challenge_plans", [])),
            "target_count": int(text_result["summary"].get("target_count", 0)) + int(multimodal_result["summary"].get("target_count", 0)),
            "max_attempts_per_plan": max(
                int(text_result["summary"].get("max_attempts_per_plan", 0)),
                int(multimodal_result["summary"].get("max_attempts_per_plan", 0)),
            ),
            "raw_question_count": len(raw_questions),
            "solver_trial_count": len(solver_trials),
            "filtered_count": len(filtered),
            "human_review_count": len(human_review),
            "rejected_count": len(rejected),
            "blacklisted_plan_count": len(text_result.get("blacklisted_plan_ids", [])) + len(multimodal_result.get("blacklisted_plan_ids", [])),
            "by_type": _count_by_type(filtered),
            "by_modality_pool": _count_by_modality_pool(filtered),
            "stop_reason": {
                "text": text_result["summary"].get("stop_reason"),
                "multimodal": multimodal_result["summary"].get("stop_reason"),
            },
        },
    }


def _merge_named_challenge_loop_results(challenge_plans: dict, results: dict[str, dict]) -> dict:
    ordered_results = list(results.values())
    raw_questions = [item for result in ordered_results for item in result.get("challenge_questions_raw", [])]
    filtered = [item for result in ordered_results for item in result.get("challenge_questions_filtered", [])]
    human_review = [item for result in ordered_results for item in result.get("challenge_questions_need_human_review", [])]
    rejected = [item for result in ordered_results for item in result.get("challenge_questions_rejected", [])]
    solver_trials = [item for result in ordered_results for item in result.get("solver_trials", [])]
    return {
        "paper_id": challenge_plans.get("paper_id", "unknown"),
        "schema_version": "v2",
        "source_challenge_plan_signature": {
            key: result.get("source_challenge_plan_signature")
            for key, result in results.items()
        },
        "challenge_loop_signature": {
            key: result.get("challenge_loop_signature")
            for key, result in results.items()
        },
        "solver_configs": next((result.get("solver_configs") for result in ordered_results if result.get("solver_configs")), []),
        "plan_order": [item for result in ordered_results for item in result.get("plan_order", [])],
        "blacklisted_plan_ids": [item for result in ordered_results for item in result.get("blacklisted_plan_ids", [])],
        "challenge_questions_raw": raw_questions,
        "solver_trials": solver_trials,
        "challenge_questions_filtered": filtered,
        "challenge_questions_need_human_review": human_review,
        "challenge_questions_rejected": rejected,
        "loop_events": [item for result in ordered_results for item in result.get("loop_events", [])],
        "summary": {
            "plan_pool_count": len(challenge_plans.get("challenge_plans", [])),
            "target_count": sum(int(result.get("summary", {}).get("target_count", 0)) for result in ordered_results),
            "max_attempts_per_plan": max(
                [int(result.get("summary", {}).get("max_attempts_per_plan", 0)) for result in ordered_results] or [0]
            ),
            "raw_question_count": len(raw_questions),
            "solver_trial_count": len(solver_trials),
            "filtered_count": len(filtered),
            "human_review_count": len(human_review),
            "rejected_count": len(rejected),
            "blacklisted_plan_count": sum(len(result.get("blacklisted_plan_ids", [])) for result in ordered_results),
            "by_type": _count_by_type(filtered),
            "by_modality_pool": _count_by_modality_pool(filtered),
            "by_asset_type": _count_question_asset_types(filtered),
            "stop_reason": {
                key: result.get("summary", {}).get("stop_reason")
                for key, result in results.items()
            },
        },
    }


def _count_by_type(items: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        key = str(item.get("challenge_type") or "unknown")
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _count_by_modality_pool(items: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        key = str(item.get("modality_pool") or "text")
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _count_plan_asset_types(plans: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for plan in plans:
        for asset_type in _challenge_plan_asset_types(plan):
            counts[asset_type] = counts.get(asset_type, 0) + 1
    return dict(sorted(counts.items()))


def _count_question_asset_types(questions: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for question in questions:
        for asset_type in _challenge_question_asset_types(question):
            counts[asset_type] = counts.get(asset_type, 0) + 1
    return dict(sorted(counts.items()))


def _accepted_asset_type_count(result: dict, asset_type: str) -> int:
    return sum(
        1
        for question in result.get("challenge_questions_filtered", [])
        if asset_type in _challenge_question_asset_types(question)
    )


def _challenge_plan_asset_types(plan: dict) -> set[str]:
    return {
        str(ref.get("asset_type") or "").strip().lower()
        for ref in plan.get("metadata", {}).get("asset_references", [])
        if str(ref.get("asset_type") or "").strip()
    }


def _challenge_question_asset_types(question: dict) -> set[str]:
    return {
        str(ref.get("asset_type") or "").strip().lower()
        for ref in question.get("asset_references", [])
        if str(ref.get("asset_type") or "").strip()
    }


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


if __name__ == "__main__":
    main()
