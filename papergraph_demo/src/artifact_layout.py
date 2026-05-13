from __future__ import annotations

import re
from pathlib import Path


FINAL_ARTIFACT_FILES = {
    "master_graph": "master_graph.json",
    "question_templates": "question_templates.json",
    "dialogue_trajectory": "dialogue_trajectory.json",
    "evaluation_report": "evaluation_report.json",
    "eval_state_graph": "eval_state_graph.json",
    "multimodal_assets": "multimodal_assets.json",
    "multimodal_asset_explanations": "multimodal_asset_explanations.json",
    "paper_clean_text": "paper_clean_text.md",
    "paper_eval_context": "paper_eval_context.md",
}

CACHE_FILES = {
    "graph_build": {
        "sections": "sections.json",
        "macro_spine": "macro_spine.json",
        "macro_spine_mermaid": "macro_spine.mmd",
        "extraction_units": "extraction_units.json",
        "kc_candidates": "kc_candidates.json",
        "kc_bank": "kc_bank.json",
        "active_kc": "active_kc.json",
        "kc_bank_reasoning_edges": "kc_bank_reasoning_edges.json",
        "edge_candidate_units": "edge_candidate_units.json",
        "edge_candidate_macro": "edge_candidate_macro.json",
        "edge_candidate_cross_macro": "edge_candidate_cross_macro.json",
        "edge_candidate_thread": "edge_candidate_thread.json",
        "verified_edges": "verified_edges.json",
        "edge_verification_log": "edge_verification_log.json",
        "edge_coverage_report": "edge_coverage_report.json",
        "reasoning_threads": "reasoning_threads.json",
        "reasoning_threads_mermaid": "reasoning_threads.mmd",
        "master_graph_mermaid": "master_graph.mmd",
        "build_graph_checkpoint": "build_graph_checkpoint.json",
    },
    "multimodal": {
        "paper_blocks": "paper_blocks.json",
        "multimodal_asset_groups": "multimodal_asset_groups.json",
        "multimodal_kc_candidates": "multimodal_kc_candidates.json",
    },
    "questions": {
        "question_generation_cache": "question_generation_cache.json",
        "challenge_plans": "challenge_plans.json",
        "challenge_questions_raw": "challenge_questions_raw.json",
        "challenge_loop_cache": "challenge_loop_cache.json",
        "challenge_loop_cache_text": "challenge_loop_cache_text.json",
        "challenge_loop_cache_multimodal": "challenge_loop_cache_multimodal.json",
        "challenge_loop_cache_multimodal_figure": "challenge_loop_cache_multimodal_figure.json",
        "challenge_loop_cache_multimodal_remaining": "challenge_loop_cache_multimodal_remaining.json",
        "challenge_solver_trials": "challenge_solver_trials.json",
        "challenge_questions_filtered": "challenge_questions_filtered.json",
        "challenge_questions_need_human_review": "challenge_questions_need_human_review.json",
        "challenge_questions_rejected": "challenge_questions_rejected.json",
        "challenge_question_generation_cache": "challenge_question_generation_cache.json",
        "thread_challenge_plans": "thread_challenge_plans.json",
        "thread_challenge_questions_raw": "thread_challenge_questions_raw.json",
        "thread_challenge_loop_cache": "thread_challenge_loop_cache.json",
        "thread_challenge_solver_trials": "thread_challenge_solver_trials.json",
        "thread_challenge_questions_filtered": "thread_challenge_questions_filtered.json",
        "thread_challenge_questions_need_human_review": "thread_challenge_questions_need_human_review.json",
        "thread_challenge_questions_rejected": "thread_challenge_questions_rejected.json",
    },
    "evaluation": {
        "evaluation_checkpoint": "evaluation_checkpoint.json",
        "claim_verification_log": "claim_verification_log.json",
        "final_state_graph": "final_state_graph.mmd",
        "final_thread_state_graph": "final_thread_state_graph.mmd",
    },
}


class PaperArtifactLayout:
    def __init__(self, base_dir: Path, paper_id: str) -> None:
        self.base_dir = base_dir
        self.paper_id = safe_paper_id(paper_id)
        self.root = paper_data_root(base_dir, self.paper_id)

    def final(self, artifact_name: str) -> Path:
        return final_artifact_path(self.base_dir, self.paper_id, artifact_name)

    def cache_dir(self, cache_name: str) -> Path:
        return cache_dir(self.base_dir, self.paper_id, cache_name)

    def cache_file(self, cache_name: str, artifact_name: str) -> Path:
        filename = CACHE_FILES.get(cache_name, {}).get(artifact_name)
        if not filename:
            raise KeyError(f"Unknown cache artifact: {cache_name}/{artifact_name}")
        return self.cache_dir(cache_name) / filename

    def rel(self, path: Path) -> str:
        return path.resolve().relative_to(self.base_dir.resolve()).as_posix()


def safe_paper_id(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return safe.strip("._-") or "Storybench"


def paper_data_root(base_dir: Path, paper_id: str) -> Path:
    return base_dir / "data" / safe_paper_id(paper_id)


def final_artifact_path(base_dir: Path, paper_id: str, artifact_name: str) -> Path:
    filename = FINAL_ARTIFACT_FILES.get(artifact_name)
    if not filename:
        raise KeyError(f"Unknown final artifact name: {artifact_name}")
    return paper_data_root(base_dir, paper_id) / filename


def cache_dir(base_dir: Path, paper_id: str, cache_name: str) -> Path:
    if not cache_name or "/" in cache_name or "\\" in cache_name:
        raise ValueError(f"Invalid cache directory name: {cache_name!r}")
    return paper_data_root(base_dir, paper_id) / "cache" / cache_name
