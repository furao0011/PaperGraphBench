from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from src.eval_turn_runner import build_eval_prompt, build_model_answer, dialogue_history_text
from src.model_client import OpenAICompatClient
from src.multimodal_question_assets import asset_context_for_prompt, question_image_paths


TEXTONLY_MODE = "text_only_no_graph"
TEXTONLY_EVAL_MODE = "text_only_no_graph_context_without_repair"
CHALLENGE_TYPES = {
    "false_premise",
    "overclaim",
    "wrong_relation",
    "unsupported_generalization",
}


def generate_textonly_question_package(
    paper_id: str,
    paper_text: str,
    client: OpenAICompatClient,
    macro_count: int,
    challenge_count: int,
    multimodal_challenge_count: int,
    multimodal_assets: list[dict],
    temperature: float,
) -> dict:
    payload = client.chat_json(
        system_prompt=_generation_system_prompt(),
        user_prompt=_generation_user_prompt(
            paper_id,
            paper_text,
            macro_count,
            challenge_count,
            multimodal_challenge_count,
            multimodal_assets,
        ),
        temperature=temperature,
    )
    return normalize_textonly_package(
        payload,
        paper_id,
        macro_count,
        challenge_count,
        multimodal_challenge_count,
        multimodal_assets,
    )


def normalize_textonly_package(
    payload: dict,
    paper_id: str,
    macro_count: int | None = None,
    challenge_count: int | None = None,
    multimodal_challenge_count: int | None = None,
    multimodal_assets: list[dict] | None = None,
) -> dict:
    if not isinstance(payload, dict):
        raise ValueError("Text-only question package must be a JSON object.")
    macros = payload.get("macro_questions")
    challenges = payload.get("challenge_questions")
    multimodal_challenges = payload.get("multimodal_challenge_questions", [])
    if not isinstance(macros, list) or not isinstance(challenges, list):
        raise ValueError("Text-only package requires macro_questions and challenge_questions lists.")
    if not isinstance(multimodal_challenges, list):
        raise ValueError("Text-only package multimodal_challenge_questions must be a list.")
    if macro_count is not None and len(macros) < macro_count:
        raise ValueError(f"Expected at least {macro_count} macro questions, got {len(macros)}.")
    if challenge_count is not None and len(challenges) < challenge_count:
        raise ValueError(f"Expected at least {challenge_count} challenge questions, got {len(challenges)}.")
    if multimodal_challenge_count is not None and len(multimodal_challenges) < multimodal_challenge_count:
        raise ValueError(
            f"Expected at least {multimodal_challenge_count} multimodal challenge questions, "
            f"got {len(multimodal_challenges)}."
        )

    normalized_macros = [
        _normalize_macro_question(question, idx)
        for idx, question in enumerate(macros[:macro_count or len(macros)], start=1)
    ]
    normalized_challenges = [
        _normalize_challenge_question(question, idx)
        for idx, question in enumerate(challenges[:challenge_count or len(challenges)], start=1)
    ]
    asset_index = {asset["asset_id"]: asset for asset in multimodal_assets or [] if asset.get("asset_id")}
    normalized_multimodal_challenges = [
        _normalize_multimodal_challenge_question(question, idx, asset_index)
        for idx, question in enumerate(
            multimodal_challenges[:multimodal_challenge_count or len(multimodal_challenges)],
            start=1,
        )
    ]
    return {
        "paper_id": paper_id,
        "generation_mode": TEXTONLY_MODE,
        "question_source": "paper_clean_text_plus_multimodal_assets_no_graph",
        "macro_questions": normalized_macros,
        "challenge_questions": normalized_challenges,
        "multimodal_challenge_questions": normalized_multimodal_challenges,
        "multimodal_assets_used": [
            _asset_prompt_payload(asset)
            for asset in multimodal_assets or []
            if asset.get("asset_id")
        ],
    }


def textonly_questions(package: dict) -> list[dict]:
    questions = []
    for idx, question in enumerate(package.get("macro_questions", []), start=1):
        item = dict(question)
        item["question_order"] = idx
        questions.append(item)
    offset = len(questions)
    for idx, question in enumerate(package.get("challenge_questions", []), start=1):
        item = dict(question)
        item["question_order"] = offset + idx
        questions.append(item)
    offset = len(questions)
    for idx, question in enumerate(package.get("multimodal_challenge_questions", []), start=1):
        item = dict(question)
        item["question_order"] = offset + idx
        questions.append(item)
    return questions


def textonly_no_repair_questions(package: dict) -> list[dict]:
    macros = [dict(question) for question in package.get("macro_questions", [])]
    text_challenges = [dict(question) for question in package.get("challenge_questions", [])]
    multimodal_challenges = [dict(question) for question in package.get("multimodal_challenge_questions", [])]
    challenges = []
    for index in range(max(len(text_challenges), len(multimodal_challenges))):
        if index < len(text_challenges):
            challenges.append(text_challenges[index])
        if index < len(multimodal_challenges):
            challenges.append(multimodal_challenges[index])
    if not macros:
        scheduled = challenges
    else:
        buckets = [[] for _ in macros]
        for index, challenge in enumerate(challenges):
            bucket_index = min(index * len(macros) // max(1, len(challenges)), len(macros) - 1)
            buckets[bucket_index].append(challenge)
        scheduled = []
        for macro, bucket in zip(macros, buckets):
            scheduled.append(macro)
            for challenge in bucket:
                challenge["scheduled_after_macro_question_id"] = macro.get("question_id")
                scheduled.append(challenge)
    for order, question in enumerate(scheduled, start=1):
        question["question_order"] = order
    return scheduled

def run_textonly_question(
    question: dict,
    paper_text: str,
    target_client: OpenAICompatClient,
    judge_client: OpenAICompatClient,
    turn_id: str,
    previous_turns: list[dict] | None = None,
    use_online_eval: bool = True,
) -> dict:
    previous_turns = previous_turns or []
    prompt = build_eval_prompt(
        paper_text=paper_text,
        dialogue_history=dialogue_history_text(previous_turns),
        question_text=question["question_text"],
        asset_context=asset_context_for_prompt(question),
    )
    answer, answer_mode = build_model_answer(
        client=target_client,
        use_online_eval=use_online_eval,
        prompt=prompt,
        target_kcs=[],
        image_paths=question_image_paths(question),
    )
    if not answer.strip():
        raise ValueError(f"Target model returned an empty answer for {question['question_id']}.")
    judge_result = judge_textonly_answer(question, answer, judge_client)
    return {
        "turn_id": turn_id,
        "question_order": question["question_order"],
        "question_id": question["question_id"],
        "question_type": question["question_type"],
        "question_text": question["question_text"],
        "model_answer": answer,
        "answer_mode": answer_mode,
        "dialogue_context_turn_count": len(previous_turns),
        "full_dialogue_context": True,
        "repair_task": False,
        "requires_multimodal_input": bool(question.get("requires_multimodal_input")),
        "asset_references": question.get("asset_references", []),
        "multimodal_input": {
            "requires_multimodal_input": bool(question.get("requires_multimodal_input")),
            "image_paths": question_image_paths(question),
            "asset_ids": [ref.get("asset_id") for ref in question.get("asset_references", []) if ref.get("asset_id")],
        },
        "judge_result": judge_result,
        "textonly_package": _question_public_payload(question),
    }


def judge_textonly_answer(question: dict, answer: str, client: OpenAICompatClient) -> dict:
    payload = client.chat_json(
        system_prompt=_judge_system_prompt(),
        user_prompt=_judge_user_prompt(question, answer),
        temperature=_judge_temperature(),
    )
    return normalize_textonly_judge_result(payload, question)


def normalize_textonly_judge_result(payload: dict, question: dict) -> dict:
    if not isinstance(payload, dict):
        raise ValueError("Text-only judge result must be a JSON object.")
    if question["question_type"] == "textonly_macro_question":
        return _normalize_macro_judge(payload, question)
    if question["question_type"] in {"textonly_challenge_question", "textonly_multimodal_challenge_question"}:
        return _normalize_challenge_judge(payload, question)
    raise ValueError(f"Unsupported text-only question_type={question.get('question_type')!r}.")


def build_textonly_report(paper_id: str, model: str, turns: list[dict], errors: list[dict]) -> dict:
    macro_metrics = _macro_metrics(turns)
    challenge_metrics = _challenge_metrics(turns)
    hallucination_metrics = _hallucination_metrics(turns)
    return {
        "paper_id": paper_id,
        "target_model": model,
        "evaluation_mode": TEXTONLY_EVAL_MODE,
        "status": "completed" if not errors else "failed",
        "ablation": {
            "graph_guided_question_generation": False,
            "full_dialogue_context": True,
            "repair_tasks_executed": False,
            "kc_coverage_metrics_computed": False,
        },
        "summary": {
            "total_turns": len(turns),
            "evaluation_status": "completed" if not errors else "failed",
            "failed_questions": len(errors),
            "question_type_counts": _question_type_counts(turns),
            "macro_textonly_metrics": macro_metrics,
            "challenge_textonly_metrics": challenge_metrics,
            "hallucination_textonly_metrics": hallucination_metrics,
            "response_length_metrics": _response_length_metrics(turns),
        },
        "errors": errors,
    }


def completed_textonly_result_exists(trajectory_path: Path, report_path: Path) -> bool:
    if not trajectory_path.exists() or not report_path.exists():
        return False
    trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    return (
        trajectory.get("evaluation_mode") == TEXTONLY_EVAL_MODE
        and report.get("evaluation_mode") == TEXTONLY_EVAL_MODE
        and report.get("status") == "completed"
    )


def load_paper_clean_text(base_dir: Path, paper_id: str, limit_env: str = "TEXTONLY_PAPER_CHAR_LIMIT") -> str:
    path = base_dir / "data" / paper_id / "paper_clean_text.md"
    if not path.exists():
        raise FileNotFoundError(f"Text-only ablation requires paper_clean_text.md: {path}")
    text = path.read_text(encoding="utf-8")
    limit = _env_nonnegative_int(limit_env, 0)
    return text[:limit] if limit else text




def load_textonly_multimodal_assets(base_dir: Path, paper_id: str, limit: int) -> list[dict]:
    if limit <= 0:
        return []
    root = base_dir / "data" / paper_id
    assets_path = root / "multimodal_assets.json"
    explanations_path = root / "multimodal_asset_explanations.json"
    if not assets_path.exists():
        return []
    assets_payload = json.loads(assets_path.read_text(encoding="utf-8"))
    explanations = _load_asset_explanations(explanations_path)
    assets = []
    for asset in assets_payload.get("assets", []):
        if not isinstance(asset, dict):
            continue
        ref = _asset_reference(
            asset,
            explanations.get(str(asset.get("asset_id") or "")),
            base_dir=base_dir,
            paper_id=paper_id,
        )
        if ref:
            assets.append(ref)
        if len(assets) >= limit:
            break
    return assets

def _generation_system_prompt() -> str:
    return (
        "You create benchmark question packages directly from the original paper text. "
        "You may use the provided figure/table asset summaries for multimodal challenge questions. "
        "Do not infer or use any graph, KC bank, edge list, thread graph, or hidden metadata. "
        "Return strict JSON only."
    )


def _generation_user_prompt(
    paper_id: str,
    paper_text: str,
    macro_count: int,
    challenge_count: int,
    multimodal_challenge_count: int,
    multimodal_assets: list[dict],
) -> str:
    assets_text = json.dumps([_asset_prompt_payload(asset) for asset in multimodal_assets], ensure_ascii=False, indent=2)
    return f"""
Create a text-only no-graph benchmark package for paper_id={paper_id!r}.

Use only this paper text:
```paper
{paper_text}
```

You may also use these extracted figure/table assets for multimodal challenge questions:
```json
{assets_text}
```

Return a JSON object with exactly these top-level keys:
- macro_questions: {macro_count} items
- challenge_questions: {challenge_count} items
- multimodal_challenge_questions: {multimodal_challenge_count} items

Each macro question must have:
- question_text: asks about a main contribution, method, evidence, or limitation of the paper
- expected_points: 3 to 5 items, each with point_id, claim, evidence

Each challenge question must have:
- challenge_type: one of false_premise, overclaim, wrong_relation, unsupported_generalization
- question_text: a question containing a trap
- forbidden_claim: the claim that the model should not accept
- expected_behavior: reject, qualify, or correct
- expected_answer: concise description of the correct behavior
- evidence: paper evidence showing why the trap is unsupported or needs qualification

Each multimodal challenge question must have all challenge fields above plus:
- asset_ids: 1 or 2 asset ids selected from the provided assets
- multimodal_dependency: why seeing the figure/table is necessary or useful

Do not include target KCs, macro ids, graph ids, edge ids, thread ids, or any graph-derived field.
"""


def _judge_system_prompt() -> str:
    return (
        "You are a conservative benchmark judge for a text-only no-graph ablation. "
        "Judge only against the provided expected points, evidence, forbidden claim, and attached figure/table assets. "
        "Return strict JSON only."
    )


def _judge_user_prompt(question: dict, answer: str) -> str:
    if question["question_type"] == "textonly_macro_question":
        rubric = {
            "question_text": question["question_text"],
            "expected_points": question["expected_points"],
        }
        schema = {
            "state": "COMPLETE | INCOMPLETE | HALLUCINATION",
            "matched_expected_point_ids": ["P1"],
            "unmatched_expected_point_ids": ["P2"],
            "hallucination_events": [
                {
                    "hallucination_type": "contradicted_evidence | fabricated_claim | overclaim",
                    "claim": "...",
                    "rationale": "...",
                }
            ],
            "rationale": "...",
        }
    else:
        rubric = {
            "question_text": question["question_text"],
            "challenge_type": question["challenge_type"],
            "forbidden_claim": question["forbidden_claim"],
            "expected_behavior": question["expected_behavior"],
            "expected_answer": question["expected_answer"],
            "evidence": question["evidence"],
            "asset_references": question.get("asset_references", []),
            "multimodal_dependency": question.get("multimodal_dependency"),
        }
        schema = {
            "state": "CHALLENGE_RESISTED | CHALLENGE_FAIL | INCOMPLETE",
            "challenge_result": {
                "failed": False,
                "resisted": True,
                "incomplete": False,
                "rationale": "...",
            },
            "hallucination_events": [
                {
                    "hallucination_type": "false_premise | overclaim | wrong_relation | unsupported_generalization",
                    "claim": "...",
                    "rationale": "...",
                }
            ],
            "rationale": "...",
        }
    return (
        "Question package rubric:\n"
        f"{json.dumps(rubric, ensure_ascii=False, indent=2)}\n\n"
        "Model answer:\n"
        f"{answer}\n\n"
        "Return JSON matching this schema:\n"
        f"{json.dumps(schema, ensure_ascii=False, indent=2)}"
    )


def _normalize_macro_question(question: dict, idx: int) -> dict:
    if not isinstance(question, dict):
        raise ValueError(f"Macro question #{idx} must be an object.")
    text = _required_str(question, "question_text", f"macro #{idx}")
    points = question.get("expected_points")
    if not isinstance(points, list) or not points:
        raise ValueError(f"Macro question #{idx} requires non-empty expected_points.")
    normalized_points = []
    for point_idx, point in enumerate(points, start=1):
        if not isinstance(point, dict):
            raise ValueError(f"Macro question #{idx} point #{point_idx} must be an object.")
        normalized_points.append(
            {
                "point_id": str(point.get("point_id") or f"P{point_idx}"),
                "claim": _required_str(point, "claim", f"macro #{idx} point #{point_idx}"),
                "evidence": _required_str(point, "evidence", f"macro #{idx} point #{point_idx}"),
            }
        )
    return {
        "question_id": f"TXT_M{idx:03d}",
        "question_type": "textonly_macro_question",
        "question_text": text,
        "expected_points": normalized_points,
        "requires_multimodal_input": False,
        "asset_references": [],
    }


def _normalize_challenge_question(question: dict, idx: int) -> dict:
    if not isinstance(question, dict):
        raise ValueError(f"Challenge question #{idx} must be an object.")
    challenge_type = _required_str(question, "challenge_type", f"challenge #{idx}").strip().lower()
    if challenge_type not in CHALLENGE_TYPES:
        raise ValueError(f"Challenge question #{idx} has invalid challenge_type={challenge_type!r}.")
    return {
        "question_id": f"TXT_C{idx:03d}",
        "question_type": "textonly_challenge_question",
        "challenge_type": challenge_type,
        "target_failure_mode": challenge_type,
        "question_text": _required_str(question, "question_text", f"challenge #{idx}"),
        "forbidden_claim": _required_str(question, "forbidden_claim", f"challenge #{idx}"),
        "expected_behavior": _required_str(question, "expected_behavior", f"challenge #{idx}"),
        "expected_answer": _required_str(question, "expected_answer", f"challenge #{idx}"),
        "evidence": _required_str(question, "evidence", f"challenge #{idx}"),
        "requires_multimodal_input": False,
        "asset_references": [],
    }


def _normalize_multimodal_challenge_question(question: dict, idx: int, asset_index: dict[str, dict]) -> dict:
    if not asset_index:
        raise ValueError("Multimodal text-only challenge requested, but no usable multimodal assets are available.")
    base = _normalize_challenge_question(question, idx)
    asset_ids = _required_asset_ids(question, idx, asset_index)
    base["question_id"] = f"TXT_MC{idx:03d}"
    base["question_type"] = "textonly_multimodal_challenge_question"
    base["requires_multimodal_input"] = True
    base["asset_references"] = [asset_index[asset_id] for asset_id in asset_ids]
    base["target_asset_ids"] = asset_ids
    base["multimodal_dependency"] = _required_str(question, "multimodal_dependency", f"multimodal challenge #{idx}")
    return base


def _normalize_macro_judge(payload: dict, question: dict) -> dict:
    expected_ids = [point["point_id"] for point in question.get("expected_points", [])]
    covered = _bounded_ids(payload.get("matched_expected_point_ids", []), expected_ids)
    missing = _bounded_ids(payload.get("unmatched_expected_point_ids", []), expected_ids)
    if not missing:
        missing = [point_id for point_id in expected_ids if point_id not in set(covered)]
    events = _normalize_hallucination_events(payload.get("hallucination_events", []))
    state = str(payload.get("state") or "").strip().upper()
    if events:
        state = "HALLUCINATION"
    elif not missing and covered:
        state = "COMPLETE"
    elif state not in {"COMPLETE", "INCOMPLETE", "HALLUCINATION"}:
        state = "INCOMPLETE"
    return {
        "state": state,
        "matched_expected_point_ids": covered,
        "unmatched_expected_point_ids": missing,
        "expected_point_match": {
            "all_expected_points_matched": state == "COMPLETE" and not missing,
            "matched_expected_point_ids": covered,
            "unmatched_expected_point_ids": missing,
        },
        "hallucination_events": events,
        "rationale": str(payload.get("rationale") or "").strip(),
    }


def _normalize_challenge_judge(payload: dict, question: dict) -> dict:
    raw_result = payload.get("challenge_result") if isinstance(payload.get("challenge_result"), dict) else {}
    failed = bool(raw_result.get("failed"))
    resisted = bool(raw_result.get("resisted"))
    incomplete = bool(raw_result.get("incomplete"))
    if not any([failed, resisted, incomplete]):
        state = str(payload.get("state") or "").strip().upper()
        failed = state == "CHALLENGE_FAIL"
        resisted = state == "CHALLENGE_RESISTED"
        incomplete = not failed and not resisted
    events = _normalize_hallucination_events(payload.get("hallucination_events", []))
    if failed and not events:
        events = [
            {
                "hallucination_type": question.get("challenge_type") or "challenge_failure",
                "claim": question.get("forbidden_claim", ""),
                "rationale": raw_result.get("rationale") or payload.get("rationale") or "",
            }
        ]
    state = "CHALLENGE_FAIL" if failed else "CHALLENGE_RESISTED" if resisted else "INCOMPLETE"
    return {
        "state": state,
        "challenge_result": {
            "failed": failed,
            "resisted": resisted,
            "incomplete": incomplete,
            "rationale": str(raw_result.get("rationale") or payload.get("rationale") or "").strip(),
        },
        "hallucination_events": events,
        "rationale": str(payload.get("rationale") or "").strip(),
    }


def _macro_metrics(turns: list[dict]) -> dict:
    macro_turns = [turn for turn in turns if turn.get("question_type") == "textonly_macro_question"]
    answer_chars = [len(str(turn.get("model_answer") or "")) for turn in macro_turns]
    hallucination_events = sum(
        len(turn.get("judge_result", {}).get("hallucination_events", []) or [])
        for turn in macro_turns
    )
    return {
        "macro_question_count": len(macro_turns),
        "hallucination_event_count": hallucination_events,
        "average_answer_chars": _average(answer_chars),
        "per_question": [
            {
                "question_id": turn.get("question_id"),
                "answer_chars": len(str(turn.get("model_answer") or "")),
                "dialogue_context_turn_count": int(turn.get("dialogue_context_turn_count") or 0),
            }
            for turn in macro_turns
        ],
    }

def _challenge_metrics(turns: list[dict]) -> dict:
    challenge_turns = [
        turn
        for turn in turns
        if turn.get("question_type") in {"textonly_challenge_question", "textonly_multimodal_challenge_question"}
    ]
    failed = resisted = incomplete = 0
    text_total = text_failed = 0
    multimodal_total = multimodal_failed = 0
    by_type: dict[str, dict[str, int]] = {}
    by_failure_mode: dict[str, dict[str, int]] = {}
    source_solver_wrong_count = 0
    source_matched_failure_count = 0
    source_solver_trial_count = 0
    source_human_review_count = 0
    challenge_hallucination_event_count = 0
    per_question = []
    for turn in challenge_turns:
        result = turn.get("judge_result", {}).get("challenge_result", {})
        is_failed = bool(result.get("failed"))
        is_resisted = bool(result.get("resisted"))
        is_incomplete = bool(result.get("incomplete"))
        source_package = turn.get("textonly_package", {})
        challenge_type = source_package.get("challenge_type") or "unknown"
        failure_mode = source_package.get("target_failure_mode") or challenge_type
        is_multimodal = bool(turn.get("requires_multimodal_input"))
        solver_summary = source_package.get("solver_trial_summary", {})
        source_solver_wrong_count += int(solver_summary.get("wrong_count") or 0)
        source_matched_failure_count += int(solver_summary.get("matched_target_failure_count") or 0)
        source_solver_trial_count += int(solver_summary.get("solver_count") or 0)
        source_human_review_count += int(bool(source_package.get("needs_human_review")))
        challenge_hallucination_event_count += len(
            turn.get("judge_result", {}).get("hallucination_events", []) or []
        )

        if is_multimodal:
            multimodal_total += 1
        else:
            text_total += 1
        if is_failed:
            failed += 1
            if is_multimodal:
                multimodal_failed += 1
            else:
                text_failed += 1
        if is_resisted:
            resisted += 1
        if is_incomplete:
            incomplete += 1

        _update_challenge_breakdown(by_type, challenge_type, is_failed, is_resisted, is_incomplete)
        _update_challenge_breakdown(by_failure_mode, failure_mode, is_failed, is_resisted, is_incomplete)
        per_question.append(
            {
                "question_id": turn.get("question_id"),
                "challenge_type": challenge_type,
                "target_failure_mode": failure_mode,
                "answer_chars": len(str(turn.get("model_answer") or "")),
                "dialogue_context_turn_count": int(turn.get("dialogue_context_turn_count") or 0),
                "requires_multimodal_input": is_multimodal,
                "target_asset_ids": source_package.get("target_asset_ids", []),
                "failed": is_failed,
                "resisted": is_resisted,
                "incomplete": is_incomplete,
            }
        )
    total = len(challenge_turns)
    return {
        "challenge_question_count": total,
        "failed_count": failed,
        "failed_rate": _safe_ratio(failed, total),
        "resisted_count": resisted,
        "resisted_rate": _safe_ratio(resisted, total),
        "incomplete_count": incomplete,
        "incomplete_rate": _safe_ratio(incomplete, total),
        "challenge_hallucination_event_count": challenge_hallucination_event_count,
        "text_challenge_count": text_total,
        "text_failed_count": text_failed,
        "text_failed_rate": _safe_ratio(text_failed, text_total),
        "multimodal_challenge_count": multimodal_total,
        "multimodal_failed_count": multimodal_failed,
        "multimodal_failed_rate": _safe_ratio(multimodal_failed, multimodal_total),
        "by_challenge_type": _finalize_challenge_breakdown(by_type),
        "by_failure_mode": _finalize_challenge_breakdown(by_failure_mode),
        "source_filter_metrics": {
            "accepted_by_challenge_loop_count": total,
            "solver_trial_count": source_solver_trial_count,
            "solver_wrong_count": source_solver_wrong_count,
            "matched_target_failure_count": source_matched_failure_count,
            "needs_human_review_count": source_human_review_count,
        },
        "per_question": per_question,
    }


def _update_challenge_breakdown(
    breakdown: dict[str, dict[str, int]],
    key: str,
    failed: bool,
    resisted: bool,
    incomplete: bool,
) -> None:
    bucket = breakdown.setdefault(
        key,
        {"question_count": 0, "failed_count": 0, "resisted_count": 0, "incomplete_count": 0},
    )
    bucket["question_count"] += 1
    bucket["failed_count"] += int(failed)
    bucket["resisted_count"] += int(resisted)
    bucket["incomplete_count"] += int(incomplete)


def _finalize_challenge_breakdown(breakdown: dict[str, dict[str, int]]) -> dict[str, dict]:
    return {
        key: {
            **counts,
            "failed_rate": _safe_ratio(counts["failed_count"], counts["question_count"]),
        }
        for key, counts in sorted(breakdown.items())
    }

def _hallucination_metrics(turns: list[dict]) -> dict:
    by_type: dict[str, int] = {}
    total = 0
    turn_count = 0
    for turn in turns:
        events = turn.get("judge_result", {}).get("hallucination_events", []) or []
        if events:
            turn_count += 1
        for event in events:
            total += 1
            htype = event.get("hallucination_type") or "unknown"
            by_type[htype] = by_type.get(htype, 0) + 1
    return {
        "hallucination_event_count": total,
        "hallucination_turn_count": turn_count,
        "hallucination_event_rate_per_turn": _safe_ratio(total, len(turns)),
        "hallucination_by_type": dict(sorted(by_type.items())),
    }


def _response_length_metrics(turns: list[dict]) -> dict:
    lengths = [len(str(turn.get("model_answer") or "")) for turn in turns]
    by_type: dict[str, list[int]] = {}
    for turn, length in zip(turns, lengths):
        question_type = str(turn.get("question_type") or "unknown")
        by_type.setdefault(question_type, []).append(length)
    return {
        "response_count": len(lengths),
        "total_answer_chars": sum(lengths),
        "average_answer_chars": _average(lengths),
        "by_question_type": {
            question_type: {
                "response_count": len(values),
                "total_answer_chars": sum(values),
                "average_answer_chars": _average(values),
            }
            for question_type, values in sorted(by_type.items())
        },
    }


def _average(values: list[int]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 4)

def _question_type_counts(turns: list[dict]) -> dict:
    counts: dict[str, int] = {}
    for turn in turns:
        qtype = turn.get("question_type") or "unknown"
        counts[qtype] = counts.get(qtype, 0) + 1
    return dict(sorted(counts.items()))


def _question_public_payload(question: dict) -> dict:
    keys = [
        "expected_points",
        "challenge_type",
        "target_failure_mode",
        "forbidden_claim",
        "expected_behavior",
        "expected_answer",
        "evidence",
        "requires_multimodal_input",
        "asset_references",
        "target_asset_ids",
        "multimodal_dependency",
        "scheduled_after_macro_question_id",
        "source_loop_question_id",
        "source_challenge_plan_id",
        "accepted_by_challenge_loop",
        "solver_trial_summary",
        "needs_human_review",
    ]
    return {key: question[key] for key in keys if key in question}




def _load_asset_explanations(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = payload.get("asset_explanations", []) if isinstance(payload, dict) else []
    return {
        str(item.get("asset_id") or ""): item
        for item in items
        if isinstance(item, dict) and str(item.get("asset_id") or "").strip()
    }


def _asset_reference(asset: dict, explanation: dict | None, *, base_dir: Path, paper_id: str) -> dict | None:
    asset_id = str(asset.get("asset_id") or "").strip()
    asset_type = str(asset.get("asset_type") or "").strip().lower()
    if not asset_id or asset_type not in {"figure", "table", "mixed"}:
        return None
    explanation = explanation or {}
    attachments = _asset_attachments(asset, base_dir=base_dir, paper_id=paper_id)
    if not attachments:
        return None
    return {
        "asset_id": asset_id,
        "asset_type": asset_type,
        "caption": asset.get("caption") or explanation.get("caption"),
        "summary": explanation.get("summary") or asset.get("nearby_context"),
        "requires_multimodal_input": True,
        "input_kind": "image" if asset_type == "figure" else "table",
        "evidence_bases": _asset_evidence_bases(explanation),
        "attachments": attachments,
        "supported_claims": explanation.get("supported_claims", [])[:5],
        "possible_misreadings": explanation.get("possible_misreadings", [])[:5],
    }


def _asset_attachments(asset: dict, *, base_dir: Path, paper_id: str) -> list[dict]:
    asset_type = str(asset.get("asset_type") or "").strip().lower()
    if asset_type == "figure":
        attachments = asset.get("attachments")
        if isinstance(attachments, list) and attachments:
            return [
                {
                    "type": str(item.get("type") or "image"),
                    "asset_id": item.get("asset_id") or asset.get("asset_id"),
                    "path": _resolve_asset_path(item.get("path"), base_dir=base_dir, paper_id=paper_id),
                    "caption": item.get("caption") or asset.get("caption"),
                }
                for item in attachments
                if isinstance(item, dict) and str(item.get("path") or "").strip()
            ]
        return [
            {
                "type": "image",
                "asset_id": asset.get("asset_id"),
                "path": _resolve_asset_path(path, base_dir=base_dir, paper_id=paper_id),
                "caption": asset.get("caption"),
            }
            for path in asset.get("image_paths", [])
            if str(path).strip()
        ]
    if asset_type in {"table", "mixed"}:
        latex = str(asset.get("normalized_latex") or "").strip()
        if not latex:
            return []
        attachments = [
            {
                "type": "table_latex",
                "asset_id": asset.get("asset_id"),
                "content": latex,
                "caption": asset.get("caption"),
            }
        ]
        if asset_type == "mixed":
            for path in asset.get("image_paths", []):
                if str(path).strip():
                    attachments.append(
                        {
                            "type": "image",
                            "asset_id": asset.get("asset_id"),
                            "path": _resolve_asset_path(path, base_dir=base_dir, paper_id=paper_id),
                            "caption": asset.get("caption"),
                        }
                    )
        return attachments
    return []


def _resolve_asset_path(raw_path: Any, *, base_dir: Path, paper_id: str) -> str:
    value = str(raw_path or "").strip()
    if not value:
        return ""
    path = Path(value)
    if path.exists():
        return str(path)
    candidates: list[Path] = []
    if path.is_absolute():
        candidates.extend(
            [
                base_dir / "rawPaper" / paper_id / "imgs" / path.name,
                base_dir / "rawPaper" / paper_id / path.name,
            ]
        )
    else:
        candidates.extend(
            [
                base_dir / value,
                base_dir / "rawPaper" / paper_id / value,
                base_dir / "rawPaper" / paper_id / "imgs" / path.name,
            ]
        )
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    raise FileNotFoundError(
        f"Cannot resolve multimodal asset file for paper_id={paper_id!r}: {value!r}. "
        f"Expected it under rawPaper/{paper_id}/imgs/ or as a valid existing path."
    )


def _asset_evidence_bases(explanation: dict) -> list[str]:
    bases = []
    supported_claims = explanation.get("supported_claims", [])
    for claim in supported_claims if isinstance(supported_claims, list) else []:
        if isinstance(claim, dict) and str(claim.get("evidence_basis") or "").strip():
            bases.append(str(claim["evidence_basis"]).strip())
    return list(dict.fromkeys(bases))[:5]


def _asset_prompt_payload(asset: dict) -> dict:
    return {
        "asset_id": asset.get("asset_id"),
        "asset_type": asset.get("asset_type"),
        "caption": asset.get("caption"),
        "summary": asset.get("summary"),
        "supported_claims": asset.get("supported_claims", []),
        "possible_misreadings": asset.get("possible_misreadings", []),
        "evidence_bases": asset.get("evidence_bases", []),
    }


def _required_asset_ids(question: dict, idx: int, asset_index: dict[str, dict]) -> list[str]:
    values = question.get("asset_ids")
    if not isinstance(values, list) or not values:
        raise ValueError(f"Multimodal challenge #{idx} requires non-empty asset_ids.")
    out = []
    seen = set()
    for value in values:
        asset_id = str(value).strip()
        if asset_id not in asset_index:
            raise ValueError(f"Multimodal challenge #{idx} references unknown asset_id={asset_id!r}.")
        if asset_id not in seen:
            out.append(asset_id)
            seen.add(asset_id)
    return out

def _normalize_hallucination_events(events: Any) -> list[dict]:
    if not isinstance(events, list):
        return []
    out = []
    for event in events:
        if not isinstance(event, dict):
            continue
        htype = str(event.get("hallucination_type") or "unknown").strip()
        out.append(
            {
                "hallucination_type": htype,
                "claim": str(event.get("claim") or "").strip(),
                "rationale": str(event.get("rationale") or "").strip(),
            }
        )
    return out


def _bounded_ids(values: Any, allowed: list[str]) -> list[str]:
    allowed_set = set(allowed)
    out = []
    seen = set()
    for value in values if isinstance(values, list) else []:
        item = str(value).strip()
        if item and item in allowed_set and item not in seen:
            out.append(item)
            seen.add(item)
    return out


def _required_str(payload: dict, key: str, context: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} requires non-empty {key}.")
    return value.strip()


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return numerator / denominator


def _env_nonnegative_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    value = int(raw)
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}.")
    return value


def _judge_temperature() -> float:
    raw = os.getenv("TEXTONLY_JUDGE_TEMPERATURE", "")
    if raw.strip() == "":
        return 0.0
    return float(raw)
