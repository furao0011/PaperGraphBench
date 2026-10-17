"""Synthetic report/trajectory pairs for offline paper-statistics tests."""

from __future__ import annotations

import json
from pathlib import Path

from statistics_inputs import report_path


def edit_json(path: Path, change) -> None:
    value = json.loads(path.read_text(encoding="utf-8"))
    change(value)
    path.write_text(json.dumps(value), encoding="utf-8")


def write_fixture(root: Path, paper="paper", model="unit-model", round="single",
                  covered=1, total=2, events=1, challenge_total=1, status="completed"):
    task = {"round": round, "paper_id": paper, "model": model}
    path = report_path(root, task)
    path.parent.mkdir(parents=True, exist_ok=True)
    targets = [f"K{index + 1}" for index in range(total)]
    coverage = {"target_kc_ids": targets, "covered_kc_ids": targets[:covered],
                "missing_kc_ids": targets[covered:], "coverage_complete": bool(total) and covered == total}
    first = {"turn_id": "T1", "question_id": "Q1", "question_type": "macro_main_question",
             "macro_id": "M1", "target_kc_ids": targets, "model_answer": "Partial macro answer.",
             "judge_result": {"state": "MAIN_PROGRESS", "coverage": coverage, "hallucination_events": []}}
    detected = [{"event_id": f"H_T2_{index + 1}", "created_at_turn": "T2", "source_question_id": "Q2",
                 "hallucination_family": "challenge_failure", "hallucination_type": "challenge_false_premise_acceptance",
                 "subtype": "false_premise", "claim": "Unsupported fixture claim", "related_kc_ids": targets[:1],
                 "status": "unresolved", "followup_turns": [], "resolved_at_turn": None,
                 "source": "judge_result.hallucination_events"} for index in range(events)]
    second = {"turn_id": "T2", "question_id": "Q2", "model_answer": "A recorded target answer.",
              "question_type": "challenge_question" if challenge_total else "review_followup",
              "requires_multimodal_input": False, "asset_references": [],
              "judge_result": {"state": "MISLED" if events else "CHALLENGE_RESISTED",
                               "hallucination_events": detected}}
    if challenge_total:
        second["judge_result"]["challenge_result"] = {"failed": bool(events)}
    turns = [first, second]
    for index in range(1, challenge_total):
        turns.append({"turn_id": f"T{index + 2}", "question_id": f"Q{index + 2}",
                      "question_type": "challenge_question", "model_answer": "Reject the premise.",
                      "judge_result": {"state": "CHALLENGE_RESISTED", "challenge_result": {"failed": False},
                                       "hallucination_events": []}})
    report = {"paper_id": paper, "target_model": model,
              "summary": {"total_turns": len(turns), "evaluation_status": status, "failed": False},
              # These runtime final-state ratios intentionally differ from initial answers.
              "coverage_metrics": {"global_kc_total": total, "global_kc_lit_count": total,
                                   "global_kc_coverage_rate": 1, "macro_completion_rate": 1},
              "macro_metrics": {"macro_total": 1, "macro_completion_rate": 1},
              "hallucination_metrics": {"hallucination_event_count": events, "hallucination_turn_count": 0,
                                         "hallucination_by_type": {"challenge_false_premise_acceptance": events} if events else {},
                                         "hallucination_by_status": {"resolved": events} if events else {},
                                         "hallucination_repair_success_rate": 1},
              "detail_completion_metrics": {"coverage_gap_count": 1, "coverage_gap_resolved_count": 1}}
    trajectory = {"paper_id": paper, "target_model": model, "turns": turns}
    path.write_text(json.dumps(report), encoding="utf-8")
    path.with_name("dialogue_trajectory.json").write_text(json.dumps(trajectory), encoding="utf-8")
    return path, task
