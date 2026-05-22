from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any


QUESTION_TYPE_LABELS = {
    "macro_main_question": "Macro",
    "challenge_question": "Challenge",
    "detail_followup": "Detail",
    "hallucination_followup": "Hallu repair",
    "review_followup": "Review",
    "thread_premise_question": "Thread premise",
    "thread_evidence_question": "Thread evidence",
    "thread_bridge_question": "Thread bridge",
    "thread_review_question": "Thread review",
}


@dataclass
class PaperArtifacts:
    paper_dir: Path
    trajectory_path: Path
    report_path: Path | None
    state_path: Path | None
    graph_path: Path | None
    question_path: Path | None

    @property
    def paper_key(self) -> str:
        return self.paper_dir.name


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "tables").mkdir(exist_ok=True)
    (out_dir / "figures").mkdir(exist_ok=True)

    artifacts = discover_artifacts(data_root)
    if not artifacts:
        raise SystemExit(f"No dialogue_trajectory.json files found under {data_root}")

    bundle = analyze_all(artifacts)
    write_tables(bundle, out_dir / "tables")
    write_summary(bundle, out_dir / "summary.md")
    write_json(out_dir / "aggregate_metrics.json", bundle["aggregate"])
    write_figures(bundle, out_dir / "figures")
    print(f"Analyzed {len(artifacts)} papers.")
    print(f"Results written to: {out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze PaperGraph evaluation trajectories and produce experiment tables/figures."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("papergraph_demo") / "data",
        help="Root directory containing per-paper evaluation artifacts.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("EMNLP2026") / "experiment_results",
        help="Output directory for CSV/JSON/Markdown/SVG results.",
    )
    return parser.parse_args()


def discover_artifacts(data_root: Path) -> list[PaperArtifacts]:
    artifacts = []
    for trajectory_path in sorted(data_root.rglob("dialogue_trajectory.json")):
        paper_dir = trajectory_path.parent
        artifacts.append(
            PaperArtifacts(
                paper_dir=paper_dir,
                trajectory_path=trajectory_path,
                report_path=maybe(paper_dir / "evaluation_report.json"),
                state_path=maybe(paper_dir / "eval_state_graph.json"),
                graph_path=maybe(paper_dir / "master_graph.json"),
                question_path=maybe(paper_dir / "question_templates.json"),
            )
        )
    return artifacts


def maybe(path: Path) -> Path | None:
    return path if path.exists() else None


def analyze_all(artifacts: list[PaperArtifacts]) -> dict[str, Any]:
    per_paper = []
    turns_table = []
    event_table = []
    challenge_table = []
    thread_table = []
    repair_table = []
    macro_table = []
    kc_table = []
    timeseries_table = []
    report_metrics = []
    benchmark_artifacts = []

    for artifact in artifacts:
        trajectory = read_json(artifact.trajectory_path)
        report = read_json(artifact.report_path) if artifact.report_path else {}
        state = read_json(artifact.state_path) if artifact.state_path else {}
        graph = read_json(artifact.graph_path) if artifact.graph_path else {}
        questions = read_json(artifact.question_path) if artifact.question_path else {}

        paper = analyze_paper(artifact, trajectory, report, state, graph, questions)
        per_paper.append(paper["metrics"])
        turns_table.extend(paper["turn_rows"])
        event_table.extend(paper["event_rows"])
        challenge_table.extend(paper["challenge_rows"])
        thread_table.extend(paper["thread_rows"])
        repair_table.extend(paper["repair_rows"])
        macro_table.extend(paper["macro_rows"])
        kc_table.extend(paper["kc_rows"])
        timeseries_table.extend(paper["timeseries_rows"])
        if paper["report_metrics"]:
            report_metrics.append(paper["report_metrics"])
        benchmark_artifacts.append(paper["benchmark_artifacts"])

    aggregate = aggregate_metrics(per_paper, turns_table, event_table, challenge_table)
    return {
        "per_paper": per_paper,
        "aggregate": aggregate,
        "turns": turns_table,
        "hallucination_events": event_table,
        "challenge_turns": challenge_table,
        "thread_turns": thread_table,
        "repair_turns": repair_table,
        "macro_metrics": macro_table,
        "kc_mentions": kc_table,
        "timeseries": timeseries_table,
        "report_metrics": report_metrics,
        "benchmark_artifacts": benchmark_artifacts,
    }


def analyze_paper(
    artifact: PaperArtifacts,
    trajectory: dict[str, Any],
    report: dict[str, Any],
    state: dict[str, Any],
    graph: dict[str, Any],
    questions: dict[str, Any],
) -> dict[str, Any]:
    turns = list(trajectory.get("turns", []))
    paper_id = str(trajectory.get("paper_id") or artifact.paper_key)
    target_model = str(trajectory.get("target_model") or "")
    short_name = short_paper_name(paper_id)

    turn_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    challenge_rows: list[dict[str, Any]] = []
    thread_rows: list[dict[str, Any]] = []
    repair_rows: list[dict[str, Any]] = []
    kc_rows: list[dict[str, Any]] = []
    timeseries_rows: list[dict[str, Any]] = []

    question_type_counts = Counter()
    judge_state_counts = Counter()
    hallucination_type_counts = Counter()
    hallucination_status_counts = Counter()
    challenge_type_counts = Counter()
    challenge_failure_type_counts = Counter()
    modality_counts = Counter()
    answer_mode_counts = Counter()

    unique_target_kcs = set()
    unique_covered_kcs = set()
    unique_macro_target_kcs = set()
    unique_macro_covered_kcs = set()
    cumulative_covered = set()

    challenge_total = 0
    challenge_failed = 0
    challenge_resisted = 0
    text_challenge_total = 0
    text_challenge_failed = 0
    mm_challenge_total = 0
    mm_challenge_failed = 0
    macro_total = 0
    macro_complete = 0
    macro_target_mentions = 0
    macro_covered_mentions = 0
    coverage_turns = 0
    coverage_complete_turns = 0
    detail_followups = 0
    detail_success = 0
    hallucination_followups = 0
    self_corrected_turns = 0
    thread_total = 0
    thread_success = 0
    review_total = 0
    review_regressions = 0
    hallucination_turns = 0
    total_hallucination_events = 0
    multimodal_turns = 0
    image_input_turns = 0
    total_answer_chars = 0
    policy_match_total = 0
    policy_match_count = 0

    macro_accumulator: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "turns": 0,
            "macro_main_turns": 0,
            "challenge_turns": 0,
            "hallucination_events": 0,
            "target_kcs": set(),
            "covered_kcs": set(),
            "missing_kcs": set(),
        }
    )

    for idx, turn in enumerate(turns, start=1):
        question_type = str(turn.get("question_type") or "")
        question_type_counts[question_type] += 1
        judge = turn.get("judge_result") or {}
        state_update = turn.get("state_update") or {}
        judge_state = str(judge.get("state") or "")
        judge_state_counts[judge_state] += 1
        macro_id = str(turn.get("macro_id") or "")
        requires_mm = bool(turn.get("requires_multimodal_input"))
        image_input = bool((turn.get("multimodal_input") or {}).get("image_paths"))
        modality = "multimodal" if requires_mm else "text"
        modality_counts[modality] += 1
        answer_mode = str(turn.get("answer_mode") or "")
        answer_mode_counts[answer_mode] += 1
        answer_chars = len(str(turn.get("model_answer") or ""))
        total_answer_chars += answer_chars
        if requires_mm:
            multimodal_turns += 1
        if image_input:
            image_input_turns += 1

        target_ids = ids(turn.get("target_kc_ids"))
        coverage = judge.get("coverage") or {}
        covered_ids = ids(coverage.get("covered_kc_ids") or judge.get("covered_kc_ids"))
        missing_ids = ids(coverage.get("missing_kc_ids") or judge.get("missing_kc_ids"))
        if not covered_ids:
            covered_ids = ids(state_update.get("lit_kc"))
        unique_target_kcs.update(target_ids)
        unique_covered_kcs.update(covered_ids)
        cumulative_covered.update(covered_ids)

        if coverage:
            coverage_turns += 1
            if bool(coverage.get("coverage_complete")) or (target_ids and set(target_ids) <= set(covered_ids)):
                coverage_complete_turns += 1

        if macro_id:
            macro_accumulator[macro_id]["turns"] += 1
            macro_accumulator[macro_id]["target_kcs"].update(target_ids)
            macro_accumulator[macro_id]["covered_kcs"].update(covered_ids)
            macro_accumulator[macro_id]["missing_kcs"].update(missing_ids)

        if question_type == "macro_main_question":
            macro_total += 1
            macro_target_mentions += len(target_ids)
            macro_covered_mentions += len(covered_ids)
            unique_macro_target_kcs.update(target_ids)
            unique_macro_covered_kcs.update(covered_ids)
            if target_ids and set(target_ids) <= set(covered_ids):
                macro_complete += 1
            if macro_id:
                macro_accumulator[macro_id]["macro_main_turns"] += 1

        events = list(judge.get("hallucination_events") or [])
        if events or judge_state in {"HALLUCINATION", "MISLED", "GLOBAL_OVERCLAIM", "REFUSE_TO_CORRECT"}:
            hallucination_turns += 1
        total_hallucination_events += len(events)
        if macro_id:
            macro_accumulator[macro_id]["hallucination_events"] += len(events)
        for event in events:
            event_type = str(event.get("hallucination_type") or event.get("type") or "unknown")
            status = str(event.get("status") or "unknown")
            hallucination_type_counts[event_type] += 1
            hallucination_status_counts[status] += 1
            event_rows.append(
                {
                    "paper_id": paper_id,
                    "paper_short": short_name,
                    "target_model": target_model,
                    "turn_index": idx,
                    "turn_id": turn.get("turn_id"),
                    "question_type": question_type,
                    "macro_id": macro_id,
                    "event_id": event.get("event_id"),
                    "hallucination_family": event.get("hallucination_family"),
                    "hallucination_type": event_type,
                    "subtype": event.get("subtype"),
                    "status": status,
                    "challenge_type": event.get("challenge_type") or turn.get("challenge_type"),
                    "target_failure_mode": event.get("target_failure_mode") or turn.get("target_failure_mode"),
                    "requires_multimodal_input": requires_mm,
                    "asset_types": ";".join(asset_types(turn)),
                    "related_kc_ids": ";".join(ids(event.get("related_kc_ids"))),
                    "claim": compact_text(event.get("claim"), 240),
                }
            )

        challenge_result = judge.get("challenge_result") or {}
        if question_type == "challenge_question":
            challenge_total += 1
            if macro_id:
                macro_accumulator[macro_id]["challenge_turns"] += 1
            challenge_type = str(turn.get("challenge_type") or challenge_result.get("challenge_type") or "")
            challenge_type_counts[challenge_type] += 1
            failed = bool(challenge_result.get("failed")) or judge_state in {"HALLUCINATION", "MISLED"}
            resisted = bool(challenge_result.get("resisted")) or judge_state == "CHALLENGE_RESISTED"
            if failed:
                challenge_failed += 1
                challenge_failure_type_counts[challenge_type] += 1
            if resisted:
                challenge_resisted += 1
            if requires_mm:
                mm_challenge_total += 1
                mm_challenge_failed += int(failed)
            else:
                text_challenge_total += 1
                text_challenge_failed += int(failed)
            challenge_rows.append(
                {
                    "paper_id": paper_id,
                    "paper_short": short_name,
                    "target_model": target_model,
                    "turn_index": idx,
                    "turn_id": turn.get("turn_id"),
                    "question_id": turn.get("question_id"),
                    "macro_id": macro_id,
                    "challenge_type": challenge_type,
                    "target_failure_mode": turn.get("target_failure_mode") or challenge_result.get("target_failure_mode"),
                    "requires_multimodal_input": requires_mm,
                    "asset_types": ";".join(asset_types(turn)),
                    "judge_state": judge_state,
                    "failed": failed,
                    "resisted": resisted,
                    "incomplete": bool(challenge_result.get("incomplete")),
                    "confidence": challenge_result.get("confidence") or judge.get("confidence"),
                }
            )

        thread_result = judge.get("thread_result") or {}
        if question_type.startswith("thread_"):
            thread_total += 1
            success = bool(thread_result.get("success")) or judge_state == "THREAD_PROGRESS"
            thread_success += int(success)
            thread_rows.append(
                {
                    "paper_id": paper_id,
                    "paper_short": short_name,
                    "target_model": target_model,
                    "turn_index": idx,
                    "turn_id": turn.get("turn_id"),
                    "thread_id": turn.get("thread_id"),
                    "thread_turn_id": turn.get("thread_turn_id"),
                    "thread_role": turn.get("thread_role"),
                    "macro_id": macro_id,
                    "judge_state": judge_state,
                    "success": success,
                    "partial": bool(thread_result.get("partial")),
                    "failed": bool(thread_result.get("failed")),
                    "confidence": thread_result.get("confidence") or judge.get("confidence"),
                }
            )

        if question_type in {"detail_followup", "hallucination_followup", "review_followup"}:
            repair_type = (turn.get("repair_context") or {}).get("repair_type") or question_type
            if question_type == "detail_followup":
                detail_followups += 1
                ok = (not missing_ids) or bool(set(ids((turn.get("repair_context") or {}).get("remaining_kc_ids"))) & set(covered_ids))
                detail_success += int(ok)
            if question_type == "hallucination_followup":
                hallucination_followups += 1
                self_corrected_turns += int(judge_state == "SELF_CORRECTED")
            if question_type == "review_followup":
                review_total += 1
                review_regressions += int(any((e.get("hallucination_type") == "review_regression") for e in events))
            repair_rows.append(
                {
                    "paper_id": paper_id,
                    "paper_short": short_name,
                    "target_model": target_model,
                    "turn_index": idx,
                    "turn_id": turn.get("turn_id"),
                    "question_type": question_type,
                    "repair_type": repair_type,
                    "root_turn_id": (turn.get("repair_context") or {}).get("root_turn_id"),
                    "macro_id": macro_id,
                    "judge_state": judge_state,
                    "covered_kc_ids": ";".join(covered_ids),
                    "missing_kc_ids": ";".join(missing_ids),
                    "requires_multimodal_input": requires_mm,
                }
            )

        policy_next = (judge.get("policy_next_action") or judge.get("next_action") or "")
        actual_next = turn.get("actual_next_action") or ""
        if actual_next:
            policy_match_total += 1
            policy_match_count += int(policy_next == actual_next)

        for kc_id in target_ids:
            kc_rows.append(kc_row(paper_id, short_name, idx, turn, kc_id, "target"))
        for kc_id in covered_ids:
            kc_rows.append(kc_row(paper_id, short_name, idx, turn, kc_id, "covered"))
        for kc_id in missing_ids:
            kc_rows.append(kc_row(paper_id, short_name, idx, turn, kc_id, "missing"))
        for kc_id in ids(state_update.get("lit_kc")):
            kc_rows.append(kc_row(paper_id, short_name, idx, turn, kc_id, "lit_update"))

        turn_rows.append(
            {
                "paper_id": paper_id,
                "paper_short": short_name,
                "target_model": target_model,
                "turn_index": idx,
                "turn_id": turn.get("turn_id"),
                "question_id": turn.get("question_id"),
                "question_type": question_type,
                "macro_id": macro_id,
                "thread_id": turn.get("thread_id"),
                "thread_role": turn.get("thread_role"),
                "challenge_type": turn.get("challenge_type") or challenge_result.get("challenge_type"),
                "target_failure_mode": turn.get("target_failure_mode") or challenge_result.get("target_failure_mode"),
                "requires_multimodal_input": requires_mm,
                "image_input": image_input,
                "asset_ids": ";".join(asset_ids(turn)),
                "asset_types": ";".join(asset_types(turn)),
                "answer_mode": answer_mode,
                "answer_chars": answer_chars,
                "judge_state": judge_state,
                "coverage_complete": bool(coverage.get("coverage_complete")) if coverage else "",
                "target_kc_count": len(target_ids),
                "covered_kc_count": len(covered_ids),
                "missing_kc_count": len(missing_ids),
                "hallucination_event_count": len(events),
                "challenge_failed": bool(challenge_result.get("failed")) if challenge_result else "",
                "challenge_resisted": bool(challenge_result.get("resisted")) if challenge_result else "",
                "actual_next_action": actual_next,
            }
        )

        timeseries_rows.append(
            {
                "paper_id": paper_id,
                "paper_short": short_name,
                "target_model": target_model,
                "turn_index": idx,
                "turn_id": turn.get("turn_id"),
                "question_type": question_type,
                "cumulative_unique_covered_kc": len(cumulative_covered),
                "cumulative_hallucination_events": sum(
                    row["paper_id"] == paper_id for row in event_rows
                ),
                "cumulative_challenge_failures": sum(
                    row["paper_id"] == paper_id and bool(row["failed"]) for row in challenge_rows
                ),
                "cumulative_challenge_turns": sum(
                    row["paper_id"] == paper_id for row in challenge_rows
                ),
            }
        )

    macro_rows = []
    for macro_id, acc in sorted(macro_accumulator.items(), key=lambda item: macro_sort_key(item[0])):
        target = acc["target_kcs"]
        covered = acc["covered_kcs"]
        macro_rows.append(
            {
                "paper_id": paper_id,
                "paper_short": short_name,
                "target_model": target_model,
                "macro_id": macro_id,
                "turns": acc["turns"],
                "macro_main_turns": acc["macro_main_turns"],
                "challenge_turns": acc["challenge_turns"],
                "hallucination_events": acc["hallucination_events"],
                "unique_target_kcs": len(target),
                "unique_covered_kcs": len(covered),
                "macro_unique_coverage_rate": safe_div(len(covered & target), len(target)),
                "remaining_missing_kcs": len(acc["missing_kcs"] - covered),
            }
        )

    metrics = {
        "paper_id": paper_id,
        "paper_short": short_name,
        "target_model": target_model,
        "total_turns": len(turns),
        "macro_main_turns": macro_total,
        "challenge_turns": challenge_total,
        "thread_turns": thread_total,
        "detail_followups": detail_followups,
        "hallucination_followups": hallucination_followups,
        "review_followups": review_total,
        "multimodal_turns": multimodal_turns,
        "image_input_turns": image_input_turns,
        "avg_answer_chars": round(safe_mean([row["answer_chars"] for row in turn_rows]), 2),
        "one_shot_macro_complete_rate": round(safe_div(macro_complete, macro_total), 4),
        "one_shot_kc_mention_coverage_rate": round(safe_div(macro_covered_mentions, macro_target_mentions), 4),
        "one_shot_unique_kc_coverage_rate": round(safe_div(len(unique_macro_covered_kcs & unique_macro_target_kcs), len(unique_macro_target_kcs)), 4),
        "all_turn_unique_kc_coverage_rate": round(safe_div(len(unique_covered_kcs & unique_target_kcs), len(unique_target_kcs)), 4),
        "coverage_complete_turn_rate": round(safe_div(coverage_complete_turns, coverage_turns), 4),
        "challenge_failure_rate": round(safe_div(challenge_failed, challenge_total), 4),
        "challenge_resistance_rate": round(safe_div(challenge_resisted, challenge_total), 4),
        "text_challenge_failure_rate": round(safe_div(text_challenge_failed, text_challenge_total), 4),
        "multimodal_challenge_failure_rate": round(safe_div(mm_challenge_failed, mm_challenge_total), 4),
        "hallucination_turn_rate": round(safe_div(hallucination_turns, len(turns)), 4),
        "hallucination_event_rate": round(safe_div(total_hallucination_events, len(turns)), 4),
        "hallucination_event_count": total_hallucination_events,
        "detail_success_rate": round(safe_div(detail_success, detail_followups), 4),
        "self_correction_turn_rate": round(safe_div(self_corrected_turns, hallucination_followups), 4),
        "thread_success_rate": round(safe_div(thread_success, thread_total), 4),
        "review_regression_rate": round(safe_div(review_regressions, review_total), 4),
        "followup_ratio": round(safe_div(detail_followups + hallucination_followups + review_total, len(turns)), 4),
        "challenge_ratio": round(safe_div(challenge_total, len(turns)), 4),
        "thread_ratio": round(safe_div(thread_total, len(turns)), 4),
        "policy_execution_rate": round(safe_div(policy_match_count, policy_match_total), 4),
        "question_type_counts": json.dumps(question_type_counts, ensure_ascii=False, sort_keys=True),
        "judge_state_counts": json.dumps(judge_state_counts, ensure_ascii=False, sort_keys=True),
        "hallucination_type_counts": json.dumps(hallucination_type_counts, ensure_ascii=False, sort_keys=True),
        "challenge_type_counts": json.dumps(challenge_type_counts, ensure_ascii=False, sort_keys=True),
        "challenge_failure_type_counts": json.dumps(challenge_failure_type_counts, ensure_ascii=False, sort_keys=True),
        "modality_counts": json.dumps(modality_counts, ensure_ascii=False, sort_keys=True),
        "answer_mode_counts": json.dumps(answer_mode_counts, ensure_ascii=False, sort_keys=True),
    }

    return {
        "metrics": metrics,
        "turn_rows": turn_rows,
        "event_rows": event_rows,
        "challenge_rows": challenge_rows,
        "thread_rows": thread_rows,
        "repair_rows": repair_rows,
        "macro_rows": macro_rows,
        "kc_rows": kc_rows,
        "timeseries_rows": timeseries_rows,
        "report_metrics": flatten_report(paper_id, short_name, target_model, report),
        "benchmark_artifacts": benchmark_artifact_stats(paper_id, short_name, target_model, graph, questions),
    }


def aggregate_metrics(
    per_paper: list[dict[str, Any]],
    turns: list[dict[str, Any]],
    events: list[dict[str, Any]],
    challenges: list[dict[str, Any]],
) -> dict[str, Any]:
    numeric_keys = [
        key
        for key, value in per_paper[0].items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    aggregate = {
        "paper_count": len(per_paper),
        "total_turns": sum(int(p["total_turns"]) for p in per_paper),
        "total_challenge_turns": sum(int(p["challenge_turns"]) for p in per_paper),
        "total_hallucination_events": len(events),
    }
    for key in numeric_keys:
        vals = [float(p[key]) for p in per_paper if p.get(key) not in {"", None}]
        aggregate[f"mean_{key}"] = round(safe_mean(vals), 4)
    aggregate["question_type_counts"] = dict(Counter(row["question_type"] for row in turns))
    aggregate["judge_state_counts"] = dict(Counter(row["judge_state"] for row in turns))
    aggregate["hallucination_type_counts"] = dict(Counter(row["hallucination_type"] for row in events))
    aggregate["challenge_failure_by_modality"] = challenge_failure_by_modality(challenges)
    aggregate["challenge_failure_by_type"] = challenge_failure_by_type(challenges)
    return aggregate


def challenge_failure_by_modality(challenges: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out = {}
    for modality in ["text", "multimodal"]:
        rows = [
            row
            for row in challenges
            if ("multimodal" if row.get("requires_multimodal_input") in {True, "True", "true"} else "text")
            == modality
        ]
        out[modality] = {
            "total": len(rows),
            "failed": sum(bool(row.get("failed")) for row in rows),
            "failure_rate": round(safe_div(sum(bool(row.get("failed")) for row in rows), len(rows)), 4),
        }
    return out


def challenge_failure_by_type(challenges: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in challenges:
        by_type[str(row.get("challenge_type") or "unknown")].append(row)
    return {
        key: {
            "total": len(rows),
            "failed": sum(bool(row.get("failed")) for row in rows),
            "failure_rate": round(safe_div(sum(bool(row.get("failed")) for row in rows), len(rows)), 4),
        }
        for key, rows in sorted(by_type.items())
    }


def write_tables(bundle: dict[str, Any], out_dir: Path) -> None:
    write_csv(out_dir / "per_paper_metrics.csv", bundle["per_paper"])
    write_csv(out_dir / "turns.csv", bundle["turns"])
    write_csv(out_dir / "hallucination_events.csv", bundle["hallucination_events"])
    write_csv(out_dir / "challenge_turns.csv", bundle["challenge_turns"])
    write_csv(out_dir / "thread_turns.csv", bundle["thread_turns"])
    write_csv(out_dir / "repair_turns.csv", bundle["repair_turns"])
    write_csv(out_dir / "macro_metrics.csv", bundle["macro_metrics"])
    write_csv(out_dir / "kc_mentions.csv", bundle["kc_mentions"])
    write_csv(out_dir / "turn_timeseries.csv", bundle["timeseries"])
    write_csv(out_dir / "report_metrics_flat.csv", bundle["report_metrics"])


def write_summary(bundle: dict[str, Any], path: Path) -> None:
    aggregate = bundle["aggregate"]
    per_paper = bundle["per_paper"]
    lines = [
        "# PaperGraph-Bench Evaluation Analysis",
        "",
        "## Aggregate",
        "",
        f"- Papers analyzed: {aggregate['paper_count']}",
        f"- Total turns: {aggregate['total_turns']}",
        f"- Challenge turns: {aggregate['total_challenge_turns']}",
        f"- Hallucination events: {aggregate['total_hallucination_events']}",
        f"- Mean one-shot KC coverage: {pct(aggregate.get('mean_one_shot_kc_mention_coverage_rate'))}",
        f"- Mean one-shot macro completion: {pct(aggregate.get('mean_one_shot_macro_complete_rate'))}",
        f"- Mean challenge failure rate: {pct(aggregate.get('mean_challenge_failure_rate'))}",
        f"- Mean text challenge failure rate: {pct(aggregate.get('mean_text_challenge_failure_rate'))}",
        f"- Mean multimodal challenge failure rate: {pct(aggregate.get('mean_multimodal_challenge_failure_rate'))}",
        f"- Mean hallucination-turn rate: {pct(aggregate.get('mean_hallucination_turn_rate'))}",
        "",
        "## Per-Paper Core Metrics",
        "",
        "| Paper | Turns | One-shot KC | Challenge fail | Text fail | MM fail | Hallu turn | Hallu events |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in per_paper:
        lines.append(
            "| {paper} | {turns} | {kc} | {chal} | {txt} | {mm} | {hallu} | {events} |".format(
                paper=row["paper_short"],
                turns=row["total_turns"],
                kc=pct(row["one_shot_kc_mention_coverage_rate"]),
                chal=pct(row["challenge_failure_rate"]),
                txt=pct(row["text_challenge_failure_rate"]),
                mm=pct(row["multimodal_challenge_failure_rate"]),
                hallu=pct(row["hallucination_turn_rate"]),
                events=row["hallucination_event_count"],
            )
        )
    lines.extend(
        [
            "",
            "## Figures",
            "",
            "- `figures/question_type_distribution.svg`",
            "- `figures/per_paper_core_metrics.svg`",
            "- `figures/challenge_failure_by_modality.svg`",
            "- `figures/challenge_failure_by_type.svg`",
            "- `figures/hallucination_type_counts.svg`",
            "- `figures/kc_coverage_over_turns.svg`",
            "- `figures/hallucination_events_over_turns.svg`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_figures(bundle: dict[str, Any], out_dir: Path) -> None:
    aggregate = bundle["aggregate"]
    per_paper = bundle["per_paper"]
    timeseries = bundle["timeseries"]
    write_bar_chart(
        out_dir / "question_type_distribution.svg",
        remap_counts(aggregate["question_type_counts"], QUESTION_TYPE_LABELS),
        "Question type distribution",
        "Turns",
    )
    write_bar_chart(
        out_dir / "hallucination_type_counts.svg",
        aggregate["hallucination_type_counts"],
        "Hallucination event types",
        "Events",
    )
    write_grouped_rate_chart(
        out_dir / "per_paper_core_metrics.svg",
        [
            {
                "label": row["paper_short"],
                "One-shot KC": float(row["one_shot_kc_mention_coverage_rate"]),
                "Challenge fail": float(row["challenge_failure_rate"]),
                "Hallu turn": float(row["hallucination_turn_rate"]),
            }
            for row in per_paper
        ],
        "Per-paper core metrics",
    )
    write_bar_chart(
        out_dir / "challenge_failure_by_modality.svg",
        {
            key: value["failure_rate"]
            for key, value in aggregate["challenge_failure_by_modality"].items()
        },
        "Challenge failure by modality",
        "Failure rate",
        value_format="rate",
    )
    write_bar_chart(
        out_dir / "challenge_failure_by_type.svg",
        {
            key.replace("_challenge", ""): value["failure_rate"]
            for key, value in aggregate["challenge_failure_by_type"].items()
        },
        "Challenge failure by type",
        "Failure rate",
        value_format="rate",
    )
    write_line_chart(
        out_dir / "kc_coverage_over_turns.svg",
        series_from_timeseries(timeseries, "cumulative_unique_covered_kc"),
        "Cumulative unique covered KCs",
        "Turn",
        "KCs",
    )
    write_line_chart(
        out_dir / "hallucination_events_over_turns.svg",
        series_from_timeseries(timeseries, "cumulative_hallucination_events"),
        "Cumulative hallucination events",
        "Turn",
        "Events",
    )
    write_pie_chart(
        out_dir / "question_type_pie.svg",
        remap_counts(aggregate["question_type_counts"], QUESTION_TYPE_LABELS),
        "Question type share",
    )


def write_bar_chart(
    path: Path,
    data: dict[str, float | int],
    title: str,
    y_label: str,
    value_format: str = "count",
) -> None:
    items = [(str(k), float(v)) for k, v in data.items() if float(v) >= 0]
    items = sorted(items, key=lambda item: item[1], reverse=True)
    width, height = 920, 520
    margin = {"left": 90, "right": 30, "top": 70, "bottom": 130}
    plot_w = width - margin["left"] - margin["right"]
    plot_h = height - margin["top"] - margin["bottom"]
    max_v = max([v for _, v in items] + [1.0])
    if value_format == "rate":
        max_v = max(1.0, max_v)
    bar_gap = 10
    bar_w = max(12, (plot_w - bar_gap * max(0, len(items) - 1)) / max(1, len(items)))
    colors = palette()
    parts = svg_header(width, height)
    parts.append(text(width / 2, 30, title, size=22, anchor="middle", weight="700"))
    parts.append(text(24, margin["top"] + plot_h / 2, y_label, size=13, rotate=-90, anchor="middle"))
    parts.append(line(margin["left"], margin["top"], margin["left"], margin["top"] + plot_h, "#555"))
    parts.append(line(margin["left"], margin["top"] + plot_h, margin["left"] + plot_w, margin["top"] + plot_h, "#555"))
    for idx, (label, value) in enumerate(items):
        x = margin["left"] + idx * (bar_w + bar_gap)
        bar_h = 0 if max_v == 0 else value / max_v * plot_h
        y = margin["top"] + plot_h - bar_h
        parts.append(rect(x, y, bar_w, bar_h, colors[idx % len(colors)], rx=4))
        shown = f"{value:.1%}" if value_format == "rate" else str(int(value))
        parts.append(text(x + bar_w / 2, y - 8, shown, size=12, anchor="middle"))
        parts.append(text(x + bar_w / 2, margin["top"] + plot_h + 18, wrap_label(label, 13), size=11, anchor="middle"))
    parts.append(svg_footer())
    path.write_text("\n".join(parts), encoding="utf-8")


def write_grouped_rate_chart(path: Path, rows: list[dict[str, Any]], title: str) -> None:
    metrics = ["One-shot KC", "Challenge fail", "Hallu turn"]
    width, height = 980, 540
    margin = {"left": 70, "right": 160, "top": 70, "bottom": 120}
    plot_w = width - margin["left"] - margin["right"]
    plot_h = height - margin["top"] - margin["bottom"]
    group_gap = 24
    group_w = max(45, (plot_w - group_gap * max(0, len(rows) - 1)) / max(1, len(rows)))
    bar_w = group_w / len(metrics) - 4
    colors = palette()
    parts = svg_header(width, height)
    parts.append(text(width / 2, 30, title, size=22, anchor="middle", weight="700"))
    parts.append(line(margin["left"], margin["top"], margin["left"], margin["top"] + plot_h, "#555"))
    parts.append(line(margin["left"], margin["top"] + plot_h, margin["left"] + plot_w, margin["top"] + plot_h, "#555"))
    for tick in range(0, 6):
        value = tick / 5
        y = margin["top"] + plot_h - value * plot_h
        parts.append(line(margin["left"], y, margin["left"] + plot_w, y, "#e6e6e6"))
        parts.append(text(margin["left"] - 10, y + 4, f"{value:.0%}", size=11, anchor="end"))
    for row_idx, row in enumerate(rows):
        x0 = margin["left"] + row_idx * (group_w + group_gap)
        for metric_idx, metric in enumerate(metrics):
            value = float(row[metric])
            x = x0 + metric_idx * (bar_w + 4)
            bar_h = value * plot_h
            y = margin["top"] + plot_h - bar_h
            parts.append(rect(x, y, bar_w, bar_h, colors[metric_idx], rx=3))
        parts.append(text(x0 + group_w / 2, margin["top"] + plot_h + 18, wrap_label(row["label"], 12), size=10, anchor="middle"))
    for metric_idx, metric in enumerate(metrics):
        y = margin["top"] + metric_idx * 24
        parts.append(rect(width - margin["right"] + 20, y, 14, 14, colors[metric_idx], rx=2))
        parts.append(text(width - margin["right"] + 42, y + 12, metric, size=12))
    parts.append(svg_footer())
    path.write_text("\n".join(parts), encoding="utf-8")


def write_line_chart(path: Path, series: dict[str, list[tuple[int, float]]], title: str, x_label: str, y_label: str) -> None:
    width, height = 980, 520
    margin = {"left": 80, "right": 180, "top": 70, "bottom": 70}
    plot_w = width - margin["left"] - margin["right"]
    plot_h = height - margin["top"] - margin["bottom"]
    max_x = max([x for points in series.values() for x, _ in points] + [1])
    max_y = max([y for points in series.values() for _, y in points] + [1])
    colors = palette()
    parts = svg_header(width, height)
    parts.append(text(width / 2, 30, title, size=22, anchor="middle", weight="700"))
    parts.append(text(width / 2, height - 20, x_label, size=13, anchor="middle"))
    parts.append(text(24, margin["top"] + plot_h / 2, y_label, size=13, rotate=-90, anchor="middle"))
    for tick in range(0, 6):
        yv = max_y * tick / 5
        y = margin["top"] + plot_h - (yv / max_y) * plot_h
        parts.append(line(margin["left"], y, margin["left"] + plot_w, y, "#e6e6e6"))
        parts.append(text(margin["left"] - 10, y + 4, f"{yv:.0f}", size=11, anchor="end"))
    parts.append(line(margin["left"], margin["top"], margin["left"], margin["top"] + plot_h, "#555"))
    parts.append(line(margin["left"], margin["top"] + plot_h, margin["left"] + plot_w, margin["top"] + plot_h, "#555"))
    for idx, (label, points) in enumerate(series.items()):
        color = colors[idx % len(colors)]
        coords = []
        for x_val, y_val in points:
            x = margin["left"] + x_val / max_x * plot_w
            y = margin["top"] + plot_h - y_val / max_y * plot_h
            coords.append((x, y))
        parts.append(polyline(coords, color))
        if coords:
            x_last, y_last = coords[-1]
            parts.append(circle(x_last, y_last, 3.5, color))
        legend_y = margin["top"] + idx * 24
        parts.append(line(width - margin["right"] + 15, legend_y + 7, width - margin["right"] + 35, legend_y + 7, color, width=3))
        parts.append(text(width - margin["right"] + 42, legend_y + 11, wrap_label(label, 20), size=11))
    parts.append(svg_footer())
    path.write_text("\n".join(parts), encoding="utf-8")


def write_pie_chart(path: Path, data: dict[str, int | float], title: str) -> None:
    items = [(str(k), float(v)) for k, v in data.items() if float(v) > 0]
    total = sum(v for _, v in items)
    width, height = 820, 520
    cx, cy, r = 250, 270, 160
    colors = palette()
    parts = svg_header(width, height)
    parts.append(text(width / 2, 30, title, size=22, anchor="middle", weight="700"))
    start = -math.pi / 2
    for idx, (label, value) in enumerate(items):
        angle = value / total * 2 * math.pi if total else 0
        end = start + angle
        parts.append(pie_slice(cx, cy, r, start, end, colors[idx % len(colors)]))
        legend_y = 95 + idx * 24
        parts.append(rect(500, legend_y, 14, 14, colors[idx % len(colors)], rx=2))
        parts.append(text(522, legend_y + 12, f"{label} ({value / total:.1%})", size=12))
        start = end
    parts.append(svg_footer())
    path.write_text("\n".join(parts), encoding="utf-8")


def series_from_timeseries(rows: list[dict[str, Any]], key: str) -> dict[str, list[tuple[int, float]]]:
    series: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for row in rows:
        series[row["paper_short"]].append((int(row["turn_index"]), float(row[key])))
    return dict(series)


def remap_counts(counts: dict[str, Any], labels: dict[str, str]) -> dict[str, Any]:
    out = {}
    for key, value in counts.items():
        out[labels.get(key, key or "unknown")] = value
    return out


def flatten_report(paper_id: str, short_name: str, target_model: str, report: dict[str, Any]) -> dict[str, Any]:
    if not report:
        return {}
    out = {"paper_id": paper_id, "paper_short": short_name, "target_model": target_model}
    for group, value in report.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, (str, int, float, bool)) or sub_value is None:
                    out[f"{group}.{sub_key}"] = sub_value
    return out


def benchmark_artifact_stats(
    paper_id: str,
    short_name: str,
    target_model: str,
    graph: dict[str, Any],
    questions: dict[str, Any],
) -> dict[str, Any]:
    kc_nodes = list(graph.get("kc_nodes") or [])
    macro_nodes = list(graph.get("macro_nodes") or [])
    reasoning_edges = list(graph.get("reasoning_edges") or [])
    reasoning_threads = list(graph.get("reasoning_threads") or [])
    active_ids = set(ids(graph.get("active_kc_ids")))
    multimodal_kcs = [node for node in kc_nodes if bool((node.get("modality") or {}).get("is_multimodal"))]
    challenge_questions = list(questions.get("challenge_questions") or [])
    thread_challenges = list(questions.get("thread_challenge_questions") or [])
    all_challenges = challenge_questions + thread_challenges
    main_questions = questions.get("macro_main_questions") or questions.get("main_questions") or []
    thread_seeds = list(questions.get("thread_question_seeds") or [])
    challenge_type_counts = Counter(
        str(q.get("challenge_type") or (q.get("question_metadata") or {}).get("challenge_type") or "unknown")
        for q in all_challenges
    )
    edge_scope_counts = Counter(str(edge.get("scope") or edge.get("source_layer") or "unknown") for edge in reasoning_edges)
    return {
        "paper_id": paper_id,
        "paper_short": short_name,
        "target_model": target_model,
        "macro_nodes": len(macro_nodes),
        "kc_nodes": len(kc_nodes),
        "active_kcs": len(active_ids),
        "multimodal_kcs": len(multimodal_kcs),
        "multimodal_kc_rate": round(safe_div(len(multimodal_kcs), len(kc_nodes)), 4),
        "reasoning_edges": len(reasoning_edges),
        "reasoning_threads": len(reasoning_threads),
        "macro_questions": len(main_questions),
        "challenge_questions": len(challenge_questions),
        "thread_challenge_questions": len(thread_challenges),
        "all_challenge_questions": len(all_challenges),
        "text_challenge_questions": sum(not question_requires_multimodal(q) for q in all_challenges),
        "multimodal_challenge_questions": sum(question_requires_multimodal(q) for q in all_challenges),
        "thread_question_seeds": len(thread_seeds),
        "challenge_type_counts": json.dumps(challenge_type_counts, ensure_ascii=False, sort_keys=True),
        "edge_scope_counts": json.dumps(edge_scope_counts, ensure_ascii=False, sort_keys=True),
    }


def question_requires_multimodal(question: dict[str, Any]) -> bool:
    metadata = question.get("question_metadata") or {}
    return bool(
        question.get("requires_multimodal_input")
        or metadata.get("requires_multimodal_input")
        or question.get("asset_references")
        or metadata.get("asset_ids")
    )


def kc_row(paper_id: str, short_name: str, idx: int, turn: dict[str, Any], kc_id: str, role: str) -> dict[str, Any]:
    return {
        "paper_id": paper_id,
        "paper_short": short_name,
        "turn_index": idx,
        "turn_id": turn.get("turn_id"),
        "question_id": turn.get("question_id"),
        "question_type": turn.get("question_type"),
        "macro_id": turn.get("macro_id"),
        "kc_id": kc_id,
        "role": role,
        "requires_multimodal_input": bool(turn.get("requires_multimodal_input")),
    }


def read_json(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def ids(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value if str(v)]
    if isinstance(value, tuple):
        return [str(v) for v in value if str(v)]
    if isinstance(value, str):
        return [value] if value else []
    return []


def asset_ids(turn: dict[str, Any]) -> list[str]:
    refs = turn.get("asset_references") or []
    if isinstance(refs, dict):
        refs = [refs]
    output = []
    for ref in refs:
        if isinstance(ref, dict) and ref.get("asset_id"):
            output.append(str(ref["asset_id"]))
    for asset_id in ids((turn.get("multimodal_input") or {}).get("asset_ids")):
        if asset_id not in output:
            output.append(asset_id)
    return output


def asset_types(turn: dict[str, Any]) -> list[str]:
    refs = turn.get("asset_references") or []
    if isinstance(refs, dict):
        refs = [refs]
    output = []
    for ref in refs:
        if isinstance(ref, dict) and ref.get("asset_type"):
            output.append(str(ref["asset_type"]))
    return list(dict.fromkeys(output))


def safe_div(num: float, den: float) -> float:
    return 0.0 if not den else num / den


def safe_mean(values: list[float]) -> float:
    return mean(values) if values else 0.0


def pct(value: Any) -> str:
    try:
        return f"{float(value):.1%}"
    except (TypeError, ValueError):
        return "n/a"


def compact_text(value: Any, max_len: int) -> str:
    text_value = " ".join(str(value or "").split())
    return text_value if len(text_value) <= max_len else text_value[: max_len - 1] + "..."


def short_paper_name(paper_id: str) -> str:
    name = paper_id.replace("_", " ")
    for marker in [" - 2025 - ", " - 2024 - ", " - "]:
        if marker in name:
            name = name.split(marker)[0]
            break
    return name[:36].strip()


def macro_sort_key(macro_id: str) -> tuple[int, str]:
    digits = "".join(ch for ch in macro_id if ch.isdigit())
    return (int(digits) if digits else 10_000, macro_id)


def palette() -> list[str]:
    return [
        "#4C78A8",
        "#F58518",
        "#54A24B",
        "#E45756",
        "#72B7B2",
        "#B279A2",
        "#FF9DA6",
        "#9D755D",
        "#BAB0AC",
        "#6B6ECF",
    ]


def svg_header(width: int, height: int) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial, Helvetica, sans-serif; fill:#1f2933;} .small{font-size:11px;}</style>',
    ]


def svg_footer() -> str:
    return "</svg>"


def esc(value: Any) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def text(x: float, y: float, content: str, size: int = 12, anchor: str = "start", weight: str = "400", rotate: int | None = None) -> str:
    transform = f' transform="rotate({rotate} {x} {y})"' if rotate is not None else ""
    if "\n" in content:
        lines = content.split("\n")
        tspans = "".join(
            f'<tspan x="{x:.1f}" dy="{0 if idx == 0 else 13}">{esc(line)}</tspan>'
            for idx, line in enumerate(lines)
        )
        return f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" font-weight="{weight}" text-anchor="{anchor}"{transform}>{tspans}</text>'
    return f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" font-weight="{weight}" text-anchor="{anchor}"{transform}>{esc(content)}</text>'


def rect(x: float, y: float, w: float, h: float, fill: str, rx: int = 0) -> str:
    return f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(0, w):.1f}" height="{max(0, h):.1f}" fill="{fill}" rx="{rx}"/>'


def line(x1: float, y1: float, x2: float, y2: float, stroke: str, width: int = 1) -> str:
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke}" stroke-width="{width}"/>'


def polyline(points: list[tuple[float, float]], stroke: str) -> str:
    coords = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    return f'<polyline points="{coords}" fill="none" stroke="{stroke}" stroke-width="2.5"/>'


def circle(x: float, y: float, r: float, fill: str) -> str:
    return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" fill="{fill}"/>'


def pie_slice(cx: float, cy: float, r: float, start: float, end: float, fill: str) -> str:
    x1 = cx + r * math.cos(start)
    y1 = cy + r * math.sin(start)
    x2 = cx + r * math.cos(end)
    y2 = cy + r * math.sin(end)
    large = 1 if end - start > math.pi else 0
    return (
        f'<path d="M {cx:.1f},{cy:.1f} L {x1:.1f},{y1:.1f} '
        f'A {r:.1f},{r:.1f} 0 {large} 1 {x2:.1f},{y2:.1f} Z" fill="{fill}" stroke="#fff" stroke-width="2"/>'
    )


def wrap_label(label: str, width: int) -> str:
    words = str(label).split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) <= width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word[:width]
    if current:
        lines.append(current)
    return "\n".join(lines[:3])


if __name__ == "__main__":
    main()
