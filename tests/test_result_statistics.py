"""Report statistics use local synthetic files, never credentials or API calls."""

from __future__ import annotations

import csv
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from summarize_results import _csv, main, print_summary, summarize
from statistics_inputs import report_path


class ResultStatisticsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix=".papergraph-test-statistics-", dir=ROOT)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def report(self, paper="paper", model="unit-model", round="single", covered=1, total=2, events=1, resolved=1, challenge_total=1, status="completed"):
        task = {"round": round, "paper_id": paper, "model": model}
        path = report_path(self.root, task)
        path.parent.mkdir(parents=True, exist_ok=True)
        report = {"paper_id": paper, "target_model": model,
                  "summary": {"total_turns": 2, "evaluation_status": status, "failed": False},
                  "coverage_metrics": {f"{name}_kc_{suffix}": value for name in ("global", "target", "critical") for suffix, value in (("lit_count", covered), ("total", total))},
                  "macro_metrics": {"macro_total": 2, "macro_completion_rate": 0.5},
                  "challenge_metrics": {"challenge_total": challenge_total, "challenge_failure_count": min(1, challenge_total)},
                  "hallucination_metrics": {"hallucination_turn_count": int(events > 0), "hallucination_event_count": events,
                                            "hallucination_by_type": {"challenge_false_premise_acceptance": events} if events else {},
                                            "hallucination_by_status": {key: value for key, value in {"resolved": resolved, "exhausted": events - resolved}.items() if value}},
                  "detail_completion_metrics": {"coverage_gap_count": 1, "coverage_gap_resolved_count": 1},
                  "thread_metrics": {"bridge_reasoning_tested_count": 1, "bridge_reasoning_success_count": 1}}
        trajectory = {"paper_id": paper, "target_model": model,
                      "turns": [{"turn_id": "T1", "question_id": "Q1"}, {"turn_id": "T2", "question_id": "Q2"}]}
        final_events = {}
        for index in range(events):
            event_id = f"H_T1_{index + 1}"
            final_events[event_id] = {
                "event_id": event_id, "created_at_turn": "T1", "source_question_id": "Q1",
                "hallucination_family": "challenge_failure", "hallucination_type": "challenge_false_premise_acceptance",
                "subtype": "false_premise", "claim": "Unsupported fixture claim", "related_kc_ids": ["K1"],
                "status": "resolved" if index < resolved else "exhausted", "followup_turns": ["T2"],
                "resolved_at_turn": "T2" if index < resolved else None, "source": "judge_result.hallucination_events",
            }
        path.write_text(json.dumps(report), encoding="utf-8")
        (path.parent / "dialogue_trajectory.json").write_text(json.dumps(trajectory), encoding="utf-8")
        state_path = path.parent / "cache" / "eval_state_graph.json"
        state_path.parent.mkdir()
        state_path.write_text(json.dumps({"paper_id": paper, "target_model": model, "hallucination_events": final_events}), encoding="utf-8")
        return path, task

    def edit(self, path, change):
        value = json.loads(path.read_text(encoding="utf-8"))
        change(value)
        path.write_text(json.dumps(value), encoding="utf-8")

    def test_unknown_plan_and_bom_csv(self):
        self.report(paper="中文论文")
        result = summarize(self.root)
        self.assertIsNone(result["task_counts"]["planned"])
        self.assertEqual(result["models"][0]["global_kc_coverage_mean"], .5)
        path = self.root / "statistics" / "per_paper.csv"
        self.assertTrue(path.read_bytes().startswith(b"\xef\xbb\xbf"))
        self.assertIn("中文论文", path.read_text(encoding="utf-8-sig"))

    def test_default_statistics_export_four_dimensions_and_classification(self):
        self.report()
        result = summarize(self.root)
        destination = self.root / "statistics"
        expected = {"summary.csv", "per_paper.csv", "summary.json", "four_dimensions.csv",
                    "classification_events.csv", "classification_events.json",
                    "classification_summary.csv", "classification_summary.json", "classification_paths.csv"}
        self.assertEqual({path.name for path in destination.iterdir()}, expected)
        with (destination / "four_dimensions.csv").open(encoding="utf-8-sig", newline="") as stream:
            dimensions = list(csv.DictReader(stream))
        self.assertEqual(len(dimensions), 6)
        self.assertEqual(len({row["dimension_id"] for row in dimensions}), 4)
        challenge = next(row for row in dimensions if row["metric"] == "challenge_induced_hallucination")
        self.assertEqual(challenge["source_metric"], "challenge_failure")
        self.assertEqual(float(challenge["mean"]), 1)
        self.assertEqual(result["classification"]["event_count"], 1)
        self.assertEqual(result["classification"]["three_fields_present_rate"], 1)
        with (destination / "classification_paths.csv").open(encoding="utf-8-sig", newline="") as stream:
            paths = list(csv.DictReader(stream))
        self.assertEqual(paths[0]["family"], "challenge_failure")
        self.assertEqual(paths[0]["event_count"], "1")
        self.assertEqual(result["four_dimensions"][1]["mean"], result["models"][0]["challenge_failure_mean"])
        console = io.StringIO()
        with redirect_stdout(console):
            print_summary(result)
        for label in ("Knowledge coverage", "Challenge-induced hallucination", "Hallucination behavior", "Error repair", "[classification]"):
            self.assertIn(label, console.getvalue())

    def test_missing_classification_state_stops_before_exporting_successful_statistics(self):
        path, _ = self.report()
        (path.parent / "cache" / "eval_state_graph.json").unlink()
        with redirect_stderr(io.StringIO()):
            self.assertEqual(main(["--result-root", str(self.root)]), 1)
        self.assertFalse((self.root / "statistics").exists())

    def test_classification_csv_protects_untrusted_labels_without_modifying_json(self):
        path, _ = self.report()
        state_path = path.parent / "cache" / "eval_state_graph.json"
        self.edit(state_path, lambda value: value["hallucination_events"]["H_T1_1"].update(subtype="=HYPERLINK(1)", claim="  +untrusted"))
        summarize(self.root)
        with (self.root / "statistics" / "classification_events.csv").open(encoding="utf-8-sig", newline="") as stream:
            event = next(csv.DictReader(stream))
        self.assertEqual(event["subtype"], "'=HYPERLINK(1)")
        self.assertEqual(event["claim"], "'  +untrusted")
        original = json.loads((self.root / "statistics" / "classification_events.json").read_text(encoding="utf-8"))[0]
        self.assertEqual(original["subtype"], "=HYPERLINK(1)")

    def test_repetitions_are_averaged_before_papers(self):
        self.report(paper="a", round="round-01", covered=0)
        self.report(paper="a", round="round-02", covered=0)
        self.report(paper="b", round="round-01", covered=2)
        model = summarize(self.root)["models"][0]
        self.assertEqual(model["global_kc_coverage_mean"], .5)
        self.assertAlmostEqual(model["global_kc_coverage_std"], 2 ** -.5)
        self.assertAlmostEqual(model["global_kc_coverage_micro"], 1 / 3)
        self.assertEqual(model["global_kc_coverage_valid_papers"], 2)
        self.assertEqual(model["global_kc_coverage_valid_runs"], 3)

    def test_zero_denominators_are_na(self):
        self.report(total=0, covered=0, events=0, resolved=0, challenge_total=0)
        model = summarize(self.root)["models"][0]
        self.assertIsNone(model["global_kc_coverage_mean"])
        self.assertIsNone(model["global_kc_coverage_std"])
        self.assertIsNone(model["hallucination_event_closure_mean"])
        self.assertEqual(model["hallucination_event_density_mean"], 0)
        self.assertEqual(model["challenge_failure_valid_papers"], 0)
        self.assertIn('null', (self.root / "statistics" / "summary.json").read_text(encoding="utf-8"))

    def test_density_above_one_and_final_event_closure(self):
        self.report(events=5, resolved=2)
        model = summarize(self.root)["models"][0]
        self.assertEqual(model["hallucination_event_density_mean"], 2.5)
        self.assertEqual(model["hallucination_event_closure_mean"], .4)
        self.assertEqual(model["hallucination_events_sum"], 5)

    def test_missing_failed_and_completed_are_separate(self):
        _, complete = self.report(paper="complete", round="round-01")
        _, failed = self.report(paper="failed", round="round-01", status="stopped")
        missing = {**failed, "paper_id": "missing"}
        (self.root / "testing.json").write_text(json.dumps({"expected_tasks": [complete, failed, missing]}))
        result = summarize(self.root)
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(result["task_counts"], {"planned": 3, "discovered": 2, "completed": 1, "incomplete": 2})
        self.assertEqual(result["models"][0]["global_kc_coverage_valid_papers"], 1)
        with redirect_stdout(io.StringIO()):
            self.assertEqual(main(["--result-root", str(self.root)]), 2)

    def test_corrupt_turn_count_fails(self):
        path, _ = self.report()
        self.edit(path, lambda value: value["summary"].update(total_turns=3))
        self.assertRaisesRegex(ValueError, "turn counts", summarize, self.root)
        self.assertFalse((self.root / "statistics").exists())

    def test_duplicate_turn_ids_fail(self):
        path, _ = self.report()
        self.edit(path.parent / "dialogue_trajectory.json", lambda value: value["turns"][1].update(turn_id="T1"))
        self.assertRaisesRegex(ValueError, "duplicate dialogue", summarize, self.root)

    def test_wrong_identity_fails(self):
        path, task = self.report()
        self.edit(path, lambda value: value.update(paper_id="other"))
        self.assertRaisesRegex(ValueError, "identity", summarize, self.root, None, [task])

    def test_duplicate_report_tasks_fail(self):
        path, _ = self.report()
        other = self.root / "duplicate" / path.relative_to(self.root)
        other.parent.mkdir(parents=True)
        other.write_bytes(path.read_bytes())
        self.assertRaisesRegex(ValueError, "Duplicate report task", summarize, self.root)

    def test_round_rate_must_have_unique_count(self):
        path, _ = self.report()
        self.edit(path, lambda value: value["macro_metrics"].update(macro_total=100000))
        self.assertRaisesRegex(ValueError, "uniquely", summarize, self.root)

    def test_wrong_final_event_counts_fail(self):
        path, _ = self.report()
        self.edit(path, lambda value: value["hallucination_metrics"].update(hallucination_event_count=3))
        self.assertRaisesRegex(ValueError, "status counts", summarize, self.root)

    def test_missing_trajectory_is_incomplete(self):
        path, _ = self.report()
        (path.parent / "dialogue_trajectory.json").unlink()
        result = summarize(self.root)
        self.assertEqual(result["status"], "incomplete")
        self.assertIsNone(result["models"][0]["global_kc_coverage_mean"])

    def test_non_integer_metric_count_rejected(self):
        path, _ = self.report()
        self.edit(path, lambda value: value["coverage_metrics"].update(global_kc_lit_count=.5))
        with redirect_stderr(io.StringIO()):
            self.assertEqual(main(["--result-root", str(self.root)]), 1)

    def test_model_ids_with_slash_preserved(self):
        self.report(model="org/model-name")
        self.assertEqual(summarize(self.root)["models"][0]["model"], "org/model-name")

    def test_measurement_manifest_supplies_task_plan(self):
        self.report(round="round-01")
        manifest = {"paper_ids": ["paper"], "target_models": ["unit-model"], "repeats": 2}
        (self.root / "measurement.json").write_text(json.dumps(manifest), encoding="utf-8")
        result = summarize(self.root)
        self.assertEqual(result["task_counts"]["planned"], 2)
        self.assertEqual(result["task_counts"]["incomplete"], 1)

    def test_unexpected_report_cannot_silently_join_plan(self):
        _, planned = self.report()
        self.report(paper="unplanned")
        self.assertRaisesRegex(ValueError, "Unexpected report", summarize, self.root, None, [planned])

    def test_duplicate_plan_cannot_double_count(self):
        _, planned = self.report()
        self.assertRaisesRegex(ValueError, "Duplicate", summarize, self.root, None, [planned, planned])

    def test_missing_model_does_not_get_zero_scores(self):
        _, planned = self.report()
        missing = {**planned, "model": "missing-model"}
        models = {item["model"]: item for item in summarize(self.root, expected_tasks=[planned, missing])["models"]}
        self.assertEqual(models["missing-model"]["discovered_tasks"], 0)
        self.assertIsNone(models["missing-model"]["global_kc_coverage_mean"])
        self.assertEqual(models["missing-model"]["status"], "incomplete")

    def test_csv_formula_text_protection_preserves_numbers_and_na(self):
        dangerous = ["=1+1", "+SUM(A1)", "-cmd", "@SUM(A1)", "\t=1", "\n@bad", "  +1"]
        rows = [{"model": text, "reason": "ordinary text", "value": -2.5, "missing": None} for text in dangerous]
        path = self.root / "cells.csv"
        _csv(path, rows, list(rows[0]))
        with path.open(encoding="utf-8-sig", newline="") as stream:
            for original, row in zip(dangerous, csv.DictReader(stream)):
                self.assertEqual(row, {"model": "'" + original, "reason": "ordinary text", "value": "-2.5", "missing": ""})

    def test_csv_escapes_model_and_reason_without_changing_json_identity(self):
        path, _ = self.report(model="-unit-model", status="stopped")
        reason = " \t\n=HYPERLINK(1)"
        self.edit(path, lambda value: value["summary"].update(failure_reason=reason))
        result = summarize(self.root)
        with (self.root / "statistics" / "per_paper.csv").open(encoding="utf-8-sig", newline="") as stream:
            row = next(csv.DictReader(stream))
        self.assertEqual((row["model"], row["reason"]), ("'-unit-model", "'" + reason))
        saved = json.loads((self.root / "statistics" / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["models"][0]["model"], "-unit-model")
        self.assertEqual(result["models"][0]["model"], "-unit-model")


if __name__ == "__main__":
    unittest.main()
