"""Paper statistics use synthetic reports/trajectories, never credentials or APIs."""

from __future__ import annotations

import csv
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))
from summarize_results import _csv, main, print_summary, summarize
from statistics_metrics import extract_metrics
from statistics_fixture import edit_json, write_fixture


class ResultStatisticsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix=".papergraph-test-statistics-", dir=ROOT)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def report(self, **kwargs):
        return write_fixture(self.root, **kwargs)

    def read_csv(self, name):
        with (self.root / "statistics" / name).open(encoding="utf-8-sig", newline="") as stream:
            return list(csv.DictReader(stream))

    def trajectory(self, path):
        return path.with_name("dialogue_trajectory.json")

    def test_unknown_plan_and_bom_csv(self):
        self.report(paper="中文论文")
        result = summarize(self.root)
        self.assertIsNone(result["task_counts"]["planned"])
        self.assertEqual(result["models"][0]["one_turn_kc_coverage"], .5)
        path = self.root / "statistics" / "per_paper.csv"
        self.assertTrue(path.read_bytes().startswith(b"\xef\xbb\xbf"))
        self.assertIn("中文论文", path.read_text(encoding="utf-8-sig"))

    def test_default_exports_eleven_files_and_eleven_paper_dimension_rows(self):
        self.report()
        result = summarize(self.root)
        expected = {"summary.csv", "per_paper.csv", "summary.json", "four_dimensions.csv", "repair_tasks.csv",
                    "classification_events.csv", "classification_events.json", "classification_summary.csv",
                    "classification_summary.json", "classification_paths.csv", "failure_category_by_model.csv"}
        self.assertEqual({path.name for path in (self.root / "statistics").iterdir()}, expected)
        dimensions = self.read_csv("four_dimensions.csv")
        self.assertEqual(len(dimensions), 11)
        self.assertEqual(len({row["dimension_id"] for row in dimensions}), 4)
        challenge = next(row for row in dimensions if row["metric"] == "challenge_induced_hallucination")
        self.assertEqual((challenge["source_metric"], challenge["aggregation"], float(challenge["value"])),
                         ("challenge_failure_rate", "pooled", 1))
        self.assertNotIn("mean", challenge)
        self.assertEqual(result["aggregation"]["primary"]["one_turn_kc_coverage"], "paper_mean")
        self.assertEqual(result["classification"]["event_count"], 1)
        self.assertEqual(self.read_csv("classification_paths.csv")[0]["event_count"], "1")
        console = io.StringIO()
        with redirect_stdout(console):
            print_summary(result)
        for label in ("Knowledge coverage", "Challenge-induced hallucination", "Hallucination behavior",
                      "Error repair", "[classification]", "[paper categories]"):
            self.assertIn(label, console.getvalue())

    def test_paper_four_categories_merge_threads_and_preserve_unmapped(self):
        path, _ = self.report(events=5)
        types = ["thread_premise_mutation_acceptance", "thread_overclaim_acceptance",
                 "thread_wrong_bridge_acceptance", "review_regression", "fabricated_claim"]
        def update_events(trajectory):
            for event, raw_type in zip(trajectory["turns"][1]["judge_result"]["hallucination_events"], types):
                event["hallucination_type"] = raw_type
        edit_json(self.trajectory(path), update_events)
        edit_json(path, lambda value: value["hallucination_metrics"].update(hallucination_by_type={key: 1 for key in types}))
        result = summarize(self.root)
        model = self.read_csv("failure_category_by_model.csv")[0]
        for key in ("contradicted_kc_claim", "challenge_false_premise_acceptance",
                    "challenge_overclaim_acceptance", "challenge_wrong_relation_acceptance"):
            self.assertEqual(model[key], "1")
        self.assertEqual((model["categorized_event_count"], model["unmapped_event_count"], model["event_count"]), ("4", "1", "5"))
        self.assertEqual(result["classification"]["unmapped_event_count"], 1)
        saved = json.loads((self.root / "statistics" / "classification_events.json").read_text(encoding="utf-8"))
        original = next(event for event in saved if event["type"] == "fabricated_claim")
        self.assertIsNone(original["paper_category"])
        self.assertIn("unmapped_paper_category", original["issues"])

    def test_no_cache_or_private_configuration_is_needed(self):
        path, _ = self.report()
        self.assertFalse((path.parent / "cache").exists())
        (path.parent / "cache").mkdir()
        (path.parent / "cache" / "eval_state_graph.json").write_text("not valid JSON", encoding="utf-8")
        with patch("src.config.load_environment", side_effect=AssertionError("must stay offline")):
            self.assertEqual(summarize(self.root)["status"], "completed")

    def test_initial_coverage_and_event_turns_ignore_runtime_final_aliases(self):
        path, _ = self.report()
        report = json.loads(path.read_text(encoding="utf-8"))
        trajectory = json.loads(self.trajectory(path).read_text(encoding="utf-8"))
        values = extract_metrics(report, trajectory)
        self.assertEqual((values["one_turn_kc_coverage"], values["one_turn_macro_complete_rate"]), (.5, 0))
        self.assertEqual(values["hallucination_turn_rate"], .5)
        model = summarize(self.root)["models"][0]
        self.assertIsNone(model["hallucination_repair_success_rate"])
        self.assertIsNone(model["detail_repair_avg_rounds"])
        self.assertEqual(self.read_csv("repair_tasks.csv"), [])

    def test_classification_csv_protects_labels_but_json_keeps_original(self):
        path, _ = self.report()
        edit_json(self.trajectory(path), lambda value: value["turns"][1]["judge_result"]["hallucination_events"][0].update(
            subtype="=HYPERLINK(1)", claim="  +untrusted"))
        summarize(self.root)
        event = self.read_csv("classification_events.csv")[0]
        self.assertEqual((event["subtype"], event["claim"]), ("'=HYPERLINK(1)", "'  +untrusted"))
        original = json.loads((self.root / "statistics" / "classification_events.json").read_text(encoding="utf-8"))[0]
        self.assertEqual(original["subtype"], "=HYPERLINK(1)")

    def test_exported_event_status_is_detection_snapshot(self):
        self.report()
        summarize(self.root)
        event = self.read_csv("classification_events.csv")[0]
        self.assertEqual(event["status_at_detection"], "unresolved")
        self.assertNotIn("status", event)
        self.assertNotIn("state_path", event)
        self.assertTrue(event["trajectory_path"].endswith("dialogue_trajectory.json"))

    def test_repetitions_are_averaged_before_papers(self):
        self.report(paper="a", round="round-01", covered=0)
        self.report(paper="a", round="round-02", covered=0)
        self.report(paper="b", round="round-01", covered=2)
        model = summarize(self.root)["models"][0]
        self.assertEqual(model["one_turn_kc_coverage"], .5)
        self.assertAlmostEqual(model["one_turn_kc_coverage_std"], 2 ** -.5)
        self.assertAlmostEqual(model["one_turn_kc_coverage_micro"], 1 / 3)
        self.assertEqual((model["one_turn_kc_coverage_valid_papers"], model["one_turn_kc_coverage_valid_runs"]), (2, 3))

    def test_zero_denominators_are_na_but_actual_zero_events_remain_zero(self):
        self.report(total=0, covered=0, events=0, challenge_total=0)
        model = summarize(self.root)["models"][0]
        self.assertIsNone(model["one_turn_kc_coverage"])
        self.assertIsNone(model["hallucination_repair_success_rate"])
        self.assertIsNone(model["detail_repair_avg_rounds"])
        self.assertEqual((model["mean_hallucination_events"], model["hallucination_event_density"]), (0, 0))
        self.assertEqual(model["challenge_failure_rate_valid_papers"], 0)

    def test_density_above_one_and_event_mean_use_trajectory_events(self):
        self.report(events=5)
        model = summarize(self.root)["models"][0]
        self.assertEqual((model["hallucination_event_density"], model["mean_hallucination_events"]), (2.5, 5))
        self.assertEqual(model["hallucination_events_sum"], 5)
        self.assertNotIn("hallucination_event_closure_mean", model)

    def test_missing_failed_and_completed_tasks_are_separate(self):
        _, complete = self.report(paper="complete", round="round-01")
        _, failed = self.report(paper="failed", round="round-01", status="stopped")
        missing = {**failed, "paper_id": "missing"}
        (self.root / "testing.json").write_text(json.dumps({"expected_tasks": [complete, failed, missing]}), encoding="utf-8")
        result = summarize(self.root)
        self.assertEqual(result["task_counts"], {"planned": 3, "discovered": 2, "completed": 1, "incomplete": 2})
        self.assertEqual(result["models"][0]["one_turn_kc_coverage_valid_papers"], 1)
        with redirect_stdout(io.StringIO()):
            self.assertEqual(main(["--result-root", str(self.root)]), 2)

    def test_corrupt_turn_count_fails(self):
        path, _ = self.report()
        edit_json(path, lambda value: value["summary"].update(total_turns=3))
        self.assertRaisesRegex(ValueError, "turn counts", summarize, self.root)
        self.assertFalse((self.root / "statistics").exists())

    def test_duplicate_turn_ids_fail(self):
        path, _ = self.report()
        edit_json(self.trajectory(path), lambda value: value["turns"][1].update(turn_id="T1"))
        self.assertRaisesRegex(ValueError, "duplicate dialogue", summarize, self.root)

    def test_wrong_identity_fails(self):
        path, task = self.report()
        edit_json(path, lambda value: value.update(paper_id="other"))
        self.assertRaisesRegex(ValueError, "identity", summarize, self.root, None, [task])

    def test_duplicate_report_tasks_fail(self):
        path, _ = self.report()
        other = self.root / "duplicate" / path.relative_to(self.root)
        other.parent.mkdir(parents=True)
        other.write_bytes(path.read_bytes())
        self.assertRaisesRegex(ValueError, "Duplicate report task", summarize, self.root)

    def test_trajectory_requires_answer_judge_and_event_list(self):
        path, _ = self.report()
        edit_json(self.trajectory(path), lambda value: value["turns"][0].pop("model_answer"))
        self.assertRaisesRegex(ValueError, "model_answer", summarize, self.root)

    def test_wrong_report_event_count_and_type_distribution_fail(self):
        path, _ = self.report()
        edit_json(path, lambda value: value["hallucination_metrics"].update(hallucination_event_count=3))
        self.assertRaisesRegex(ValueError, "event count", summarize, self.root)
        edit_json(path, lambda value: value["hallucination_metrics"].update(hallucination_event_count=1, hallucination_by_type={"fabricated_claim": 1}))
        self.assertRaisesRegex(ValueError, "hallucination_by_type", summarize, self.root)

    def test_duplicate_event_ids_fail_instead_of_silent_deduplication(self):
        path, _ = self.report(events=2)
        edit_json(self.trajectory(path), lambda value: value["turns"][1]["judge_result"]["hallucination_events"][1].update(event_id="H_T2_1"))
        self.assertRaisesRegex(ValueError, "Duplicate classification event", summarize, self.root)

    def test_missing_trajectory_is_incomplete(self):
        path, _ = self.report()
        self.trajectory(path).unlink()
        result = summarize(self.root)
        self.assertEqual(result["status"], "incomplete")
        self.assertIsNone(result["models"][0]["one_turn_kc_coverage"])

    def test_model_ids_with_slash_preserved(self):
        self.report(model="org/model-name")
        self.assertEqual(summarize(self.root)["models"][0]["model"], "org/model-name")

    def test_measurement_manifest_supplies_task_plan(self):
        self.report(round="round-01")
        (self.root / "measurement.json").write_text(json.dumps({"paper_ids": ["paper"], "target_models": ["unit-model"], "repeats": 2}), encoding="utf-8")
        counts = summarize(self.root)["task_counts"]
        self.assertEqual((counts["planned"], counts["incomplete"]), (2, 1))

    def test_unexpected_report_and_duplicate_plan_cannot_join_silently(self):
        _, planned = self.report()
        self.assertRaisesRegex(ValueError, "Duplicate", summarize, self.root, None, [planned, planned])
        self.report(paper="unplanned")
        self.assertRaisesRegex(ValueError, "Unexpected report", summarize, self.root, None, [planned])

    def test_missing_model_does_not_get_zero_scores(self):
        _, planned = self.report()
        models = {item["model"]: item for item in summarize(self.root, expected_tasks=[planned, {**planned, "model": "missing-model"}])["models"]}
        self.assertEqual(models["missing-model"]["discovered_tasks"], 0)
        self.assertIsNone(models["missing-model"]["one_turn_kc_coverage"])
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
        edit_json(path, lambda value: value["summary"].update(failure_reason=reason))
        summarize(self.root)
        row = self.read_csv("per_paper.csv")[0]
        self.assertEqual((row["model"], row["reason"]), ("'-unit-model", "'" + reason))
        saved = json.loads((self.root / "statistics" / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["models"][0]["model"], "-unit-model")


if __name__ == "__main__":
    unittest.main()
