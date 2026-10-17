"""Frozen-paper regression checks use synthetic statistics and no APIs."""

from __future__ import annotations

import copy
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
from verify_paper_results import REFERENCE, main, verify_paper_results


class PaperVerificationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix=".papergraph-test-verification-", dir=ROOT)
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.reference = json.loads(REFERENCE.read_text(encoding="utf-8"))
        categories = self.reference["category_metrics"]
        models, category_rows = [], []
        for source in self.reference["models"]:
            counts = source["categories"]
            row = {"model": source["model"], "status": "completed", "completed_tasks": 12,
                   "completed_papers": 12, "incomplete_tasks": 0,
                   **dict(zip(self.reference["main_metrics"], source["main"])),
                   **dict(zip(self.reference["repair_metrics"], source["repair"])),
                   "mean_answer_chars_per_turn": source["chars_per_turn"],
                   "hallucination_events_sum": sum(counts)}
            row["total_turns_sum"] = round(row["mean_total_turns"] * 12)
            models.append(row)
            category_rows.append({"model": row["model"], "status": "completed", "task_total": 12,
                                  "completed_tasks": 12, "incomplete_tasks": 0,
                                  **dict(zip(categories, counts)), "categorized_event_count": sum(counts),
                                  "unmapped_event_count": 0, "event_count": sum(counts)})
        # A fully paper-matching synthetic case has no pre-existing cost delta.
        self.reference_without_known_costs = self.directory / "reference-without-known-costs.json"
        reference_without_known = copy.deepcopy(self.reference)
        reference_without_known["known_cost_discrepancies"] = []
        self.reference_without_known_costs.write_text(json.dumps(reference_without_known), encoding="utf-8")
        self.summary = {"status": "completed", "models": models,
                        "task_counts": {"planned": None, "discovered": 120, "completed": 120, "incomplete": 0},
                        "classification": {"event_count": 1491, "categorized_event_count": 1491, "unmapped_event_count": 0}}
        self.categories = category_rows
        self.save()

    def save(self):
        (self.directory / "summary.json").write_text(json.dumps(self.summary), encoding="utf-8")
        with (self.directory / "failure_category_by_model.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(self.categories[0]))
            writer.writeheader()
            writer.writerows(self.categories)

    def model(self, name):
        return next(row for row in self.summary["models"] if row["model"] == name)

    def test_frozen_reference_preserves_all_ten_models_and_150_core_cells(self):
        self.assertEqual(len(self.reference["models"]), 10)
        self.assertIn("fireworks/kimi-k2p5", {row["model"] for row in self.reference["models"]})
        result = verify_paper_results(self.directory, self.reference_without_known_costs)
        self.assertEqual(result["status"], "passed")
        for key, expected in (("main_metrics", 70), ("repair_metrics", 40), ("failure_categories", 40), ("cost", 10)):
            self.assertEqual(result["checks"][key], {"checked": expected, "matched": expected})

    def test_glm_cost_discrepancy_is_explicit_and_never_replaced(self):
        actual = 1737.0548245614036
        self.model("glm-4.5v")["mean_answer_chars_per_turn"] = actual
        self.save()
        result = verify_paper_results(self.directory)
        self.assertEqual(result["status"], "core_pass_with_cost_discrepancy")
        self.assertEqual(result["core_discrepancies"], [])
        self.assertEqual(result["checks"]["cost"], {"checked": 10, "matched": 9})
        issue = result["cost_discrepancies"][0]
        self.assertEqual(issue["model"], "glm-4.5v")
        self.assertEqual(issue["actual"], actual)
        self.assertEqual(issue["reference"], 1736.1)
        self.assertEqual(result["known_cost_discrepancies"], [issue])
        self.assertEqual(result["unexpected_cost_discrepancies"], [])
        unchanged = json.loads((self.directory / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(next(row for row in unchanged["models"] if row["model"] == "glm-4.5v")["mean_answer_chars_per_turn"], actual)

    def test_new_cost_error_fails_instead_of_reusing_known_warning(self):
        self.model("gpt-5")["mean_answer_chars_per_turn"] = 900
        self.save()
        result = verify_paper_results(self.directory)
        self.assertEqual(result["status"], "failed_cost_regression")
        self.assertEqual(result["core_discrepancies"], [])
        self.assertEqual(result["known_cost_discrepancies"], [])
        self.assertEqual(result["unexpected_cost_discrepancies"][0]["actual"], 900)
        with redirect_stdout(io.StringIO()):
            self.assertEqual(main(["--statistics-dir", str(self.directory)]), 2)

    def test_changed_glm_actual_is_not_permitted_by_model_name(self):
        self.model("glm-4.5v")["mean_answer_chars_per_turn"] = 1737.1
        self.save()
        result = verify_paper_results(self.directory)
        self.assertEqual(result["status"], "failed_cost_regression")
        self.assertEqual(result["known_cost_discrepancies"], [])
        self.assertEqual(result["unexpected_cost_discrepancies"][0]["actual"], 1737.1)
        with redirect_stdout(io.StringIO()):
            self.assertEqual(main(["--statistics-dir", str(self.directory)]), 2)

    def test_glm_changing_to_paper_number_is_still_archived_cost_regression(self):
        self.model("glm-4.5v")["mean_answer_chars_per_turn"] = 1736.1
        self.save()
        result = verify_paper_results(self.directory)
        self.assertEqual(result["status"], "failed_cost_regression")
        issue = result["unexpected_cost_discrepancies"][0]
        self.assertEqual(issue["actual"], 1736.1)
        self.assertEqual(issue["expected_archived_actual"], 1737.0548245614036)

    def test_display_rounding_tolerance_does_not_create_false_cost_errors(self):
        self.model("gpt-5")["mean_answer_chars_per_turn"] = 832.2246575342466
        self.save()
        self.assertEqual(verify_paper_results(self.directory, self.reference_without_known_costs)["status"], "passed")

    def test_main_repair_and_category_mismatches_are_reported(self):
        self.model("gpt-5-mini")["one_turn_kc_coverage"] = .5
        self.model("gpt-5-mini")["hallucination_repair_success_rate"] = 1
        self.categories[2]["challenge_false_premise_acceptance"] += 1
        self.save()
        result = verify_paper_results(self.directory)
        self.assertEqual(result["status"], "core_failed")
        fields = {issue["field"] for issue in result["core_discrepancies"]}
        self.assertTrue({"one_turn_kc_coverage", "hallucination_repair_success_rate",
                         "challenge_false_premise_acceptance"} <= fields)

    def test_wrong_task_inventory_unmapped_and_extra_models_fail(self):
        self.summary["task_counts"]["completed"] = 119
        self.summary["classification"]["unmapped_event_count"] = 1
        extra = copy.deepcopy(self.summary["models"][0])
        extra["model"] = "unexpected-model"
        self.summary["models"].append(extra)
        self.save()
        result = verify_paper_results(self.directory)
        self.assertEqual(result["status"], "core_failed")
        self.assertTrue(any(issue["field"] == "summary.models" for issue in result["core_discrepancies"]))

    def test_missing_metrics_and_duplicate_model_rows_are_invalid_not_na(self):
        self.model("gpt-5").pop("one_turn_kc_coverage")
        self.save()
        with self.assertRaisesRegex(ValueError, "finite number"):
            verify_paper_results(self.directory)
        self.summary["models"].append(copy.deepcopy(self.summary["models"][0]))
        self.save()
        with self.assertRaisesRegex(ValueError, "duplicate model"):
            verify_paper_results(self.directory)

    def test_cli_output_optional_report_and_core_mismatch_exit(self):
        destination = self.directory / "verification.json"
        self.model("glm-4.5v")["mean_answer_chars_per_turn"] = 1737.0548245614036
        self.save()
        with redirect_stdout(io.StringIO()) as stdout:
            status = main(["--statistics-dir", str(self.directory), "--output", str(destination)])
        self.assertEqual(status, 0)
        self.assertEqual(json.loads(stdout.getvalue())["status"], "core_pass_with_cost_discrepancy")
        self.assertTrue(destination.is_file())
        self.model("gpt-5")["mean_total_turns"] = 1
        self.save()
        with redirect_stdout(io.StringIO()):
            self.assertEqual(main(["--statistics-dir", str(self.directory)]), 2)

    def test_missing_files_return_clear_error(self):
        (self.directory / "failure_category_by_model.csv").unlink()
        with redirect_stderr(io.StringIO()) as stderr:
            self.assertEqual(main(["--statistics-dir", str(self.directory)]), 1)
        self.assertIn("Paper verification error", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
