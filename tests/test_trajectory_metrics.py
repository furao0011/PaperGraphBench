"""Paper metric definitions use recorded turns, not final-state report aliases."""

from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from statistics_metrics import METRICS, aggregate, count, extract_metrics
from statistics_repairs import extract_repair_tasks


def turn(index, kind="macro_main_question", *, covered=None, targets=None, complete=None,
         event_types=(), state="MAIN_PROGRESS", failed=None, multimodal=False, context=None):
    targets = ["K1", "K2"] if targets is None else targets
    covered = targets if covered is None else covered
    coverage = {"target_kc_ids": targets, "covered_kc_ids": covered,
                "missing_kc_ids": [key for key in targets if key not in covered]}
    if complete is not None:
        coverage["coverage_complete"] = complete
    judge = {"state": state, "coverage": coverage,
             "hallucination_events": [{"hallucination_type": kind} for kind in event_types]}
    if failed is not None:
        judge["challenge_result"] = {"failed": failed}
    return {"turn_id": f"T{index}", "question_id": f"Q{index}", "question_type": kind,
            "model_answer": "abc", "requires_multimodal_input": multimodal,
            "asset_references": [], "judge_result": judge, "repair_context": context or {}}


def metrics(turns):
    return extract_metrics({"summary": {"total_turns": len(turns)},
                            "coverage_metrics": {"global_kc_coverage_rate": 1.0},
                            "macro_metrics": {"macro_completion_rate": 1.0}}, {"turns": turns})


class TrajectoryMetricTests(unittest.TestCase):
    def test_first_macro_answers_ignore_recovered_final_coverage(self):
        values = metrics([turn(1, covered=["K1"]), turn(2, "detail_followup")])
        self.assertEqual(values["one_turn_kc_coverage"], .5)
        self.assertEqual(values["one_turn_kc_coverage_numerator"], 1)
        self.assertEqual(values["one_turn_macro_complete_rate"], 0)
        self.assertNotIn("global_kc_coverage", values)
        self.assertIsNone(values["detail_repair_success_rate"])

    def test_main_alias_and_explicit_complete_flag_follow_paper(self):
        values = metrics([turn(1, "main", covered=["K1"], complete=True)])
        self.assertEqual(values["one_turn_kc_coverage"], .5)
        self.assertEqual(values["one_turn_macro_complete_rate"], 1)

    def test_only_ordinary_challenges_and_original_failure_contract(self):
        turns = [turn(1),
                 turn(2, "challenge_question", failed=False,
                      event_types=["challenge_false_premise_acceptance"]),
                 turn(3, "challenge_question", failed=True, multimodal=True),
                 turn(4, "thread_challenge_question", failed=True,
                      event_types=["thread_wrong_bridge_acceptance"])]
        values = metrics(turns)
        self.assertEqual(values["challenge_failure_rate_numerator"], 1)
        self.assertEqual(values["challenge_failure_rate_denominator"], 2)
        self.assertEqual(values["text_challenge_failure_rate"], 0)
        self.assertEqual(values["multimodal_challenge_failure_rate"], 1)
        self.assertEqual(values["mean_hallucination_events"], 2)

    def test_legacy_challenge_event_and_asset_reference_modalities(self):
        challenge = turn(1, "challenge_question", event_types=["challenge_wrong_relation_acceptance"])
        challenge["asset_references"] = [{"asset_id": "figure-1"}]
        values = metrics([challenge])
        self.assertEqual(values["challenge_failure_rate"], 1)
        self.assertEqual(values["multimodal_challenge_failure_rate"], 1)
        self.assertIsNone(values["text_challenge_failure_rate"])

    def test_hallucination_turns_count_events_not_persistent_state(self):
        values = metrics([turn(1, state="HALLUCINATION"),
                          turn(2, event_types=["contradicted_kc_claim"] * 3)])
        self.assertEqual(values["hallucination_turn_rate"], .5)
        self.assertEqual(values["hallucination_event_density"], 1.5)
        self.assertEqual(values["mean_hallucination_events"], 3)
        self.assertEqual(values["mean_hallucination_events_denominator"], 1)
        self.assertEqual(values["mean_total_turns"], 2)
        self.assertEqual(values["mean_answer_chars_per_turn"], 3)

    def test_explicit_legacy_coverage_is_supported_but_missing_data_fails(self):
        legacy = turn(1, covered=["K1"])
        coverage = legacy["judge_result"].pop("coverage")
        legacy["target_kc_ids"] = coverage["target_kc_ids"]
        legacy["judge_result"]["covered_kc_ids"] = coverage["covered_kc_ids"]
        self.assertEqual(metrics([legacy])["one_turn_kc_coverage"], .5)
        del legacy["judge_result"]["covered_kc_ids"]
        with self.assertRaisesRegex(ValueError, "missing covered_kc_ids"):
            metrics([legacy])

    def test_zero_events_are_zero_while_unexecuted_branches_are_na(self):
        values = metrics([turn(1)])
        self.assertEqual(values["mean_hallucination_events"], 0)
        self.assertEqual(values["hallucination_turn_rate"], 0)
        for name in ("challenge_failure_rate", "detail_repair_success_rate", "hallucination_repair_avg_rounds"):
            self.assertIsNone(values[name])
            self.assertEqual(values[name + "_denominator"], 0)

    def test_malformed_sources_raise_instead_of_using_report_counts(self):
        changes = [lambda item: item.pop("model_answer"),
                   lambda item: item["judge_result"].update(hallucination_events=None),
                   lambda item: item["judge_result"]["coverage"].update(coverage_complete="true")]
        for change in changes:
            item = turn(1)
            change(item)
            with self.assertRaises(ValueError):
                metrics([item])
        with self.assertRaisesRegex(ValueError, "total_turns disagrees"):
            extract_metrics({"summary": {"total_turns": 2}}, {"turns": [turn(1)]})
        with self.assertRaisesRegex(ValueError, "unique"):
            metrics([turn(1), turn(1)])


class TrajectoryRepairTests(unittest.TestCase):
    def test_tasks_group_by_root_and_stop_at_first_success_not_final_event_state(self):
        context = {"repair_type": "hallucination", "root_turn_id": "T1", "root_question_id": "Q1"}
        trajectory = {"turns": [turn(1, event_types=["challenge_overclaim_acceptance"]),
            turn(2, "hallucination_followup", context=context, state="HALLUCINATION"),
            turn(3, "thread_question"),
            turn(4, "hallucination_followup", context=context, state="SELF_CORRECTED"),
            turn(5, "hallucination_followup", context=context, state="HALLUCINATION")]}
        task = extract_repair_tasks(trajectory)[0]
        self.assertTrue(task["success"])
        self.assertEqual(task["attempts"], 2)
        self.assertEqual(task["turn_span"], 3)
        self.assertEqual(task["success_turn_id"], "T4")
        values = metrics(trajectory["turns"])
        self.assertEqual(values["hallucination_repair_success_rate"], 1)
        self.assertEqual(values["hallucination_repair_avg_rounds"], 2)

    def test_detail_remaining_kcs_and_exhausted_hallucination_task(self):
        detail = {"repair_type": "detail", "root_turn_id": "T1", "root_question_id": "Q1", "remaining_kc_ids": ["K2"]}
        hallucination = {"repair_type": "hallucination", "root_turn_id": "T2", "root_question_id": "Q2"}
        turns = [turn(1, covered=[]), turn(2, event_types=["contradicted_kc_claim"]),
                 turn(3, "detail_followup", covered=["K2"], complete=False, context=detail),
                 *(turn(index, "hallucination_followup", state="REFUSE_TO_CORRECT", context=hallucination) for index in (4, 5, 6))]
        tasks = extract_repair_tasks({"turns": turns})
        self.assertEqual(len(tasks), 2)
        self.assertTrue(tasks[0]["success"])
        self.assertTrue(tasks[1]["exhausted"])
        self.assertEqual(metrics(turns)["detail_repair_success_rate"], 1)
        self.assertEqual(metrics(turns)["hallucination_repair_success_rate"], 0)
        self.assertEqual(metrics(turns)["hallucination_repair_avg_rounds"], 3)

    def test_bad_repair_roots_are_not_silently_reconstructed(self):
        for context in ({"repair_type": "detail"},
                        {"repair_type": "detail", "root_turn_id": "missing", "root_question_id": "Q1"},
                        {"repair_type": "detail", "root_turn_id": "T1", "root_question_id": "wrong"}):
            with self.assertRaises(ValueError):
                extract_repair_tasks({"turns": [turn(1), turn(2, "detail_followup", context=context)]})


class PaperAggregationTests(unittest.TestCase):
    def row(self, paper, round, covered, targets, failed, challenges):
        turns = [turn(1, targets=[f"K{index}" for index in range(targets)], covered=[f"K{index}" for index in range(covered)])]
        turns.extend(turn(index + 2, "challenge_question", failed=index < failed,
                          event_types=["challenge_overclaim_acceptance"] if index < failed else [])
                     for index in range(challenges))
        return {"paper_id": paper, "model": "model", "round": round, "status": "completed", **metrics(turns)}

    def test_primary_paper_means_vs_pooled_rates_with_unequal_repetitions(self):
        rows = [self.row("a", "round-01", 1, 1, 1, 1),
                self.row("a", "round-02", 0, 1, 0, 1),
                self.row("b", "round-01", 0, 9, 0, 9)]
        result = aggregate(rows, None)[0]
        self.assertEqual(result["one_turn_kc_coverage"], .25)
        self.assertAlmostEqual(result["one_turn_kc_coverage_micro"], 1 / 11)
        self.assertAlmostEqual(result["challenge_failure_rate"], 1 / 11)
        self.assertEqual(result["challenge_failure_rate_paper_mean"], .25)
        self.assertEqual(result["mean_hallucination_events"], .25)
        self.assertEqual(result["mean_total_turns"], 6)
        self.assertEqual(result["hallucination_events_sum"], 1)
        self.assertEqual(result["total_turns_sum"], 14)
        self.assertNotIn("challenge_failure_rate_mean", result)
        self.assertTrue(all(key in result for key in METRICS))

    def test_missing_tasks_do_not_become_zero_scored_models(self):
        missing = {"paper_id": "a", "model": "model", "round": "single", "status": "missing"}
        result = aggregate([missing], [missing])[0]
        self.assertEqual(result["status"], "incomplete")
        self.assertIsNone(result["mean_hallucination_events"])
        self.assertIsNone(result["hallucination_events_sum"])
        self.assertEqual(result["completed_tasks"], 0)

    def test_count_integrity_and_metric_pairs_are_validated(self):
        row = self.row("a", "single", 1, 2, 0, 1)
        bad = copy.deepcopy(row)
        bad["challenge_failure_rate"] = .5
        with self.assertRaisesRegex(ValueError, "disagrees"):
            aggregate([bad], None)
        for value in (True, -1, 1.5):
            with self.assertRaises(ValueError):
                count(value, "count")


if __name__ == "__main__":
    unittest.main()
