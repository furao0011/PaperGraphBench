"""Regression tests for the publication's original four-category mapping."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from statistics_taxonomy import (
    PAPER_CATEGORIES, PAPER_CATEGORY_COLUMNS, PAPER_CATEGORY_LABELS,
    collapsed_failure_type, paper_category_rows,
)


class PaperTaxonomyTests(unittest.TestCase):
    @staticmethod
    def task(model="model", paper="paper", round="round-01", status="completed"):
        return {"model": model, "paper_id": paper, "round": round, "status": status}

    def event(self, raw_type, *, event_id="H_T1_1", **scope):
        return {**self.task(**scope), "event_id": event_id, "type": raw_type}

    def test_four_categories_and_original_table_labels_have_stable_order(self):
        self.assertEqual(PAPER_CATEGORIES, (
            "contradicted_kc_claim", "challenge_false_premise_acceptance",
            "challenge_overclaim_acceptance", "challenge_wrong_relation_acceptance"))
        self.assertEqual([PAPER_CATEGORY_LABELS[key]["en"] for key in PAPER_CATEGORIES],
                         ["Contrad. KC", "False Prem.", "Overclaim", "Wrong Rel."])
        self.assertTrue(all(PAPER_CATEGORY_LABELS[key]["zh"] for key in PAPER_CATEGORIES))

    def test_exact_nine_raw_type_mapping_matches_the_paper(self):
        expected = {
            "contradicted_kc_claim": "contradicted_kc_claim",
            "review_regression": "contradicted_kc_claim",
            "challenge_false_premise_acceptance": "challenge_false_premise_acceptance",
            "thread_premise_mutation_acceptance": "challenge_false_premise_acceptance",
            "challenge_overclaim_acceptance": "challenge_overclaim_acceptance",
            "thread_overclaim_acceptance": "challenge_overclaim_acceptance",
            "challenge_wrong_relation_acceptance": "challenge_wrong_relation_acceptance",
            "thread_wrong_bridge_acceptance": "challenge_wrong_relation_acceptance",
            "thread_reasoning_hallucination": "challenge_wrong_relation_acceptance",
        }
        for raw_type, category in expected.items():
            with self.subTest(raw_type=raw_type):
                self.assertEqual(collapsed_failure_type(raw_type), category)
        events = [self.event(raw, event_id=str(index)) for index, raw in enumerate(expected)]
        result = paper_category_rows(events, [self.task()])[0]
        self.assertEqual([result[key] for key in PAPER_CATEGORIES], [2, 2, 2, 3])
        self.assertEqual(result["categorized_event_count"], 9)
        self.assertEqual(result["unmapped_event_count"], 0)

    def test_generic_fabricated_unknown_and_blank_types_are_explicitly_unmapped(self):
        values = ["fabricated_claim", "challenge_failure", "HALLUCINATION", "unknown", "", "  ",
                  None, "Contradicted_KC_Claim", " contradicted_kc_claim "]
        for raw in values:
            with self.subTest(raw_type=raw):
                self.assertIsNone(collapsed_failure_type(raw))
        result = paper_category_rows([self.event(raw, event_id=str(index)) for index, raw in enumerate(values)], [self.task()])[0]
        self.assertEqual(result["event_count"], len(values))
        self.assertEqual(result["unmapped_event_count"], len(values))
        self.assertEqual(result["categorized_event_count"], 0)
        self.assertTrue(all(result[key] == 0 for key in PAPER_CATEGORIES))

    def test_no_events_keeps_all_four_zero_categories_and_csv_columns(self):
        result = paper_category_rows([], [self.task()])[0]
        self.assertEqual(set(result), set(PAPER_CATEGORY_COLUMNS))
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["event_count"], 0)
        self.assertTrue(all(result[key] == 0 for key in PAPER_CATEGORIES))

    def test_partial_models_keep_status_and_exclude_incomplete_task_events(self):
        rows = [self.task(), self.task(paper="missing", status="missing"),
                self.task(model="no-completed", status="incomplete")]
        events = [self.event("review_regression"), self.event("review_regression", paper="missing"),
                  self.event("review_regression", model="no-completed")]
        result = {row["model"]: row for row in paper_category_rows(events, rows)}
        self.assertEqual(result["model"]["event_count"], 1)
        self.assertEqual(result["model"]["status"], "incomplete")
        self.assertEqual(result["model"]["task_total"], 2)
        self.assertEqual(result["model"]["completed_tasks"], 1)
        self.assertEqual(result["no-completed"]["event_count"], 0)
        self.assertEqual(result["no-completed"]["status"], "incomplete")

    def test_local_event_ids_are_not_merged_across_rounds_papers_or_models(self):
        scopes = [{"model": model, "paper": paper, "round": round}
                  for model in ("a", "b") for paper in ("p1", "p2") for round in ("round-01", "round-02")]
        rows = [self.task(**scope) for scope in scopes]
        events = [self.event("review_regression", **scope) for scope in scopes]
        result = paper_category_rows(list(reversed(events)), list(reversed(rows)))
        self.assertEqual([row["model"] for row in result], ["a", "b"])
        self.assertEqual([row["event_count"] for row in result], [4, 4])
        self.assertEqual([row["contradicted_kc_claim"] for row in result], [4, 4])

    def test_unmapped_events_are_not_inferred_from_family_or_subtype(self):
        event = self.event("challenge_failure")
        event.update(family="contradicted_kc_claim", subtype="challenge_overclaim_acceptance")
        result = paper_category_rows([event], [self.task()])[0]
        self.assertEqual(result["unmapped_event_count"], 1)
        self.assertEqual(result["categorized_event_count"], 0)

    def test_mapped_and_unmapped_counts_reconcile_exactly(self):
        events = [self.event("review_regression"), self.event("fabricated_claim", event_id="H_T2_1")]
        result = paper_category_rows(events, [self.task()])[0]
        self.assertEqual(sum(result[key] for key in PAPER_CATEGORIES), result["categorized_event_count"])
        self.assertEqual(result["categorized_event_count"] + result["unmapped_event_count"], result["event_count"])

    def test_unknown_task_or_same_scoped_event_duplicate_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "no task"):
            paper_category_rows([self.event("review_regression", paper="not-planned")], [self.task()])
        event = self.event("review_regression")
        with self.assertRaisesRegex(ValueError, "Duplicate scoped"):
            paper_category_rows([event, dict(event)], [self.task()])


if __name__ == "__main__":
    unittest.main()
