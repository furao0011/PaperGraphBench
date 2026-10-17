"""Independent offline coverage of final event validation and label summaries."""

from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "papergraph_demo"))
sys.path.insert(0, str(ROOT / "scripts"))
from statistics_classification import EVENT_COLUMNS, collect_classification


class ClassificationStatisticsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix=".papergraph-test-", dir=ROOT)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def write(self, path, payload):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")

    def event(self, event_id="H_T1_1", **changes):
        return {"event_id": event_id, "created_at_turn": "T1", "source_question_id": "Q1",
                "hallucination_family": "challenge_failure", "hallucination_type": "challenge_false_premise_acceptance",
                "subtype": "false_premise", "related_kc_ids": ["KC1"], "claim": "A false premise",
                "status": "resolved", "followup_turns": ["T2"], "resolved_at_turn": "T2", "source": "judge_state", **changes}

    def task(self, events=None, *, model="model", paper="paper", round="round-01"):
        events = events or {}
        directory = self.root / round / "evaluation" / model / paper
        identity = {"paper_id": paper, "target_model": model}
        self.write(directory / "evaluation_report.json", {**identity,
            "summary": {"evaluation_status": "completed", "failed": False, "total_turns": 2},
            "hallucination_metrics": {"hallucination_event_count": len(events),
                "hallucination_by_type": dict(Counter(event.get("hallucination_type") or "unknown" for event in events.values())),
                "hallucination_by_status": dict(Counter(event.get("status") or "unknown" for event in events.values()))}})
        self.write(directory / "dialogue_trajectory.json", {**identity, "turns": [
            {"turn_id": "T1", "question_id": "Q1"}, {"turn_id": "T2", "question_id": "Q2"}]})
        self.write(directory / "cache" / "eval_state_graph.json", {**identity, "hallucination_events": events})
        return {"round": round, "paper_id": paper, "model": model, "status": "completed",
                "report_path": (directory / "evaluation_report.json").relative_to(self.root).as_posix()}

    def edit(self, row, filename, change):
        path = self.root / Path(row["report_path"]).parent / filename
        payload = json.loads(path.read_text(encoding="utf-8"))
        change(payload)
        self.write(path, payload)

    def test_zero_events_preserves_model_and_na_rate(self):
        result = collect_classification(self.root, [self.task()])
        self.assertEqual(result["events"], [])
        self.assertEqual(result["summary"]["status"], "completed")
        self.assertIsNone(result["summary"]["three_fields_present_rate"])
        self.assertEqual(result["summary"]["models"][0]["model"], "model")
        self.assertEqual(result["summary"]["models"][0]["event_count"], 0)

    def test_full_event_scope_trace_fields_and_stable_columns(self):
        row = self.task({"H_T1_1": self.event()})
        result = collect_classification(self.root, [row])
        event = result["events"][0]
        self.assertEqual(set(event), set(EVENT_COLUMNS))
        self.assertEqual(event["related_kc_ids"], ["KC1"])
        self.assertEqual(event["created_at_turn"], "T1")
        self.assertEqual(event["resolved_at_turn"], "T2")
        self.assertEqual(event["status"], "resolved")
        self.assertTrue(event["non_repeating_path"])
        self.assertEqual(event["issues"], [])
        self.assertEqual(result["summary"]["three_fields_present_rate"], 1)

    def test_missing_duplicate_and_state_only_labels_are_not_fabricated(self):
        missing = self.event("H_missing", hallucination_family=None, hallucination_type="", subtype="  ")
        duplicate = self.event("H_duplicate", hallucination_family="review_regression", hallucination_type="review_regression", subtype="hallucination")
        row = self.task({event["event_id"]: event for event in (self.event(), missing, duplicate)})
        result = collect_classification(self.root, [row])
        indexed = {event["event_id"]: event for event in result["events"]}
        self.assertIsNone(indexed["H_missing"]["family"])
        self.assertEqual(indexed["H_missing"]["subtype"], "  ")
        self.assertIn("missing_subtype", indexed["H_missing"]["issues"])
        self.assertIn("duplicate_labels", indexed["H_duplicate"]["issues"])
        self.assertIn("state_only_subtype", indexed["H_duplicate"]["issues"])
        self.assertEqual(result["summary"]["three_fields_present_count"], 2)
        self.assertEqual(result["summary"]["non_repeating_path_count"], 1)

    def test_null_and_literal_null_label_paths_remain_distinct(self):
        row = self.task({"A": self.event("A", subtype=None), "B": self.event("B", subtype="null")})
        paths = collect_classification(self.root, [row])["summary"]["classification_counts"]
        self.assertEqual({path["subtype"] for path in paths}, {None, "null"})

    def test_repeated_local_ids_across_tasks_are_not_deduplicated(self):
        rows = [self.task({"H_T1_1": self.event()}, round=round, paper=paper)
                for round in ("round-01", "round-02") for paper in ("paper-a", "paper-b")]
        result = collect_classification(self.root, list(reversed(rows)))
        self.assertEqual(result["summary"]["event_count"], 4)
        self.assertEqual(len({(event["round"], event["paper_id"], event["event_id"]) for event in result["events"]}), 4)
        self.assertEqual(result, collect_classification(self.root, rows))

    def test_incomplete_tasks_are_excluded_but_models_remain_visible(self):
        completed = self.task(model="zero-events")
        missing = {"round": "round-01", "model": "missing-model", "paper_id": "missing-paper",
                   "status": "missing", "report_path": "does-not-exist.json"}
        result = collect_classification(self.root, [completed, missing])
        self.assertEqual(result["summary"]["status"], "incomplete")
        self.assertEqual(result["summary"]["task_counts"], {"total": 2, "completed": 1, "incomplete": 1})
        models = {model["model"]: model for model in result["summary"]["models"]}
        self.assertIsNone(models["missing-model"]["three_fields_present_rate"])
        self.assertEqual(models["missing-model"]["status"], "incomplete")

    def test_empty_related_ids_are_flagged_not_invented(self):
        row = self.task({"A": self.event("A", related_kc_ids=[])})
        event = collect_classification(self.root, [row])["events"][0]
        self.assertEqual(event["related_kc_ids"], [])
        self.assertIn("empty_related_kc_ids", event["issues"])

    def test_missing_state_raises_instead_of_using_turn_snapshot(self):
        row = self.task()
        (self.root / Path(row["report_path"]).parent / "cache" / "eval_state_graph.json").unlink()
        with self.assertRaisesRegex(ValueError, "Missing final classification state"):
            collect_classification(self.root, [row])

    def test_state_must_be_event_dictionary(self):
        row = self.task()
        self.edit(row, "cache/eval_state_graph.json", lambda data: data.update(hallucination_events=[]))
        with self.assertRaisesRegex(ValueError, "event dictionary"):
            collect_classification(self.root, [row])

    def test_identity_mismatch_in_each_artifact_is_rejected(self):
        for index, filename in enumerate(("evaluation_report.json", "dialogue_trajectory.json", "cache/eval_state_graph.json")):
            row = self.task(paper=f"paper-{index}")
            self.edit(row, filename, lambda data: data.update(target_model="wrong-model"))
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                collect_classification(self.root, [row])

    def test_event_count_and_type_status_distributions_must_match(self):
        changes = {"hallucination_event_count": 2, "hallucination_by_type": {"wrong": 1}, "hallucination_by_status": {"unresolved": 1}}
        for index, (field, value) in enumerate(changes.items()):
            row = self.task({"A": self.event("A")}, paper=f"paper-{index}")
            self.edit(row, "evaluation_report.json", lambda data: data["hallucination_metrics"].update({field: value}))
            with self.assertRaises(ValueError):
                collect_classification(self.root, [row])

    def test_event_key_source_and_repair_references_must_be_valid(self):
        changes = [{"event_id": "wrong"}, {"created_at_turn": "T_missing"},
                   {"source_question_id": "Q_wrong"}, {"followup_turns": ["T_missing"]},
                   {"resolved_at_turn": "T_missing"}, {"related_kc_ids": [""]}]
        for index, change in enumerate(changes):
            event = self.event("A")
            event.update(change)
            row = self.task({"A": event}, paper=f"paper-{index}")
            with self.assertRaises(ValueError):
                collect_classification(self.root, [row])

    def test_duplicate_task_rejected(self):
        row = self.task()
        with self.assertRaisesRegex(ValueError, "Duplicate classification task"):
            collect_classification(self.root, [row, copy.deepcopy(row)])

    def test_unsafe_report_path_rejected(self):
        row = self.task()
        row["report_path"] = "../outside/evaluation_report.json"
        with self.assertRaisesRegex(ValueError, "escapes"):
            collect_classification(self.root, [row])


if __name__ == "__main__":
    unittest.main()
