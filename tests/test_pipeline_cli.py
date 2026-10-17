"""Public selection and completed-output checks without provider requests."""

from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "papergraph_demo"))

from src.config import load_environment
from src.pipeline_artifacts import completed_evaluation
from src.pipeline_cli import main, parser
from src.pipeline_plans import plan_jobs, select_models


class PipelineSelectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix=".papergraph-test-", dir=ROOT)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.addCleanup(patch.stopall)
        patch.dict(os.environ, {}, clear=True).start()
        patch("src.pipeline_plans.PROJECT_ROOT", self.root).start()
        load_environment(self.root)

    def test_defaults_to_one_configured_target(self):
        os.environ["EVAL_TARGET_MODEL"] = "my-target"
        self.assertEqual(select_models(None), ["my-target"])
        os.environ["EVAL_BATCH_MODELS"] = "alpha,beta"
        self.assertEqual(select_models(None), ["alpha", "beta"])
        self.assertEqual(select_models(["explicit"]), ["explicit"])

    def test_directory_paper_id_keeps_dots(self):
        source = self.root / "Mediation2.0-15pages"
        source.mkdir()
        (source / "doc_0.md").write_text("A source document", encoding="utf-8")
        args = parser().parse_args(["build", "--input", str(source)])
        self.assertEqual(plan_jobs(args)[0].paper_id, source.name)

    def test_evaluate_all_uses_packages_without_raw_papers(self):
        paper = self.root / "data" / "package-only"
        paper.mkdir(parents=True)
        for filename in ("master_graph.json", "question_templates.json"):
            (paper / filename).write_text(json.dumps({"paper_id": paper.name}), encoding="utf-8")
        args = parser().parse_args(["evaluate", "--all"])
        jobs = plan_jobs(args)
        self.assertEqual([job.paper_id for job in jobs], [paper.name])
        self.assertFalse((self.root / "rawPaper").exists())

    def test_pdf_name_collisions_fail_before_ocr(self):
        folder = self.root / "pdfInput"
        folder.mkdir()
        for filename in ("a+b.pdf", "a b.pdf"):
            (folder / filename).write_bytes(b"%PDF-fixture")
        with self.assertRaisesRegex(ValueError, "duplicate paper IDs"):
            plan_jobs(parser().parse_args(["parse", "--all"]))

    def test_check_supports_per_model_credentials_without_global_target(self):
        os.environ.update({
            "API_KEY": "fixture-judge", "BASE_URL": "https://example.test/v1", "LLM_MODEL": "judge",
            "EVAL_BATCH_MODELS": "alpha,beta",
            "EVAL_TARGET_API_KEY_ALPHA": "fixture-alpha", "EVAL_TARGET_BASE_URL_ALPHA": "https://alpha.test/v1",
            "EVAL_TARGET_API_KEY_BETA": "fixture-beta", "EVAL_TARGET_BASE_URL_BETA": "https://beta.test/v1",
        })
        with redirect_stdout(io.StringIO()):
            self.assertEqual(main(["check", "--stage", "evaluate"]), 0)
        self.assertNotIn("EVAL_TARGET_MODEL", os.environ)
        self.assertNotIn("EVAL_TARGET_API_KEY", os.environ)

    def test_init_never_overwrites_configuration(self):
        existing = self.root / "config.env"
        existing.write_text("API_KEY=keep-private\n", encoding="utf-8")
        with patch("sys.stderr", new=io.StringIO()):
            self.assertEqual(main(["init", "--config", str(existing)]), 1)
        self.assertEqual(existing.read_text(encoding="utf-8"), "API_KEY=keep-private\n")


class CompletedEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix=".papergraph-test-", dir=ROOT)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.report_path = self.root / "evaluation_report.json"
        self.report = {"paper_id": "paper", "target_model": "model", "summary": {
            "evaluation_status": "completed", "total_turns": 2, "failed": False,
        }}
        self.trajectory = {"paper_id": "paper", "target_model": "model", "turns": [
            {"turn_id": "T1"}, {"turn_id": "T2"},
        ]}

    def write(self):
        self.report_path.write_text(json.dumps(self.report), encoding="utf-8")
        (self.root / "dialogue_trajectory.json").write_text(json.dumps(self.trajectory), encoding="utf-8")

    def test_complete_report_requires_matching_trajectory(self):
        self.write()
        self.assertTrue(completed_evaluation(self.report_path, "paper", "model"))
        self.report["summary"]["total_turns"] = 99
        self.write()
        with self.assertRaisesRegex(ValueError, "turn counts"):
            completed_evaluation(self.report_path, "paper", "model")

    def test_failed_and_duplicate_turns_are_not_success(self):
        self.report["summary"]["failed"] = True
        self.write()
        with self.assertRaisesRegex(ValueError, "failed=false"):
            completed_evaluation(self.report_path, "paper", "model")
        self.report["summary"]["failed"] = False
        self.trajectory["turns"][1]["turn_id"] = "T1"
        self.write()
        with self.assertRaisesRegex(ValueError, "duplicate"):
            completed_evaluation(self.report_path, "paper", "model")

    def test_stopped_or_wrong_model_is_not_completed(self):
        self.report["summary"]["evaluation_status"] = "stopped_by_max_turns"
        self.write()
        self.assertFalse(completed_evaluation(self.report_path, "paper", "model"))
        self.trajectory["target_model"] = "other"
        self.write()
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            completed_evaluation(self.report_path, "paper", "model")

    def test_missing_turns_is_not_completed(self):
        self.trajectory.pop("turns")
        self.write()
        with self.assertRaisesRegex(ValueError, "no dialogue turns"):
            completed_evaluation(self.report_path, "paper", "model")


if __name__ == "__main__":
    unittest.main()
