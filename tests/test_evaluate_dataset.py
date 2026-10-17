"""Offline checks for existing-package testing, independent runs and statistics."""

from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import evaluate_dataset as testing


class DatasetTestingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix=".papergraph-test-", dir=ROOT)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.addCleanup(patch.stopall)
        patch.dict(os.environ, {}, clear=True).start()
        patch.object(testing, "ROOT", self.root).start()
        patch("src.pipeline_plans.PROJECT_ROOT", self.root).start()
        self.config = self.root / "private.env"
        self.config.write_text("EVAL_TARGET_MODEL=fixture-target\nAPI_KEY=secret-value\nBASE_URL=https://judge.test/v1\n", encoding="utf-8")
        self.paper("sample")
        self.check = patch.object(testing, "pipeline_main", return_value=0).start()
        self.validate_graph = patch.object(testing, "validate_graph_inputs", return_value={"fixture": True}).start()
        self.validate_questions = patch.object(testing, "validate_question_inputs").start()
        patch.object(testing, "build_stage_signature", side_effect=self.signature).start()
        self.runner = patch.object(testing, "run_stage").start()
        self.summary = Mock(return_value={"status": "completed", "task_counts": {"completed": 1}})
        patch.dict(sys.modules, {"summarize_results": types.SimpleNamespace(summarize=self.summary, print_summary=Mock())}).start()

    def paper(self, paper_id, root=None):
        folder = (root or self.root / "data") / paper_id
        folder.mkdir(parents=True)
        for name in ("master_graph", "question_templates"):
            (folder / f"{name}.json").write_text(json.dumps({"paper_id": paper_id}), encoding="utf-8")

    def signature(self, job, stage, model):
        root = Path(os.environ["PAPERGRAPH_DATA_ROOT"]) / job.paper_id
        return hashlib.sha256(b"".join((root / name).read_bytes() for name in ("master_graph.json", "question_templates.json"))).hexdigest()

    def call(self, *args):
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            return testing.main(["--config", str(self.config), *args])

    def session(self):
        return next((self.root / "outputs" / "testing").iterdir())

    def test_default_tests_all_existing_packages_once_and_exports_statistics(self):
        self.paper("second")
        self.assertEqual(self.call(), 0)
        self.runner.assert_called_once()
        command, environment, _ = self.runner.call_args.args
        self.assertEqual(command[0], "evaluate")
        self.assertIn("--all", command)
        self.assertNotIn("build", command)
        self.assertEqual(environment["PAPERGRAPH_DATA_ROOT"], str(self.root / "data"))
        manifest = json.loads((self.session() / "testing.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["paper_ids"], ["sample", "second"])
        self.assertEqual(manifest["status"], "completed")
        self.assertEqual(len(manifest["expected_tasks"]), 2)
        self.summary.assert_called_once_with(self.session(), self.session() / "statistics", manifest["expected_tasks"])
        self.assertTrue((self.session() / "main.log").is_file())

    def test_repeats_use_distinct_result_and_log_directories_on_same_inputs(self):
        self.assertEqual(self.call("--repeats", "3", "--models", "alpha", "beta", "--workers", "2"), 0)
        calls = self.runner.call_args_list
        self.assertEqual(len(calls), 3)
        self.assertEqual(len({call.args[1]["EVAL_RESULT_ROOT"] for call in calls}), 3)
        self.assertEqual(len({call.args[1]["PAPERGRAPH_LOG_ROOT"] for call in calls}), 3)
        self.assertEqual(len({call.args[1]["PAPERGRAPH_DATA_ROOT"] for call in calls}), 1)
        self.assertTrue(all(call.args[0][0] == "evaluate" and "--force" in call.args[0] for call in calls))
        self.assertEqual(len(self.summary.call_args.args[2]), 6)

    def test_existing_data_default_ignores_generated_data_root_setting(self):
        with self.config.open("a", encoding="utf-8") as file:
            file.write("PAPERGRAPH_DATA_ROOT=outputs/old-generated-packages\n")
        self.assertEqual(self.call(), 0)
        self.assertEqual(self.runner.call_args.args[1]["PAPERGRAPH_DATA_ROOT"], str(self.root / "data"))

    def test_explicit_custom_dataset_and_one_paper_are_forwarded(self):
        custom = self.root / "custom-data"
        self.paper("other", custom)
        self.assertEqual(self.call("--data-root", str(custom), "--paper-id", "other"), 0)
        command, env, _ = self.runner.call_args.args
        self.assertEqual(env["PAPERGRAPH_DATA_ROOT"], str(custom))
        self.assertEqual(command[command.index("--paper-id") + 1], "other")
        self.assertEqual(json.loads((self.session() / "testing.json").read_text(encoding="utf-8"))["paper_ids"], ["other"])

    def test_preflight_is_evaluation_only_and_failed_configuration_creates_no_session(self):
        self.check.return_value = 2
        self.assertEqual(self.call(), 1)
        self.assertEqual(self.check.call_args.args[0][:3], ["check", "--stage", "evaluate"])
        self.runner.assert_not_called()
        self.summary.assert_not_called()
        self.assertFalse((self.root / "outputs").exists())

    def test_invalid_package_stops_before_provider_or_session(self):
        self.validate_questions.side_effect = ValueError("missing required image")
        self.assertEqual(self.call(), 1)
        self.check.assert_not_called()
        self.runner.assert_not_called()
        self.assertFalse((self.root / "outputs").exists())

    def test_dry_run_has_no_output_writes_or_provider_checks(self):
        before = {str(path.relative_to(self.root)) for path in self.root.rglob("*")}
        self.assertEqual(self.call("--dry-run"), 0)
        self.assertEqual(before, {str(path.relative_to(self.root)) for path in self.root.rglob("*")})
        self.check.assert_not_called()
        self.runner.assert_not_called()
        self.summary.assert_not_called()
        self.validate_graph.assert_not_called()
        self.validate_questions.assert_not_called()

    def test_failed_round_stops_and_never_reports_statistics_as_success(self):
        self.runner.side_effect = RuntimeError("fixture child failure")
        self.assertEqual(self.call("--repeats", "3"), 1)
        self.assertEqual(self.runner.call_count, 1)
        self.summary.assert_not_called()
        self.assertEqual(json.loads((self.session() / "testing.json").read_text(encoding="utf-8"))["status"], "failed")

    def test_mutated_data_stops_later_rounds(self):
        self.runner.side_effect = lambda *args, **kwargs: (self.root / "data" / "sample" / "master_graph.json").write_text("changed")
        self.assertEqual(self.call("--repeats", "2"), 1)
        self.assertEqual(self.runner.call_count, 1)
        self.summary.assert_not_called()
        self.assertEqual(json.loads((self.session() / "testing.json").read_text(encoding="utf-8"))["status"], "failed")

    def test_incomplete_statistics_marks_failed_session(self):
        self.summary.return_value = {"status": "incomplete"}
        self.assertEqual(self.call(), 1)
        self.assertEqual(json.loads((self.session() / "testing.json").read_text(encoding="utf-8"))["status"], "failed")

    def test_manifest_excludes_credentials_and_environment_restored(self):
        before = dict(os.environ)
        self.assertEqual(self.call(), 0)
        self.assertEqual(dict(os.environ), before)
        manifest = (self.session() / "testing.json").read_text(encoding="utf-8")
        self.assertNotIn("secret-value", manifest)
        self.assertNotIn("judge.test", manifest)
        self.assertNotIn("API_KEY", manifest)

    def test_invalid_repeats_and_output_model_collisions_fail_before_writes(self):
        self.assertEqual(self.call("--repeats", "0"), 1)
        self.assertEqual(self.call("--models", "a/b", "a:b"), 1)
        if os.name == "nt":
            self.assertEqual(self.call("--models", "CaseSensitive", "casesensitive"), 1)
        self.runner.assert_not_called()
        self.assertFalse((self.root / "outputs").exists())


class ExistingDatasetHttpTests(unittest.TestCase):
    def setUp(self):
        from test_pipeline_http import PipelineHttpTests
        self.fixture = PipelineHttpTests("test_real_cli_target_judge_and_completed_report")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def _run(self, repeats=1):
        return subprocess.run([sys.executable, str(ROOT / "scripts" / "start.py"), "test",
            "--config", str(self.fixture.config_path), "--data-root", str(self.fixture.root / "data"),
            "--output-root", str(self.fixture.root / "testing"), "--repeats", str(repeats)],
            cwd=ROOT.parent, env=self.fixture._isolated_env(), capture_output=True,
            text=True, encoding="utf-8", errors="replace", timeout=45)

    def test_existing_package_to_two_real_http_evaluations_and_statistics(self):
        before = {path.name: path.read_bytes() for path in (self.fixture.root / "data" / "tiny").iterdir()}
        process = self._run(repeats=2)
        logs = "\n".join(path.read_text(encoding="utf-8") for path in (self.fixture.root / "testing").rglob("*.log"))
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr + logs)
        session = next((self.fixture.root / "testing").iterdir())
        manifest = json.loads((session / "testing.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["status"], "completed")
        self.assertEqual(len(manifest["expected_tasks"]), 2)
        self.assertEqual(json.loads((session / "statistics" / "summary.json").read_text(encoding="utf-8"))["task_counts"]["completed"], 2)
        self.assertTrue((session / "statistics" / "summary.csv").is_file())
        self.assertEqual(before, {path.name: path.read_bytes() for path in (self.fixture.root / "data" / "tiny").iterdir()})

    def test_http_failure_marks_session_failed_without_successful_statistics(self):
        self.fixture.server.reject_target = True
        process = self._run(repeats=2)
        self.assertNotEqual(process.returncode, 0)
        session = next((self.fixture.root / "testing").iterdir())
        self.assertEqual(json.loads((session / "testing.json").read_text(encoding="utf-8"))["status"], "failed")
        self.assertFalse((session / "round-02").exists())
        self.assertFalse((session / "statistics" / "summary.csv").exists())


if __name__ == "__main__":
    unittest.main()
