"""Offline coverage of the one-command fixed-package measurement workflow."""

from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_measurement as measurement


class MeasurementTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix=".papergraph-test-", dir=ROOT)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.addCleanup(patch.stopall)
        patch.dict(os.environ, {}, clear=True).start()
        patch.object(measurement, "ROOT", self.root).start()
        self.config = self.root / "private.env"
        self.config.write_text("EVAL_TARGET_MODEL=fixture-target\nAPI_KEY=fixture-secret-key\nBASE_URL=https://secret.test/v1\n", encoding="utf-8")
        self.paper("sample")
        self.calls = []
        self.check = patch.object(measurement, "pipeline_main", return_value=0).start()
        self.runner = patch.object(measurement, "run_stage", side_effect=self.stage).start()

    def paper(self, paper_id):
        path = self.root / "rawPaper" / paper_id
        path.mkdir(parents=True)
        (path / "doc_0.md").write_text("A source document", encoding="utf-8")
        (path / "figure.png").write_bytes(b"fixture-image")
        (path / "cache").mkdir()
        (path / "cache" / "private.json").write_text("not-an-input", encoding="utf-8")

    def stage(self, argv, env, log, *, project_root):
        self.calls.append((list(argv), dict(env)))
        dataset = Path(env["PAPERGRAPH_DATA_ROOT"])
        if argv[0] in {"build", "questions"}:
            for raw in Path(env["RAW_PAPER_ROOT"]).iterdir():
                folder = dataset / raw.name
                folder.mkdir(parents=True, exist_ok=True)
                name = "master_graph.json" if argv[0] == "build" else "question_templates.json"
                (folder / name).write_text(json.dumps({"paper_id": raw.name}), encoding="utf-8")
        else:
            Path(env["EVAL_RESULT_ROOT"]).mkdir(parents=True)

    def run_main(self, *args):
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            return measurement.main(["--config", str(self.config), *args])

    def session(self):
        return next((self.root / "outputs" / "measurement").iterdir())

    def test_build_once_then_three_independent_rounds(self):
        self.assertEqual(self.run_main(), 0)
        self.assertEqual([argv[0] for argv, _ in self.calls], ["build", "questions", "evaluate", "evaluate", "evaluate"])
        roots = [env["EVAL_RESULT_ROOT"] for argv, env in self.calls if argv[0] == "evaluate"]
        self.assertEqual(len(set(roots)), 3)
        self.assertTrue(all("round-" in root for root in roots))
        self.assertEqual(len({env["PAPERGRAPH_LOG_ROOT"] for _, env in self.calls}), 5)
        self.assertTrue(all("--force" in argv for argv, _ in self.calls))
        frozen = json.loads((self.session() / "frozen_inputs.json").read_text(encoding="utf-8"))["files"]
        self.assertIn("dataset/sample/master_graph.json", frozen)
        self.assertIn("dataset/sample/question_templates.json", frozen)
        self.assertIn("rawPaper/sample/figure.png", frozen)
        self.assertFalse(any("cache/" in name for name in frozen))

    def test_preflight_precedes_output_creation_and_stages(self):
        def check(argv):
            self.assertFalse((self.root / "outputs").exists())
            self.assertFalse(self.calls)
            return 2
        self.check.side_effect = check
        self.assertEqual(self.run_main(), 1)
        self.runner.assert_not_called()
        self.assertFalse((self.root / "outputs").exists())

    def test_real_preflight_rejects_incomplete_api_configuration(self):
        from src.pipeline_cli import main as real_pipeline_main
        self.check.side_effect = real_pipeline_main
        self.assertEqual(self.run_main(), 1)
        self.runner.assert_not_called()
        self.assertFalse((self.root / "outputs").exists())

    def test_selection_and_config_are_forwarded(self):
        self.paper("other")
        self.assertEqual(self.run_main("--paper-id", "sample", "--models", "alpha", "beta", "--workers", "2", "--repeats", "1"), 0)
        for argv, _ in self.calls:
            self.assertIn("--paper-id", argv)
            self.assertIn("sample", argv)
            self.assertIn(str(self.config), argv)
            self.assertEqual(argv[argv.index("--workers") + 1], "2")
        self.assertEqual(self.calls[-1][0][-3:], ["--models", "alpha", "beta"])
        self.assertEqual(sorted(path.name for path in (self.session() / "rawPaper").iterdir()), ["sample"])

    def test_manifests_exclude_credentials_and_environment_is_restored(self):
        before = dict(os.environ)
        self.assertEqual(self.run_main(), 0)
        self.assertEqual(dict(os.environ), before)
        raw = (self.session() / "measurement.json").read_text(encoding="utf-8")
        self.assertNotIn("fixture-secret-key", raw)
        self.assertNotIn("secret.test", raw)
        self.assertNotIn("API_KEY", raw)
        manifest = json.loads(raw)
        self.assertEqual(manifest["status"], "completed")
        self.assertEqual(manifest["paper_count"], 1)
        self.assertIn("source_revision", manifest)

    def test_dry_run_does_not_write_or_check_apis(self):
        before = sorted(str(path.relative_to(self.root)) for path in self.root.rglob("*"))
        self.assertEqual(self.run_main("--dry-run"), 0)
        self.assertEqual(sorted(str(path.relative_to(self.root)) for path in self.root.rglob("*")), before)
        self.check.assert_not_called()
        self.runner.assert_not_called()

    def test_failed_stage_stops_without_later_stages(self):
        def fail(argv, *args, **kwargs):
            self.calls.append((argv, {}))
            raise RuntimeError("fixture failure")
        self.runner.side_effect = fail
        self.assertEqual(self.run_main(), 1)
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(json.loads((self.session() / "measurement.json").read_text())["status"], "failed")

    def test_mutated_frozen_inputs_stop_remaining_rounds(self):
        def mutate(argv, env, log, *, project_root):
            self.stage(argv, env, log, project_root=project_root)
            if argv[0] == "evaluate":
                (Path(env["PAPERGRAPH_DATA_ROOT"]) / "sample" / "master_graph.json").write_text("changed", encoding="utf-8")
        self.runner.side_effect = mutate
        self.assertEqual(self.run_main(), 1)
        self.assertEqual([argv[0] for argv, _ in self.calls], ["build", "questions", "evaluate"])

    def test_invalid_repeats_and_path_selection_fail_without_writes(self):
        self.assertEqual(self.run_main("--repeats", "0"), 1)
        self.assertEqual(self.run_main("--paper-id", "../sample"), 1)
        self.assertFalse((self.root / "outputs").exists())

    def test_all_papers_is_discovered_not_hardcoded(self):
        self.paper("second")
        self.assertEqual(self.run_main("--repeats", "1"), 0)
        manifest = json.loads((self.session() / "measurement.json").read_text())
        self.assertEqual(manifest["paper_ids"], ["sample", "second"])
        self.assertTrue(all("--all" in argv for argv, _ in self.calls))

    def test_source_archive_explicitly_records_unavailable_revision(self):
        self.assertEqual(self.run_main("--repeats", "1"), 0)
        revision = json.loads((self.session() / "measurement.json").read_text())["source_revision"]
        self.assertIsNone(revision["commit"])
        self.assertIn("unavailable", revision["status"])

    def test_private_config_files_are_not_copied_with_source_assets(self):
        (self.root / "rawPaper" / "sample" / "config.env").write_text("API_KEY=must-not-copy", encoding="utf-8")
        self.assertEqual(self.run_main("--repeats", "1"), 0)
        self.assertFalse((self.session() / "rawPaper" / "sample" / "config.env").exists())


if __name__ == "__main__":
    unittest.main()
