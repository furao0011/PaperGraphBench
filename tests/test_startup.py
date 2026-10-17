"""Offline checks for one-command setup, mandatory API settings and forwarding."""

from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import start as startup
from src.pipeline_plans import PaperJob


class StartupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix=".papergraph-test-", dir=ROOT)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        shutil.copyfile(ROOT / "config.example.env", self.root / "config.example.env")
        self.addCleanup(patch.stopall)
        patch.dict(os.environ, {}, clear=True).start()
        patch.object(startup, "ROOT", self.root).start()
        patch("src.pipeline_cli.PROJECT_ROOT", self.root).start()
        patch("src.pipeline_plans.PROJECT_ROOT", self.root).start()
        self.stdout, self.stderr = io.StringIO(), io.StringIO()

    def call(self, args):
        with redirect_stdout(self.stdout), redirect_stderr(self.stderr):
            return startup.main(args)

    def write_config(self, extra="", *, full=False):
        text = (
            "API_KEY=secret-judge-value\nBASE_URL=https://judge.example.test/v1\nLLM_MODEL=judge\n"
            "EVAL_TARGET_API_KEY=secret-target-value\n"
            "EVAL_TARGET_BASE_URL=https://target.example.test/v1\nEVAL_TARGET_MODEL=target\n"
        )
        if full:
            text += (
                "VISION_API_KEY=secret-vision-value\nVISION_BASE_URL=https://vision.example.test/v1\nVISION_MODEL=vision\n"
                "EMBED_API_KEY=secret-embed-value\nEMBED_BASE_URL=https://embed.example.test/v1\nEMBED_MODEL=embed\n"
            )
        path = self.root / "config.env"
        path.write_text(text + extra, encoding="utf-8")
        return path

    def test_no_arguments_select_public_example_and_existing_cli(self):
        tokens, args = startup.startup_arguments([])
        self.assertEqual(tokens, ["run", "--paper-id", "ConVis"])
        self.assertEqual(args.command, "run")

    def test_batch_and_explicit_command_options_are_forwarded(self):
        tokens, args = startup.startup_arguments(["--all", "--workers", "2"])
        self.assertEqual(tokens, ["run", "--all", "--workers", "2"])
        self.assertTrue(args.all)
        tokens, args = startup.startup_arguments(["evaluate", "--all", "--models", "alpha", "beta"])
        self.assertEqual(args.models, ["alpha", "beta"])
        self.assertNotIn("ConVis", tokens)

    def test_config_option_before_or_after_command(self):
        for tokens in (["--config", "settings/private.env", "evaluate"], ["evaluate", "--config", "settings/private.env"]):
            _, args = startup.startup_arguments(tokens)
            self.assertEqual(args.config, "settings/private.env")
            self.assertEqual(startup.configuration_path(args.config), self.root / "settings/private.env")

    def test_missing_config_created_then_missing_fields_stop_before_execution(self):
        with patch("src.pipeline_cli.execute_jobs") as execute:
            status = self.call([])
        self.assertNotEqual(status, 0)
        self.assertTrue((self.root / "config.env").is_file())
        execute.assert_not_called()
        for field in ("API_KEY", "BASE_URL", "LLM_MODEL", "VISION_API_KEY", "EMBED_BASE_URL", "EVAL_TARGET_MODEL"):
            self.assertIn(field, self.stdout.getvalue() + self.stderr.getvalue())
        self.assertIn(str(self.root / "config.env"), self.stdout.getvalue())

    def test_existing_config_is_never_overwritten(self):
        path = self.write_config()
        before = path.read_bytes()
        self.assertFalse(startup.create_configuration(path))
        self.assertEqual(path.read_bytes(), before)
        with patch("src.pipeline_cli.plan_jobs", return_value=[PaperJob("ConVis", ("evaluate",))]), patch("src.pipeline_cli.execute_jobs") as execute:
            self.assertEqual(self.call(["evaluate"]), 0)
        execute.assert_called_once()
        self.assertEqual(path.read_bytes(), before)

    def test_evaluation_requires_only_judge_and_target_by_default(self):
        self.write_config()
        with patch("src.pipeline_cli.plan_jobs", return_value=[PaperJob("ConVis", ("evaluate",))]), patch("src.pipeline_cli.execute_jobs"), patch("src.api_probe.probe_apis") as probe:
            self.assertEqual(self.call(["evaluate"]), 0)
        probe.assert_not_called()
        self.assertIn("only if CLAIM_VERIFY_ENABLED=true", self.stdout.getvalue())

    def test_claim_verification_adds_embedding_requirements(self):
        self.write_config("CLAIM_VERIFY_ENABLED=true\n")
        with patch("src.pipeline_cli.execute_jobs") as execute:
            self.assertNotEqual(self.call(["evaluate"]), 0)
        execute.assert_not_called()
        self.assertIn("Missing required setting: EMBED_API_KEY", self.stderr.getvalue())

    def test_invalid_configuration_does_not_expose_key_values(self):
        self.write_config("BASE_URL=https://invalid.example.test/v1/chat/completions\n")
        # Environment has precedence over the file, just as in normal deployment.
        os.environ["BASE_URL"] = "https://invalid.example.test/v1/chat/completions"
        with patch("src.pipeline_cli.execute_jobs") as execute:
            self.assertNotEqual(self.call(["evaluate"]), 0)
        execute.assert_not_called()
        output = self.stdout.getvalue() + self.stderr.getvalue()
        self.assertIn("BASE_URL", output)
        self.assertNotIn("secret-judge-value", output)
        self.assertNotIn("secret-target-value", output)

    def test_environment_credentials_are_supported_and_legacy_dotenv_ignored(self):
        (self.root / ".env").write_text("API_KEY=legacy-secret\n", encoding="utf-8")
        os.environ.update({
            "API_KEY": "environment-judge", "BASE_URL": "https://judge.example.test/v1", "LLM_MODEL": "judge",
            "EVAL_TARGET_API_KEY": "environment-target", "EVAL_TARGET_BASE_URL": "https://target.example.test/v1", "EVAL_TARGET_MODEL": "target",
        })
        with patch("src.pipeline_cli.plan_jobs", return_value=[PaperJob("ConVis", ("evaluate",))]), patch("src.pipeline_cli.execute_jobs"):
            self.assertEqual(self.call(["evaluate"]), 0)
        self.assertEqual(os.environ["API_KEY"], "environment-judge")
        self.assertNotIn("legacy-secret", self.stdout.getvalue() + self.stderr.getvalue())

    def test_custom_config_is_created_at_selected_path(self):
        with patch("src.pipeline_cli.execute_jobs") as execute:
            self.assertNotEqual(self.call(["--config", "private/settings.env", "evaluate"]), 0)
        execute.assert_not_called()
        self.assertTrue((self.root / "private/settings.env").is_file())
        self.assertFalse((self.root / "config.env").exists())

    def test_help_and_dry_run_do_not_initialize_or_call_apis(self):
        with patch.object(startup, "pipeline_main", return_value=0) as command:
            self.assertEqual(self.call(["--help"]), 0)
            command.assert_not_called()
            self.assertEqual(self.call(["--dry-run"]), 0)
            command.assert_called_once_with(["run", "--dry-run", "--paper-id", "ConVis"])
        self.assertFalse((self.root / "config.env").exists())

    def test_full_pipeline_checks_all_roles_and_pdf_also_checks_ocr(self):
        self.write_config(full=True)
        with patch.object(startup, "pipeline_main", return_value=0) as command:
            self.assertEqual(self.call(["run", "--pdf", "paper.pdf"]), 0)
        self.assertEqual(command.call_args_list[0].args[0], ["check", "--stage", "all"])
        self.assertEqual(command.call_args_list[1].args[0], ["check", "--stage", "parse"])
        self.assertEqual(command.call_args_list[2].args[0], ["run", "--pdf", "paper.pdf"])

    def test_measurement_uses_its_own_parser_and_preserves_config_and_options(self):
        tokens, args = startup.startup_arguments(["--config", "private.env", "measure", "--repeats", "2", "--workers", "3"])
        self.assertEqual(args.command, "measure")
        self.assertEqual((args.repeats, args.workers), (2, 3))
        self.assertEqual(tokens, ["--config", "private.env", "--repeats", "2", "--workers", "3"])
        self.assertIsNone(args.paper_id)

    def test_measurement_preflight_then_delegates_without_command_token(self):
        self.write_config(full=True)
        with patch.object(startup, "pipeline_main", return_value=0) as check, patch("run_measurement.main", return_value=0) as measure:
            self.assertEqual(self.call(["measure", "--workers", "2", "--models", "alpha"]), 0)
        check.assert_called_once_with(["check", "--stage", "all", "--models", "alpha"])
        measure.assert_called_once_with(["--workers", "2", "--models", "alpha"])

    def test_measurement_dry_run_does_not_create_config_or_run_preflight(self):
        with patch.object(startup, "pipeline_main") as check, patch("run_measurement.main", return_value=0) as measure:
            self.assertEqual(self.call(["measure", "--dry-run"]), 0)
        check.assert_not_called()
        measure.assert_called_once_with(["--dry-run"])
        self.assertFalse((self.root / "config.env").exists())

    def test_measurement_missing_credentials_stop_before_measurement(self):
        with patch("run_measurement.main") as measure:
            self.assertNotEqual(self.call(["measure"]), 0)
        measure.assert_not_called()
        self.assertIn("VISION_API_KEY", self.stderr.getvalue())

    def test_measurement_help_lists_measurement_arguments_without_files(self):
        with patch("run_measurement.main") as measure:
            self.assertEqual(self.call(["measure", "--help"]), 0)
        measure.assert_not_called()
        self.assertIn("--repeats", self.stdout.getvalue())
        self.assertIn("--output-root", self.stdout.getvalue())
        self.assertFalse((self.root / "config.env").exists())

    def test_existing_dataset_test_parser_defaults_to_all_and_one_round(self):
        tokens, args = startup.startup_arguments(["test", "--workers", "2"])
        self.assertEqual(tokens, ["--workers", "2"])
        self.assertEqual(args.command, "test")
        self.assertEqual(args.repeats, 1)
        self.assertIsNone(args.paper_id)
        self.assertEqual(args.data_root, "data")

    def test_existing_dataset_test_needs_only_evaluation_configuration(self):
        self.write_config()
        with patch.object(startup, "pipeline_main", return_value=0) as check, patch("evaluate_dataset.main", return_value=0) as test:
            self.assertEqual(self.call(["test", "--models", "alpha"]), 0)
        check.assert_called_once_with(["check", "--stage", "evaluate", "--models", "alpha"])
        test.assert_called_once_with(["--models", "alpha"])
        self.assertNotIn("VISION_API_KEY", self.stdout.getvalue())
        self.assertNotIn("EMBED_API_KEY", self.stdout.getvalue())

    def test_existing_dataset_test_help_and_dry_run_make_no_files(self):
        with patch.object(startup, "pipeline_main") as check, patch("evaluate_dataset.main", return_value=0) as test:
            self.assertEqual(self.call(["test", "--help"]), 0)
            test.assert_not_called()
            self.assertIn("--data-root", self.stdout.getvalue())
            self.assertEqual(self.call(["test", "--dry-run"]), 0)
        check.assert_not_called()
        test.assert_called_once_with(["--dry-run"])
        self.assertFalse((self.root / "config.env").exists())


class StartupWrapperTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("powershell"), "Windows PowerShell unavailable")
    def test_powershell_help_requires_neither_uv_nor_config(self):
        environment = dict(os.environ)
        environment["PATH"] = ""
        completed = subprocess.run(
            [shutil.which("powershell"), "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(ROOT / "start.ps1"), "--help"],
            cwd=ROOT.parent, env=environment, capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("MUST provide your own API keys", completed.stdout)
        self.assertIn("measure --workers 2", completed.stdout)

    @unittest.skipUnless(shutil.which("powershell"), "Windows PowerShell unavailable")
    def test_powershell_missing_uv_has_actionable_install_message(self):
        environment = dict(os.environ)
        environment["PATH"] = ""
        completed = subprocess.run(
            [shutil.which("powershell"), "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(ROOT / "start.ps1")],
            cwd=ROOT.parent, env=environment, capture_output=True, timeout=30,
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn(b"uv is required", completed.stdout + completed.stderr)
        self.assertIn(b"https://docs.astral.sh/uv/", completed.stdout + completed.stderr)

    @unittest.skipUnless(shutil.which("powershell"), "Windows PowerShell unavailable")
    def test_powershell_forwards_arguments_and_uses_repository_cwd(self):
        with tempfile.TemporaryDirectory(prefix=".papergraph-test-", dir=ROOT) as folder:
            temp = Path(folder)
            (temp / "uv.ps1").write_text(
                '@{ WorkingDirectory=(Get-Location).Path; Arguments=@($args) } | ConvertTo-Json -Depth 3 | Set-Content -Encoding UTF8 -LiteralPath $env:TEST_RESULT\nexit 0\n',
                encoding="utf-8",
            )
            result = temp / "forwarded.json"
            environment = dict(os.environ, PATH=str(temp), TEST_RESULT=str(result))
            completed = subprocess.run(
                [shutil.which("powershell"), "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(ROOT / "start.ps1"),
                 "evaluate", "--config", "private settings.env", "--models", "alpha", "beta"],
                cwd=ROOT.parent, env=environment, capture_output=True, timeout=30,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            payload = json.loads(result.read_text(encoding="utf-8-sig"))
            self.assertEqual(Path(payload["WorkingDirectory"]), ROOT)
            self.assertEqual(payload["Arguments"], ["run", "--locked", "python", "scripts/start.py", "evaluate", "--config", "private settings.env", "--models", "alpha", "beta"])

    def test_wrappers_quote_arguments_and_anchor_to_repository(self):
        powershell = (ROOT / "start.ps1").read_text(encoding="utf-8")
        bash = (ROOT / "start.sh").read_text(encoding="utf-8")
        self.assertIn("-LiteralPath $PSScriptRoot", powershell)
        self.assertIn("scripts/start.py @args", powershell)
        self.assertNotIn("Invoke-Expression", powershell)
        self.assertIn('scripts/start.py "$@"', bash)
        self.assertIn('${BASH_SOURCE[0]}', bash)


if __name__ == "__main__":
    unittest.main()
