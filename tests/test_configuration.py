"""Local configuration and opt-in API-probe tests; never contact providers."""

from __future__ import annotations

import io
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch


REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "papergraph_demo"))

from src.api_probe import probe_apis  # noqa: E402
from src.config import load_environment  # noqa: E402
from src.pipeline_cli import main as pipeline_main  # noqa: E402
from src.preflight import validate_config  # noqa: E402


CHAT = {
    "API_KEY": "unit-chat-key",
    "BASE_URL": "https://chat.example.test/v1",
    "LLM_MODEL": "unit-judge",
}
EMBEDDING = {
    "EMBED_API_KEY": "unit-embed-key",
    "EMBED_BASE_URL": "https://embed.example.test/v1",
    "EMBED_MODEL": "unit-embedding",
}
VISION = {
    "VISION_API_KEY": "unit-vision-key",
    "VISION_BASE_URL": "https://vision.example.test/v1",
    "VISION_MODEL": "unit-vision",
}
TARGET = {
    "EVAL_TARGET_API_KEY": "unit-target-key",
    "EVAL_TARGET_BASE_URL": "https://target.example.test/v1",
    "EVAL_TARGET_MODEL": "unit-target",
}


class ConfigurationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix=".papergraph-test-", dir=REPOSITORY)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_environment_overrides_config_and_roots_ignore_cwd(self) -> None:
        (self.root / "config.env").write_text(
            "API_KEY=file-key\nPAPERGRAPH_DATA_ROOT=stored-data\n",
            encoding="utf-8",
        )
        elsewhere = self.root / "elsewhere"
        elsewhere.mkdir()
        previous = Path.cwd()
        try:
            with patch.dict(os.environ, {"API_KEY": "process-key"}, clear=True):
                os.chdir(elsewhere)
                selected = load_environment(self.root)
                self.assertEqual(os.environ["API_KEY"], "process-key")
                self.assertEqual(
                    Path(os.environ["PAPERGRAPH_DATA_ROOT"]), self.root / "stored-data"
                )
                self.assertEqual(Path(os.environ["RAW_PAPER_ROOT"]), self.root / "rawPaper")
                self.assertEqual(
                    Path(os.environ["EVAL_RESULT_ROOT"]), self.root / "outputs/evaluation"
                )
        finally:
            os.chdir(previous)
        self.assertEqual(selected, self.root / "config.env")

    def test_default_does_not_read_legacy_dotenv(self) -> None:
        (self.root / ".env").write_text("API_KEY=legacy-secret\n", encoding="utf-8")
        with patch.dict(os.environ, {}, clear=True):
            selected = load_environment(self.root)
            self.assertNotIn("API_KEY", os.environ)
            self.assertEqual(os.environ["ALLOW_MOCK_EVAL"], "false")
        self.assertEqual(selected, self.root / "config.env")

    def test_explicit_missing_config_is_an_error(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(FileNotFoundError, "PAPERGRAPH_CONFIG file does not exist"):
                load_environment(self.root, "missing.env")

    def test_preflight_aggregates_stage_requirements(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            errors = validate_config(
                ["parse_pdf", "build_graph", "generate_questions", "evaluate"], self.root
            )
        for name in (
            "PADDLE_OCR_API_URL", "PADDLE_OCR_TOKEN",
            "API_KEY", "BASE_URL", "LLM_MODEL",
            "EMBED_API_KEY", "EMBED_BASE_URL", "EMBED_MODEL",
            "VISION_API_KEY", "VISION_BASE_URL", "VISION_MODEL",
            "EVAL_TARGET_API_KEY", "EVAL_TARGET_BASE_URL", "EVAL_TARGET_MODEL",
        ):
            self.assertIn(f"Missing required setting: {name}.", errors)

    def test_build_requires_embeddings_but_not_target_or_ocr(self) -> None:
        with patch.dict(os.environ, CHAT | VISION, clear=True):
            errors = validate_config(["build_graph"], self.root)
        for name in EMBEDDING:
            self.assertIn(f"Missing required setting: {name}.", errors)
        self.assertFalse(any("EVAL_TARGET_" in error for error in errors))
        self.assertFalse(any("PADDLE_OCR_" in error for error in errors))

    def test_evaluate_without_claim_verification_needs_no_embedding(self) -> None:
        with patch.dict(os.environ, CHAT | TARGET | {"CLAIM_VERIFY_ENABLED": "false"}, clear=True):
            errors = validate_config(["evaluate"], self.root)
        self.assertEqual(errors, [])

    def test_evaluate_with_claim_verification_requires_embedding(self) -> None:
        with patch.dict(os.environ, CHAT | TARGET | {"CLAIM_VERIFY_ENABLED": "true"}, clear=True):
            errors = validate_config(["evaluate"], self.root)
        for name in EMBEDDING:
            self.assertIn(f"Missing required setting: {name}.", errors)

    def test_default_multimodal_build_rejects_missing_vision(self) -> None:
        with patch.dict(os.environ, CHAT | EMBEDDING, clear=True):
            errors = validate_config(["build_graph"], self.root)
        for name in VISION:
            self.assertIn(f"Missing required setting: {name}.", errors)

    def test_formal_pipeline_rejects_mock_and_offline_flags(self) -> None:
        unsafe = {"ALLOW_MOCK_EVAL": "true", "ALLOW_OFFLINE_FALLBACK": "true"}
        with patch.dict(os.environ, CHAT | TARGET | unsafe, clear=True):
            errors = validate_config(["evaluate"], self.root)
        self.assertTrue(any("ALLOW_MOCK_EVAL must be false" in error for error in errors))
        self.assertTrue(any("ALLOW_OFFLINE_FALLBACK must be false" in error for error in errors))

    def test_bad_config_reports_never_echo_secret_values(self) -> None:
        secret = "unique-private-key-987654"
        (self.root / "config.env").write_text(
            f"API_KEY={secret}\nBASE_URL=https://chat.example.test/v1?token={secret}\n",
            encoding="utf-8",
        )
        with patch.dict(os.environ, {}, clear=True):
            errors = validate_config(["evaluate"], self.root)
        self.assertTrue(any("BASE_URL must be an HTTP(S) URL" in error for error in errors))
        self.assertNotIn(secret, "\n".join(errors))

    def test_plain_preflight_makes_no_network_or_probe_calls(self) -> None:
        with patch.dict(os.environ, CHAT | TARGET, clear=True):
            with patch("socket.socket.connect", side_effect=AssertionError("network attempted")):
                with patch("src.api_probe.OpenAICompatClient") as client:
                    errors = validate_config(["evaluate"], self.root)
                    client.assert_not_called()
        self.assertEqual(errors, [])

    def test_check_command_does_not_probe_without_explicit_opt_in(self) -> None:
        output = io.StringIO()
        with patch.dict(os.environ, CHAT | TARGET, clear=True):
            with patch("src.pipeline_cli.PROJECT_ROOT", self.root):
                with patch(
                    "src.pipeline_cli.validate_config",
                    side_effect=lambda stages: validate_config(stages, self.root),
                ):
                    with patch("src.api_probe.probe_apis") as probe:
                        with patch("socket.socket.connect", side_effect=AssertionError("network attempted")):
                            with redirect_stdout(output):
                                status = pipeline_main(["check", "--stage", "evaluate"])
                        probe.assert_not_called()
        self.assertEqual(status, 0)
        self.assertIn("no provider requests made", output.getvalue())

    def test_probe_refuses_invalid_configuration_without_client(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with patch("src.api_probe.validate_config", return_value=["Missing required setting: API_KEY."]):
                with patch("src.api_probe.OpenAICompatClient") as client:
                    messages = probe_apis(["evaluate"])
                    client.assert_not_called()
        self.assertEqual(messages, ["Configuration failed: Missing required setting: API_KEY."])

    def test_explicit_probe_uses_mock_protocol_and_redacts_secrets(self) -> None:
        secret = "probe-secret-24680"

        class FakeClient:
            def __init__(self, config):
                self.cfg = config

            def chat_json(self, *_args, **_kwargs):
                raise RuntimeError(f"provider sent Bearer {secret}")

            def chat_text(self, *_args, **_kwargs):
                return "ready"

        with patch.dict(os.environ, CHAT | TARGET | {"API_KEY": secret}, clear=True):
            with patch("src.api_probe.validate_config", return_value=[]):
                with patch("src.api_probe.OpenAICompatClient", FakeClient):
                    messages = probe_apis(["evaluate"])

        self.assertEqual(len(messages), 2)
        self.assertNotIn(secret, "\n".join(messages))
        self.assertIn("[REDACTED]", messages[0])
        self.assertEqual(messages[1], "Target/chat text: OK")


if __name__ == "__main__":
    unittest.main()
