"""Offline CLI-to-provider protocol tests using only a loopback HTTP server."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[1]
PAPER_ID = "tiny"
TARGET_MODEL = "target-mini"
JUDGE_MODEL = "judge-mini"
PAPER_MARKER = "UNIQUE_ORIGINAL_PAPER_SENTINEL_7E31"


class _ProviderHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802 - HTTP protocol method
        length = int(self.headers.get("Content-Length", "0"))
        try:
            payload = json.loads(self.rfile.read(length))
        except (ValueError, UnicodeDecodeError):
            self._respond(400, {"error": "invalid JSON"})
            return

        request = {
            "path": self.path,
            "authorization": self.headers.get("Authorization"),
            "payload": payload,
        }
        with self.server.requests_lock:
            self.server.requests.append(request)

        model = payload.get("model")
        if self.path != "/v1/chat/completions":
            self._respond(404, {"error": "unknown endpoint"})
            return
        if model == TARGET_MODEL and self.server.reject_target:
            self._respond(401, {"error": "test provider rejected target credentials"})
            return
        if model == TARGET_MODEL:
            content = "The paper's method uses the stated graph-grounded bridge."
        elif model == JUDGE_MODEL and payload.get("response_format") == {"type": "json_object"}:
            system = payload.get("messages", [{}])[0].get("content", "")
            if "follow-up question" in system:
                content = json.dumps({"question_text": "How does the paper's bridge work?"})
            else:
                content = json.dumps(
                    {
                        "state": "MAIN_PROGRESS",
                        "coverage": {
                            "target_kc_ids": ["KC1"],
                            "covered_kc_ids": ["KC1"],
                            "missing_kc_ids": [],
                            "coverage_complete": True,
                            "confidence": 0.99,
                        },
                        "hallucination_events": [],
                        "recommended_tasks": [],
                        "next_action": "next_main_question",
                        "confidence": 0.99,
                        "judge_explanation": "The answer covers the sole target KC.",
                    }
                )
        else:
            self._respond(400, {"error": "unexpected model or request shape"})
            return
        self._respond(200, {"choices": [{"message": {"role": "assistant", "content": content}}]})

    def _respond(self, status: int, body: dict) -> None:
        encoded = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, _format: str, *_args: object) -> None:
        pass


class PipelineHttpTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="papergraph-http-", dir=REPOSITORY)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _ProviderHandler)
        self.server.daemon_threads = True
        self.server.requests = []
        self.server.requests_lock = threading.Lock()
        self.server.reject_target = False
        self.server_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.server_thread.start()
        self.addCleanup(self._stop_server)

        self.log_path = REPOSITORY / "outputs" / "logs" / PAPER_ID / f"evaluate_{TARGET_MODEL}.log"
        if self.log_path.exists():
            self.skipTest(f"Refusing to append to an existing test log: {self.log_path}")
        self.preexisting_log_dirs = {
            path: path.exists() for path in (self.log_path.parent, self.log_path.parent.parent, self.log_path.parent.parent.parent)
        }
        self.addCleanup(self._remove_test_log)
        self._write_fixture()

    def _stop_server(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.server_thread.join(timeout=5)

    def _remove_test_log(self) -> None:
        self.log_path.unlink(missing_ok=True)
        for path, existed in self.preexisting_log_dirs.items():
            if not existed:
                try:
                    path.rmdir()  # Removes only an empty test-created directory.
                except OSError:
                    pass

    def _write_fixture(self) -> None:
        paper_dir = self.root / "data" / PAPER_ID
        paper_dir.mkdir(parents=True)
        text_path = paper_dir / "paper_clean_text.md"
        self.paper_text_path = text_path
        text_path.write_text(
            f"# Tiny research paper\n\n## Abstract\n{PAPER_MARKER}\n"
            "The method uses a graph-grounded bridge between claims and evidence.\n",
            encoding="utf-8",
        )
        graph = {
            "paper_id": PAPER_ID,
            "paper_title": "Tiny research paper",
            "storage": {"path_base": "repository"},
            "paper_text_path": text_path.as_posix(),
            "macro_nodes": [
                {
                    "macro_id": "M1",
                    "order": 1,
                    "title": "Method",
                    "role": "method",
                    "summary": "The graph-grounded bridge",
                    "kc_ids": ["KC1"],
                    "active_kc_ids": ["KC1"],
                    "bank_kc_count": 1,
                    "active_kc_count": 1,
                }
            ],
            "macro_edges": [],
            "kc_nodes": [
                {
                    "kc_id": "KC1",
                    "macro_id": "M1",
                    "type": "method",
                    "short_label": "Graph-grounded bridge",
                    "full_claim": "The method uses a graph-grounded bridge between claims and evidence.",
                    "claim": "The method uses a graph-grounded bridge between claims and evidence.",
                    "importance": "critical",
                    "evidence": [{"section": "Abstract", "span_id": "S1", "text": "The method uses a graph-grounded bridge between claims and evidence."}],
                    "must_include": ["graph-grounded bridge"],
                    "acceptable_variants": [],
                    "forbidden_claims": [],
                    "flags": {"active_for_question_generation": True},
                }
            ],
            "active_kc_ids": ["KC1"],
            "reasoning_edges": [],
            "reasoning_paths": [],
            "reasoning_threads": [],
            "diagnostics": {
                "master_graph_kc_source": "bank",
                "graph_signature": {"graph_kc_ids": ["KC1"]},
            },
        }
        questions = {
            "paper_id": PAPER_ID,
            "schema_version": "v2",
            "macro_main_questions": [
                {
                    "question_id": "Q_M1",
                    "question_type": "macro_main_question",
                    "macro_id": "M1",
                    "target_kc_ids": ["KC1"],
                    "question_text": "What bridge does the method use?",
                    "expected_coverage": {"must_cover": ["KC1"], "optional_cover": []},
                    "requires_multimodal_input": False,
                    "asset_references": [],
                }
            ],
            "thread_question_seeds": [],
            "review_question_seeds": [],
            "challenge_questions": [],
            "thread_challenge_questions": [],
            "multi_hop_questions": [],
            "reserved_followup_templates": [],
        }
        for name, payload in (("master_graph.json", graph), ("question_templates.json", questions)):
            (paper_dir / name).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        base_url = f"http://127.0.0.1:{self.server.server_port}/v1"
        settings = {
            "API_KEY": "fake-judge-key",
            "BASE_URL": base_url,
            "LLM_MODEL": JUDGE_MODEL,
            "EVAL_TARGET_API_KEY": "fake-target-key",
            "EVAL_TARGET_BASE_URL": base_url,
            "EVAL_TARGET_MODEL": TARGET_MODEL,
            "PAPERGRAPH_DATA_ROOT": (self.root / "data").as_posix(),
            "RAW_PAPER_ROOT": (self.root / "rawPaper").as_posix(),
            "EVAL_RESULT_ROOT": (self.root / "outputs" / "evaluation").as_posix(),
            "USE_ONLINE_EVAL": "true",
            "ALLOW_OFFLINE_FALLBACK": "false",
            "ALLOW_MOCK_EVAL": "false",
            "CLAIM_VERIFY_ENABLED": "false",
            "PAPERGRAPH_PROGRESS": "false",
            "PAPERGRAPH_LLM_MAX_RETRIES": "0",
            "PAPERGRAPH_LLM_TIMEOUT_S": "5",
        }
        self.config_path = self.root / "config.env"
        self.config_path.write_text("\n".join(f"{key}={value}" for key, value in settings.items()) + "\n", encoding="utf-8")
        self.report_path = self.root / "outputs" / "evaluation" / TARGET_MODEL / PAPER_ID / "evaluation_report.json"
        self.trajectory_path = self.report_path.with_name("dialogue_trajectory.json")

    @staticmethod
    def _isolated_env() -> dict[str, str]:
        allowed = {"PATH", "SYSTEMROOT", "WINDIR", "COMSPEC", "TEMP", "TMP", "USERPROFILE", "LOCALAPPDATA", "APPDATA", "PROGRAMDATA"}
        env = {name: value for name, value in os.environ.items() if name.upper() in allowed}
        env.update(
            {
                "PYTHONIOENCODING": "utf-8",
                "PYTHONDONTWRITEBYTECODE": "1",
                "NO_PROXY": "127.0.0.1,localhost",
                "no_proxy": "127.0.0.1,localhost",
            }
        )
        return env

    def _evaluate(self, *, force: bool = True) -> subprocess.CompletedProcess[str]:
        command = [sys.executable, str(REPOSITORY / "run.py"), "evaluate", "--config", str(self.config_path), "--paper-id", PAPER_ID]
        if force:
            command.append("--force")
        return subprocess.run(
            command,
            cwd=REPOSITORY,
            env=self._isolated_env(),
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
            check=False,
        )

    def test_real_cli_target_judge_and_completed_report(self) -> None:
        process = self._evaluate()
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr + self._log_text())
        report = json.loads(self.report_path.read_text(encoding="utf-8"))
        trajectory = json.loads(self.trajectory_path.read_text(encoding="utf-8"))
        self.assertEqual(report["summary"]["evaluation_status"], "completed")
        self.assertIs(report["summary"]["failed"], False)
        self.assertEqual(report["summary"]["total_turns"], len(trajectory["turns"]))
        self.assertGreaterEqual(len(trajectory["turns"]), 1)
        self.assertEqual(trajectory["turns"][0]["question_type"], "macro_main_question")
        self.assertEqual(trajectory["turns"][0]["answer_mode"], "online_target_text")
        self.assertEqual(trajectory["turns"][0]["judge_result"]["state"], "MAIN_PROGRESS")

        requests = self.server.requests
        target_requests = [item for item in requests if item["payload"].get("model") == TARGET_MODEL]
        judge_requests = [item for item in requests if item["payload"].get("model") == JUDGE_MODEL]
        self.assertTrue(target_requests, requests)
        self.assertTrue(judge_requests, requests)
        self.assertTrue(all(item["path"] == "/v1/chat/completions" for item in requests))
        self.assertTrue(all(item["authorization"] == "Bearer fake-target-key" for item in target_requests))
        self.assertTrue(all(item["authorization"] == "Bearer fake-judge-key" for item in judge_requests))
        self.assertTrue(any(PAPER_MARKER in item["payload"]["messages"][1]["content"] for item in target_requests))
        self.assertTrue(any(item["payload"].get("response_format") == {"type": "json_object"} for item in judge_requests))

    def test_provider_401_fails_without_completed_report(self) -> None:
        self.server.reject_target = True
        process = self._evaluate()
        self.assertNotEqual(process.returncode, 0, process.stdout + process.stderr)
        self.assertIn("HTTP 401", self._log_text())
        self.assertTrue(any(item["payload"].get("model") == TARGET_MODEL for item in self.server.requests))
        self.assertFalse(self.report_path.exists(), "A rejected target request must not create a completed report.")

    def test_completed_cache_reuses_output_and_paper_change_reruns(self) -> None:
        first = self._evaluate(force=False)
        self.assertEqual(first.returncode, 0, first.stdout + first.stderr + self._log_text())
        first_request_count = len(self.server.requests)
        self.assertGreater(first_request_count, 0)
        self.assertEqual(json.loads(self.report_path.read_text(encoding="utf-8"))["summary"]["evaluation_status"], "completed")

        second = self._evaluate(force=False)
        self.assertEqual(second.returncode, 0, second.stdout + second.stderr + self._log_text())
        self.assertEqual(len(self.server.requests), first_request_count, "An unchanged completed result should make no provider requests.")
        self.assertIn("reused", second.stdout + second.stderr)

        revised_marker = "REVISED_PAPER_CONTEXT_SENTINEL_A92D"
        self.paper_text_path.write_text(
            self.paper_text_path.read_text(encoding="utf-8") + f"\n{revised_marker}\n",
            encoding="utf-8",
        )
        third = self._evaluate(force=False)
        self.assertEqual(third.returncode, 0, third.stdout + third.stderr + self._log_text())
        self.assertGreater(len(self.server.requests), first_request_count, "A changed paper must invalidate the completed result.")
        later_target_requests = [
            item for item in self.server.requests[first_request_count:]
            if item["payload"].get("model") == TARGET_MODEL
        ]
        self.assertTrue(any(revised_marker in item["payload"]["messages"][1]["content"] for item in later_target_requests))
        self.assertEqual(json.loads(self.report_path.read_text(encoding="utf-8"))["summary"]["evaluation_status"], "completed")

    def _log_text(self) -> str:
        return self.log_path.read_text(encoding="utf-8") if self.log_path.exists() else ""


if __name__ == "__main__":
    unittest.main()
