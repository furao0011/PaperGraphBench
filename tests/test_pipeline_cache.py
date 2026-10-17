"""Offline input/configuration and output integrity tests for pipeline stamps."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "papergraph_demo"))

from src.artifact_layout import PaperArtifactLayout
from src.pipeline_plans import PaperJob
from src import pipeline_cache as cache


class PipelineCacheTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix=".pipeline-cache-test-", dir=REPOSITORY)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.base = self.root / "papergraph_demo"
        self.base.mkdir()
        self.environment = patch.dict(os.environ, {"PAPERGRAPH_DATA_ROOT": str(self.root / "data"), "RAW_PAPER_ROOT": str(self.root / "rawPaper"), "LLM_MODEL": "judge", "EVAL_TARGET_MODEL": "target", "EVAL_TARGET_BASE_URL": "http://localhost/v1"}, clear=True)
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.project = patch.object(cache, "PROJECT_ROOT", self.root)
        self.project.start()
        self.addCleanup(self.project.stop)
        self.layout_mock = patch.object(cache, "paper_layout", side_effect=lambda paper_id: PaperArtifactLayout(self.base, paper_id))
        self.layout_mock.start()
        self.addCleanup(self.layout_mock.stop)
        self.raw = self.root / "rawPaper" / "paper"
        self.raw.mkdir(parents=True)
        self.write(self.raw / "doc_0.md", "original paper")
        self.image = self.raw / "imgs" / "figure.jpg"
        self.write(self.image, b"image-v1")
        self.layout = PaperArtifactLayout(self.base, "paper")
        self.paper_dir = self.layout.root
        self.write(self.paper_dir / "paper_clean_text.md", "clean text")
        self.write(self.paper_dir / "paper_eval_context.md", "evaluation context")
        self.asset = {"asset_id": "figure", "image_paths": ["rawPaper/paper/imgs/figure.jpg"], "attachments": [{"type": "image", "path": "rawPaper/paper/imgs/figure.jpg"}]}
        self.write_json(self.paper_dir / "multimodal_assets.json", {"paper_id": "paper", "assets": [self.asset]})
        self.write_json(self.paper_dir / "multimodal_asset_explanations.json", {"paper_id": "paper", "explanations": []})
        self.graph = {"paper_id": "paper", "storage": {"path_base": "repository"}, "paper_text_path": "data/paper/paper_clean_text.md", "evaluation_paper_text_path": "data/paper/paper_eval_context.md", "multimodal_assets_path": "data/paper/multimodal_assets.json", "multimodal_asset_explanations_path": "data/paper/multimodal_asset_explanations.json", "diagnostics": {"master_graph_kc_source": "bank"}, "kc_bank_path": "data/paper/cache/unpublished.json", "kc_nodes": [{"kc_id": "KC1"}]}
        self.write_json(self.layout.final("master_graph"), self.graph)
        self.write_json(self.layout.final("question_templates"), {"paper_id": "paper", "macro_main_questions": [{"question_id": "Q1", "asset_references": [self.asset]}]})
        self.job = PaperJob("paper", ("build_graph", "generate_questions", "evaluate"), self.raw)

    def write(self, path: Path, value: str | bytes):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value if isinstance(value, bytes) else value.encode("utf-8"))

    def write_json(self, path: Path, value):
        self.write(path, json.dumps(value))

    def signature(self, stage="evaluate", model="target"):
        return cache.build_stage_signature(self.job, stage, model if stage == "evaluate" else None)

    def test_build_tracks_markdown_and_images(self):
        first = self.signature("build_graph")
        self.write(self.raw / "doc_0.md", "changed paper")
        second = self.signature("build_graph")
        self.assertNotEqual(first, second)
        self.write(self.image, b"image-v2")
        self.assertNotEqual(second, self.signature("build_graph"))

    def test_hidden_and_cache_files_are_not_inputs(self):
        original = self.signature("build_graph")
        self.write(self.raw / "cache" / "checkpoint.json", "new")
        self.write(self.raw / ".hidden" / "doc_0.md", "new")
        self.write(self.raw / ".lock", "new")
        self.assertEqual(original, self.signature("build_graph"))

    def test_single_markdown_input_tracks_referenced_images(self):
        markdown = self.raw / "doc_0.md"
        self.write(markdown, "# Paper\n\n![Figure](imgs/figure.jpg)\n")
        job = PaperJob("paper", ("build_graph",), markdown)
        first = cache.build_stage_signature(job, "build_graph")
        self.write(self.image, b"changed image")
        self.assertNotEqual(first, cache.build_stage_signature(job, "build_graph"))

    def test_pdf_bytes_are_inputs(self):
        pdf = self.root / "paper.pdf"
        self.write(pdf, b"pdf-v1")
        job = PaperJob("paper", ("parse_pdf",), self.raw, pdf)
        first = cache.build_stage_signature(job, "parse_pdf")
        self.write(pdf, b"pdf-v2")
        self.assertNotEqual(first, cache.build_stage_signature(job, "parse_pdf"))

    def test_questions_track_graph_text_assets_explanations_and_images(self):
        for path in (self.layout.final("master_graph"), self.paper_dir / "paper_clean_text.md", self.paper_dir / "multimodal_assets.json", self.paper_dir / "multimodal_asset_explanations.json", self.image):
            with self.subTest(path=path.name):
                before = self.signature("generate_questions")
                original = path.read_bytes()
                if path.suffix == ".json":
                    payload = json.loads(original)
                    payload["signature_test"] = "changed"
                    self.write_json(path, payload)
                else:
                    self.write(path, original + b"changed")
                self.assertNotEqual(before, self.signature("generate_questions"))
                self.write(path, original)

    def test_evaluation_tracks_questions_and_context(self):
        before = self.signature()
        self.write_json(self.layout.final("question_templates"), {"paper_id": "paper", "macro_main_questions": [{"question_id": "Q2"}]})
        after_questions = self.signature()
        self.assertNotEqual(before, after_questions)
        self.write(self.paper_dir / "paper_eval_context.md", "changed context")
        self.assertNotEqual(after_questions, self.signature())

    def test_model_and_algorithm_configuration_change_signatures(self):
        before = self.signature()
        self.assertNotEqual(before, self.signature(model="another-model"))
        os.environ["EVAL_MAX_TURNS"] = "10"
        self.assertNotEqual(before, self.signature())
        before = self.signature()
        os.environ["EVAL_TARGET_BASE_URL_TARGET"] = "http://localhost/another-api"
        self.assertNotEqual(before, self.signature())
        before_build = self.signature("build_graph")
        os.environ["GRAPH_REASONING_EDGE_SOURCE"] = "verified"
        self.assertNotEqual(before_build, self.signature("build_graph"))
        before_build = self.signature("build_graph")
        os.environ["UNIT_DECOMP_MAX_CHARS"] = "2000"
        self.assertNotEqual(before_build, self.signature("build_graph"))

    def test_credentials_and_runtime_controls_do_not_change_signatures(self):
        before = self.signature()
        os.environ.update({"API_KEY": "private-1", "EVAL_TARGET_API_KEY_TARGET": "private-2", "PADDLE_OCR_TOKEN": "private-3", "EMBED_API_KEY": "private-4", "EVAL_RESUME": "true", "PAPERGRAPH_RESTART": "true", "PAPERGRAPH_PROGRESS": "false", "EVAL_BATCH_WORKERS": "8", "QUESTION_CACHE_PATH": "another-cache"})
        self.assertEqual(before, self.signature())
        before_build = self.signature("build_graph")
        os.environ["API_KEY"] = "new-private-key"
        self.assertEqual(before_build, self.signature("build_graph"))

    def test_output_hash_and_signature_both_required(self):
        output = self.root / "results" / "evaluation_report.json"
        self.write(output, "original output")
        signature = self.signature()
        self.assertFalse(cache.stage_is_current(output, signature))
        cache.record_stage_completion(output, signature)
        stamp = output.parent / "cache" / "pipeline" / "evaluation_report.json.json"
        self.assertTrue(stamp.is_file())
        self.assertTrue(cache.stage_is_current(output, signature))
        self.assertFalse(cache.stage_is_current(output, "other-signature"))
        self.assertNotIn("private", stamp.read_text(encoding="utf-8"))
        self.write(output, "corrupt output")
        self.assertFalse(cache.stage_is_current(output, signature))

    def test_corrupt_stamp_is_not_silently_ignored(self):
        output = self.root / "evaluation_report.json"
        stamp = output.parent / "cache" / "pipeline" / "evaluation_report.json.json"
        self.write(stamp, "{invalid JSON")
        with self.assertRaisesRegex(ValueError, "Invalid pipeline completion stamp"):
            cache.stage_is_current(output, self.signature())
        self.write_json(stamp, {"version": 1, "signature": "x"})
        with self.assertRaisesRegex(ValueError, "schema"):
            cache.stage_is_current(output, self.signature())

    def test_missing_required_asset_is_an_error(self):
        self.image.unlink()
        with self.assertRaises(FileNotFoundError):
            self.signature()

    def test_first_attempt_requires_restart_and_records_signature(self):
        output = self.root / "results" / "evaluation_report.json"
        signature = self.signature()
        self.assertTrue(cache.prepare_stage_execution(output, signature))
        attempt = output.parent / "cache" / "pipeline" / "evaluation_report.json.attempt.json"
        self.assertEqual(json.loads(attempt.read_text(encoding="utf-8")), {"signature": signature})
        self.assertFalse(cache.stage_is_current(output, signature))

    def test_same_interrupted_attempt_can_resume(self):
        output = self.root / "results" / "evaluation_report.json"
        signature = self.signature()
        cache.prepare_stage_execution(output, signature)
        self.assertFalse(cache.prepare_stage_execution(output, signature))

    def test_changed_attempt_requires_restart_without_deleting_caches(self):
        output = self.root / "results" / "evaluation_report.json"
        algorithm_cache = output.parent / "cache" / "evaluation_checkpoint.json"
        self.write(algorithm_cache, "prior checkpoint")
        old_signature = self.signature()
        cache.prepare_stage_execution(output, old_signature)
        os.environ["EVAL_MAX_TURNS"] = "5"
        new_signature = self.signature()
        self.assertTrue(cache.prepare_stage_execution(output, new_signature))
        self.assertEqual(algorithm_cache.read_text(encoding="utf-8"), "prior checkpoint")
        self.assertFalse(cache.prepare_stage_execution(output, new_signature))

    def test_corrupt_attempt_is_an_explicit_error(self):
        output = self.root / "results" / "evaluation_report.json"
        attempt = output.parent / "cache" / "pipeline" / "evaluation_report.json.attempt.json"
        self.write(attempt, "{bad JSON")
        with self.assertRaisesRegex(ValueError, "Invalid pipeline attempt stamp"):
            cache.prepare_stage_execution(output, self.signature())
        self.write_json(attempt, {"signature": ""})
        with self.assertRaisesRegex(ValueError, "schema"):
            cache.prepare_stage_execution(output, self.signature())

    def test_attempt_does_not_replace_completed_output_stamp(self):
        output = self.root / "results" / "evaluation_report.json"
        self.write(output, "completed output")
        signature = self.signature()
        cache.record_stage_completion(output, signature)
        self.assertTrue(cache.prepare_stage_execution(output, signature))
        self.assertTrue(cache.stage_is_current(output, signature))


if __name__ == "__main__":
    unittest.main()
