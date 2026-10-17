"""Regression tests for published paper packages and portable artifact paths.

These tests use only local files and standard-library unittest; no model API is
called. Public-package tests skip explicitly when a sparse checkout excludes
the tracked ConVis example. Path-contract tests always run with temporary data.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path, PureWindowsPath
from unittest.mock import patch


REPOSITORY = Path(__file__).resolve().parents[1]
DEMO = REPOSITORY / "papergraph_demo"
PUBLIC_GRAPH = REPOSITORY / "data" / "ConVis" / "master_graph.json"
PUBLIC_ASSETS = REPOSITORY / "data" / "ConVis" / "multimodal_assets.json"
PUBLIC_QUESTIONS = REPOSITORY / "data" / "ConVis" / "question_templates.json"
PUBLIC_TEXT = REPOSITORY / "data" / "ConVis" / "paper_clean_text.md"

sys.path.insert(0, str(DEMO))

from src.artifact_layout import PaperArtifactLayout  # noqa: E402
from src.artifact_paths import resolve_asset_path, resolve_graph_path  # noqa: E402
from src.evaluation_inputs import load_kc_bank, repair_questions_for_graph  # noqa: E402
from src.multimodal_question_assets import (  # noqa: E402
    load_multimodal_asset_index,
    question_image_paths,
)
from src.paper_context import load_full_paper_text  # noqa: E402


@contextmanager
def changed_directory(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


class PublishedConVisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        required = (PUBLIC_GRAPH, PUBLIC_ASSETS, PUBLIC_QUESTIONS, PUBLIC_TEXT)
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise unittest.SkipTest(
                "The tracked ConVis public package is absent from this checkout: "
                + ", ".join(missing)
            )
        cls.graph = json.loads(PUBLIC_GRAPH.read_text(encoding="utf-8"))
        cls.assets = json.loads(PUBLIC_ASSETS.read_text(encoding="utf-8"))
        cls.questions = json.loads(PUBLIC_QUESTIONS.read_text(encoding="utf-8"))

    def test_embedded_full_bank_needs_no_unreleased_cache(self) -> None:
        self.assertEqual(self.graph["paper_id"], "ConVis")
        self.assertEqual(len(self.graph["kc_nodes"]), 145)
        self.assertEqual(self.graph["diagnostics"]["master_graph_kc_source"], "bank")
        graph = dict(self.graph)
        graph["kc_bank_path"] = "data/ConVis/cache/graph_build/not-in-public-release.json"

        bank = load_kc_bank(graph, DEMO)

        self.assertEqual(bank["paper_id"], "ConVis")
        self.assertEqual(bank["kc_nodes"], self.graph["kc_nodes"])
        self.assertEqual(bank["source"], "master_graph.kc_nodes")

    def test_incomplete_embedded_bank_is_rejected(self) -> None:
        graph = dict(self.graph)
        graph["kc_nodes"] = self.graph["kc_nodes"][:-1]
        with self.assertRaisesRegex(ValueError, "Embedded KC Bank"):
            load_kc_bank(graph, DEMO)

    def test_published_relative_images_resolve_in_this_repository(self) -> None:
        figure = next(
            asset for asset in self.assets["assets"]
            if asset.get("asset_type") == "figure" and asset.get("image_paths")
        )
        published_path = figure["image_paths"][0]
        self.assertTrue(published_path.startswith("rawPaper/ConVis/"))
        self.assertNotIn("\\", published_path)
        self.assertFalse(PureWindowsPath(published_path).is_absolute())
        expected = (
            REPOSITORY / published_path
        ).resolve()
        self.assertTrue(expected.is_file(), "The public ConVis OCR image must be checked out")

        with patch.dict(os.environ, {
            "PAPERGRAPH_DATA_ROOT": str(REPOSITORY / "data"),
            "RAW_PAPER_ROOT": str(REPOSITORY / "rawPaper"),
        }):
            index = load_multimodal_asset_index(self.graph, DEMO)

        self.assertEqual(Path(index[figure["asset_id"]]["image_paths"][0]), expected)
        self.assertEqual(
            Path(index[figure["asset_id"]]["attachments"][0]["path"]), expected
        )
        self.assertEqual(figure["image_paths"][0], published_path, "Loading must not mutate the release")

    def test_question_attachments_are_local_existing_images(self) -> None:
        with patch.dict(os.environ, {
            "PAPERGRAPH_DATA_ROOT": str(REPOSITORY / "data"),
            "RAW_PAPER_ROOT": str(REPOSITORY / "rawPaper"),
        }):
            questions = repair_questions_for_graph(self.graph, self.questions, DEMO)

        paths = [
            path
            for question in questions.get("challenge_questions", [])
            for path in question_image_paths(question)
        ]
        self.assertTrue(paths, "The published ConVis challenges must exercise image paths")
        raw_root = (REPOSITORY / "rawPaper" / "ConVis").resolve()
        for raw_path in paths:
            path = Path(raw_path)
            self.assertTrue(path.is_absolute())
            self.assertTrue(path.is_file(), f"A challenge references a missing image: {path}")
            self.assertTrue(path.is_relative_to(raw_root))

    def test_missing_published_image_is_not_silently_ignored(self) -> None:
        with tempfile.TemporaryDirectory(prefix=".papergraph-test-", dir=REPOSITORY) as empty_raw_root:
            with patch.dict(os.environ, {
                "PAPERGRAPH_DATA_ROOT": str(REPOSITORY / "data"),
                "RAW_PAPER_ROOT": empty_raw_root,
            }):
                with self.assertRaisesRegex(FileNotFoundError, "missing image"):
                    load_multimodal_asset_index(self.graph, DEMO)

    def test_full_text_uses_public_data_not_a_local_paper_draft(self) -> None:
        with tempfile.TemporaryDirectory(prefix=".papergraph-test-", dir=REPOSITORY) as elsewhere:
            with patch.dict(os.environ, {
                "PAPERGRAPH_DATA_ROOT": str(REPOSITORY / "data"),
                "EVAL_PAPER_CHAR_LIMIT": "0",
            }):
                with changed_directory(Path(elsewhere)):
                    text = load_full_paper_text(
                        self.graph, DEMO, prefer_evaluation_context=True
                    )
        self.assertEqual(text, PUBLIC_TEXT.read_text(encoding="utf-8"))


class ArtifactPathContractTests(unittest.TestCase):
    def test_legacy_absolute_images_have_explicit_paper_suffix_migration(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPOSITORY) as temporary:
            base_dir = Path(temporary) / "repo" / "papergraph_demo"
            base_dir.mkdir(parents=True)
            raw_root = Path(temporary) / "relocated-ocr"
            with patch.dict(os.environ, {"RAW_PAPER_ROOT": str(raw_root)}):
                resolved = resolve_asset_path(
                    r"D:\publisher\rawPaper\ConVis\imgs\figure.jpg",
                    base_dir,
                    {"paper_id": "ConVis"},
                )
            self.assertEqual(resolved, raw_root / "ConVis" / "imgs" / "figure.jpg")

    def test_custom_data_root_is_repository_relative_and_cwd_independent(self) -> None:
        with tempfile.TemporaryDirectory(prefix=".papergraph-test-", dir=REPOSITORY) as temporary:
            root = Path(temporary)
            fake_repository = root / "repo"
            base_dir = fake_repository / "papergraph_demo"
            base_dir.mkdir(parents=True)
            elsewhere = root / "elsewhere"
            elsewhere.mkdir()
            text_path = fake_repository / "custom-data" / "ConVis" / "paper_clean_text.md"
            text_path.parent.mkdir(parents=True)
            text_path.write_text("A compact paper fixture.", encoding="utf-8")
            graph = {"paper_id": "ConVis", "paper_text_path": "data/ConVis/paper_clean_text.md"}

            with patch.dict(os.environ, {
                "PAPERGRAPH_DATA_ROOT": "custom-data",
                "EVAL_PAPER_CHAR_LIMIT": "0",
            }):
                with changed_directory(elsewhere):
                    layout = PaperArtifactLayout(base_dir, "ConVis")
                    resolved = resolve_graph_path(graph["paper_text_path"], base_dir, graph)
                    text = load_full_paper_text(graph, base_dir)

            self.assertEqual(layout.final("master_graph"), text_path.parent / "master_graph.json")
            self.assertEqual(layout.storage_metadata, {"path_base": "repository"})
            self.assertEqual(resolved, text_path)
            self.assertEqual(text, "A compact paper fixture.")

    def test_legacy_demo_path_and_declared_storage_base_are_distinct(self) -> None:
        with tempfile.TemporaryDirectory(prefix=".papergraph-test-", dir=REPOSITORY) as temporary:
            root = Path(temporary) / "repo"
            base_dir = root / "papergraph_demo"
            base_dir.mkdir(parents=True)
            with patch.dict(os.environ, {"PAPERGRAPH_DATA_ROOT": "custom-data"}):
                legacy = resolve_graph_path(
                    "papergraph_demo/data/ConVis/paper_clean_text.md", base_dir, {}
                )
                declared = resolve_graph_path(
                    "data/ConVis/paper_clean_text.md",
                    base_dir,
                    {"storage": {"path_base": "repository"}},
                )
            self.assertEqual(legacy, root / "papergraph_demo/data/ConVis/paper_clean_text.md")
            self.assertEqual(declared, root / "data/ConVis/paper_clean_text.md")

    def test_invalid_storage_metadata_and_wrong_paper_assets_fail(self) -> None:
        with tempfile.TemporaryDirectory(prefix=".papergraph-test-", dir=REPOSITORY) as temporary:
            base_dir = Path(temporary) / "repo" / "papergraph_demo"
            base_dir.mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, "Unsupported graph.storage.path_base"):
                resolve_graph_path(
                    "data/ConVis/master_graph.json",
                    base_dir,
                    {"storage": {"path_base": "unknown"}},
                )
            with self.assertRaisesRegex(ValueError, "differs from graph paper id"):
                resolve_asset_path(
                    r"D:\publisher\rawPaper\OtherPaper\imgs\figure.jpg",
                    base_dir,
                    {"paper_id": "ConVis"},
                )


if __name__ == "__main__":
    unittest.main()
