"""Offline tests for published-artifact path migration; no credentials are read."""
from __future__ import annotations

import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout


REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "scripts"))
from normalize_artifact_paths import main, normalize_document, normalize_path, planned_changes


class NormalizeArtifactPathTests(unittest.TestCase):
    def test_absolute_windows_and_posix_references_become_repository_relative(self):
        for source in (r"D:\publisher\rawPaper\Paper\imgs\figure.png", "/home/publisher/rawPaper/Paper/imgs/figure.png"):
            self.assertEqual(normalize_path(source, "Paper"), "rawPaper/Paper/imgs/figure.png")
        self.assertEqual(normalize_path(r"C:\repo\data\Paper\paper.md", "Paper"), "data/Paper/paper.md")

    def test_relative_backslashes_are_normalized_and_shared_graph_is_preserved(self):
        self.assertEqual(normalize_path(r"rawPaper\Paper\imgs\a.png", "Paper"), "rawPaper/Paper/imgs/a.png")
        self.assertEqual(normalize_path("data/graphs/macro_spine.json", "Paper"), "data/graphs/macro_spine.json")

    def test_unknown_absolute_cross_paper_and_traversal_are_rejected(self):
        for value in (r"C:\private\image.png", "/tmp/image.png", r"C:relative\file.png", "file:///tmp/picture.png", "/x/rawPaper/Other/img.png", "rawPaper/Other/img.png", "rawPaper/Paper/../Other/img.png", "/x/rawPaper/Paper/./img.png"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                normalize_path(value, "Paper")

    def test_only_path_fields_and_master_storage_change(self):
        semantic = r"The caption quotes C:\private\reference.txt and rawPaper\Paper."
        original = {"paper_id": "Paper", "caption": semantic, "text": semantic,
                    "question": semantic, "reasoning_paths": [{"path_id": "P1", "text": semantic}],
                    "image_paths": [r"D:\repo\rawPaper\Paper\imgs\one.png"],
                    "attachments": [{"path": r"rawPaper\Paper\imgs\one.png", "caption": semantic}],
                    "diagnostics": {"kc_count": 10}}
        before = copy.deepcopy(original)
        normalized, changes = normalize_document(original, "Paper", master_graph=True)
        self.assertEqual(original, before)
        self.assertEqual(len(changes), 3)
        self.assertEqual(normalized["caption"], semantic)
        self.assertEqual(normalized["question"], semantic)
        self.assertEqual(normalized["reasoning_paths"], original["reasoning_paths"])
        self.assertEqual(normalized["diagnostics"], original["diagnostics"])
        self.assertEqual(normalized["attachments"][0]["path"], "rawPaper/Paper/imgs/one.png")
        self.assertEqual(normalized["storage"], {"path_base": "repository"})
        self.assertEqual(normalize_document(normalized, "Paper", master_graph=True), (normalized, []))

    def test_document_identity_and_storage_errors_are_not_hidden(self):
        for document in ({"paper_id": "Other"}, {"storage": {"path_base": "unknown"}}, {"storage": []}):
            with self.assertRaises(ValueError):
                normalize_document(document, "Paper", master_graph=True)

    def test_cli_check_is_read_only_then_write_is_idempotent(self):
        with tempfile.TemporaryDirectory(dir=REPOSITORY) as temporary:
            root = Path(temporary)
            paper = root / "Paper"
            paper.mkdir()
            target = paper / "master_graph.json"
            target.write_text(json.dumps({"paper_id": "Paper", "raw_paper_path": r"rawPaper\Paper"}), encoding="utf-8")
            original = target.read_bytes()
            arguments = ["--data-root", str(root), "--paper-id", "Paper"]
            with redirect_stdout(io.StringIO()):
                self.assertEqual(main(arguments), 1)
                self.assertEqual(target.read_bytes(), original)
                self.assertEqual(main(arguments + ["--write"]), 0)
                self.assertEqual(main(arguments), 0)
            self.assertEqual(json.loads(target.read_text(encoding="utf-8"))["raw_paper_path"], "rawPaper/Paper")

    def test_invalid_later_artifact_prevents_any_writes_and_cache_is_not_read(self):
        with tempfile.TemporaryDirectory(dir=REPOSITORY) as temporary:
            root = Path(temporary)
            paper = root / "Paper"
            (paper / "cache").mkdir(parents=True)
            (paper / "cache" / "private.json").write_text("not JSON", encoding="utf-8")
            first = paper / "a.json"
            first.write_text('{"path": "rawPaper\\\\Paper"}', encoding="utf-8")
            self.assertEqual(len(planned_changes(root)), 1)
            (paper / "z.json").write_text('{"path": "C:/unknown/figure.png"}', encoding="utf-8")
            before = first.read_bytes()
            with redirect_stderr(io.StringIO()):
                self.assertEqual(main(["--data-root", str(root), "--write"]), 2)
            self.assertEqual(first.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
