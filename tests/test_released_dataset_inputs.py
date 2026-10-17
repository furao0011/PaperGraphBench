"""Check every released package without APIs, private configuration or caches."""

from __future__ import annotations

import hashlib
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "papergraph_demo"))
from src.evaluation_inputs import load_kc_bank
from src.multimodal_question_assets import load_multimodal_asset_index, question_image_paths
from src.paper_context import load_full_paper_text
from src.pipeline_artifacts import validate_graph_inputs, validate_question_inputs


class ReleasedDatasetInputTests(unittest.TestCase):
    def test_all_twelve_released_packages_are_self_contained_and_read_only(self):
        packages = sorted((ROOT / "data").glob("*/master_graph.json"))
        self.assertEqual(len(packages), 12, "Run this release check on the full public data checkout.")
        roots = [ROOT / "data", ROOT / "rawPaper"]
        extensions = {".json", ".md", ".mmd", ".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff"}

        def tree():
            return {path.relative_to(ROOT).as_posix() for root in roots for path in root.rglob("*")}

        def fingerprints():
            return {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                    for root in roots for path in root.rglob("*") if path.is_file()
                    and "cache" not in path.relative_to(root).parts
                    and not any(part.startswith(".") for part in path.relative_to(root).parts)
                    and path.suffix.lower() in extensions}

        original_open = Path.open

        def public_input_only(path, mode="r", *args, **kwargs):
            resolved = path.resolve()
            for root in roots:
                if resolved.is_relative_to(root):
                    if "cache" in resolved.relative_to(root).parts:
                        raise AssertionError(f"Released evaluation must not read an unpublished cache: {resolved}")
                    if any(flag in mode for flag in ("w", "a", "x", "+")):
                        raise AssertionError(f"Preflight must not write source inputs: {resolved}")
            return original_open(path, mode, *args, **kwargs)

        before_tree, before_bytes = tree(), fingerprints()
        environment = {"PAPERGRAPH_DATA_ROOT": str(ROOT / "data"), "RAW_PAPER_ROOT": str(ROOT / "rawPaper")}
        image_count, kc_count = 0, 0
        with patch.dict(os.environ, environment, clear=True), patch.object(Path, "open", new=public_input_only), \
             patch("urllib.request.urlopen", side_effect=AssertionError("Preflight must not call an API")), \
             patch("src.config.load_environment", side_effect=AssertionError("Preflight must not read private configuration")), \
             patch("src.config.load_settings", side_effect=AssertionError("Preflight must not read private configuration")):
            for graph_path in packages:
                paper_id = graph_path.parent.name
                with self.subTest(paper_id=paper_id):
                    graph = validate_graph_inputs(paper_id)
                    questions = validate_question_inputs(paper_id, graph)
                    self.assertEqual(graph["paper_id"], paper_id)
                    self.assertEqual(questions["paper_id"], paper_id)
                    bank = load_kc_bank({**graph, "kc_bank_path": f"data/{paper_id}/cache/not-published.json"}, ROOT / "papergraph_demo")
                    self.assertEqual(bank["source"], "master_graph.kc_nodes")
                    self.assertEqual(len(bank["kc_nodes"]), len(graph["kc_nodes"]))
                    self.assertTrue(all((node.get("full_claim") or node.get("claim") or "").strip()
                                        and node.get("evidence") for node in bank["kc_nodes"]))
                    kc_count += len(bank["kc_nodes"])
                    self.assertTrue(load_full_paper_text(graph, ROOT / "papergraph_demo").strip())
                    assets = load_multimodal_asset_index(graph, ROOT / "papergraph_demo")
                    images = [path for asset in assets.values() for path in asset.get("image_paths", [])]
                    images += [path for items in questions.values() if isinstance(items, list)
                               for question in items if isinstance(question, dict)
                               for path in question_image_paths(question)]
                    self.assertTrue(images, "Each released multimodal paper must expose readable images.")
                    for image in images:
                        path = Path(image)
                        self.assertTrue(path.is_absolute() and path.is_file(), str(path))
                        self.assertTrue(path.resolve().is_relative_to(ROOT / "rawPaper"), str(path))
                    image_count += len(images)
        self.assertGreater(kc_count, 0)
        self.assertGreater(image_count, 0)
        self.assertEqual(tree(), before_tree, "Preflight created or removed source paths.")
        self.assertEqual(fingerprints(), before_bytes, "Preflight changed released source bytes.")


if __name__ == "__main__":
    unittest.main()
