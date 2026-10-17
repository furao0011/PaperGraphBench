"""Online repair generation handles optional hints without inventing evidence."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import Mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "papergraph_demo"))
from src.dialogue_engine import generate_followup_question


class FollowupGenerationTests(unittest.TestCase):
    def generate(self, kcs):
        client = Mock()
        client.is_ready.return_value = True
        client.chat_json.return_value = {"question_text": "Can you recheck this claim against the paper?"}
        question = generate_followup_question(
            "hallucination_followup", {"question_id": "Q1", "question_text": "Original question"}, kcs, client,
            repair_context={"active_hallucinations": [{"claim": "Unsupported original claim"}]},
        )
        return question, client.chat_json.call_args.kwargs["user_prompt"]

    def test_empty_forbidden_claims_still_generates_online_repair(self):
        question, prompt = self.generate([{"kc_id": "K1", "forbidden_claims": []}])
        self.assertEqual(question["question_type"], "hallucination_followup")
        self.assertIn("Followup hints: []", prompt)
        self.assertIn("Unsupported original claim", prompt)

    def test_optional_forbidden_claims_do_not_require_placeholder_data(self):
        question, prompt = self.generate([{"kc_id": "K1"}])
        self.assertEqual(question["target_kc_ids"], ["K1"])
        self.assertIn("Followup hints: []", prompt)

    def test_hints_from_all_target_kcs_are_preserved_and_deduplicated(self):
        _, prompt = self.generate([
            {"kc_id": "K1", "forbidden_claims": []},
            {"kc_id": "K2", "forbidden_claims": [{"followup_hint": "Check evidence A"}, {"followup_hint": "Check evidence B"}]},
            {"kc_id": "K3", "forbidden_claims": [{"followup_hint": "Check evidence A"}, {}]},
        ])
        self.assertIn('Followup hints: ["Check evidence A", "Check evidence B"]', prompt)


if __name__ == "__main__":
    unittest.main()
