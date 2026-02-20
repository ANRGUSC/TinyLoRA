from __future__ import annotations

import unittest

from tinylora.data import QADatasetSpec, build_prompt, build_sft_target


class DataTests(unittest.TestCase):
    def test_defaults(self) -> None:
        spec = QADatasetSpec()
        self.assertEqual(spec.dataset_name, "gsm8k")
        self.assertEqual(spec.dataset_config, "main")
        self.assertEqual(spec.question_field, "question")
        self.assertEqual(spec.answer_field, "answer")

    def test_prompt_and_target(self) -> None:
        p = build_prompt("2+2?")
        self.assertIn("Question: 2+2?", p)
        self.assertIn("#### <answer>", p)
        t = build_sft_target("  #### 4  ")
        self.assertEqual(t, "#### 4")


if __name__ == "__main__":
    unittest.main()

