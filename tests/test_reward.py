from __future__ import annotations

import unittest

from tinylora.reward import exact_match_reward, extract_final_answer


class RewardTests(unittest.TestCase):
    def test_extract_boxed(self) -> None:
        text = "Reasoning...\n#### 1,234\n"
        self.assertEqual(extract_final_answer(text), "1234")

    def test_extract_latex_boxed(self) -> None:
        text = "Therefore the final answer is \\boxed{1,234}."
        self.assertEqual(extract_final_answer(text), "1234")

    def test_extract_latex_boxed_nested(self) -> None:
        text = "Result: \\boxed{\\frac{3}{4}}"
        self.assertEqual(extract_final_answer(text), "\\frac{3}{4}")

    def test_extract_prefers_gsm8k_marker(self) -> None:
        text = "intermediate \\boxed{99}\n#### 42"
        self.assertEqual(extract_final_answer(text), "42")

    def test_extract_fallback_number(self) -> None:
        text = "The final value is 42."
        self.assertEqual(extract_final_answer(text), "42")

    def test_exact_match(self) -> None:
        pred = "Let's solve. #### 15"
        ref = "work ... #### 15"
        self.assertEqual(exact_match_reward(pred, ref), 1.0)
        self.assertEqual(exact_match_reward("#### 14", ref), 0.0)


if __name__ == "__main__":
    unittest.main()
