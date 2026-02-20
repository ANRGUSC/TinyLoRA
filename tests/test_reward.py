from __future__ import annotations

import unittest

from tinylora.reward import exact_match_reward, extract_final_answer


class RewardTests(unittest.TestCase):
    def test_extract_boxed(self) -> None:
        text = "Reasoning...\n#### 1,234\n"
        self.assertEqual(extract_final_answer(text), "1234")

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
