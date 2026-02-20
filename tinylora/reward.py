from __future__ import annotations

import re


BOXED_RE = re.compile(r"####\s*([^\n\r]+)")
NUMBER_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")


def normalize_answer(text: str) -> str:
    value = text.strip()
    value = value.replace(",", "")
    value = value.rstrip(".")
    return value


def extract_final_answer(text: str) -> str:
    if not text:
        return ""
    m = BOXED_RE.search(text)
    if m:
        return normalize_answer(m.group(1))

    # Fallback: use last detected number.
    matches = NUMBER_RE.findall(text)
    if matches:
        return normalize_answer(matches[-1])
    return normalize_answer(text)


def exact_match_reward(prediction_text: str, reference_text: str) -> float:
    pred = extract_final_answer(prediction_text)
    ref = extract_final_answer(reference_text)
    return 1.0 if pred == ref and pred != "" else 0.0
