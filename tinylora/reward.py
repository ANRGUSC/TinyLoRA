from __future__ import annotations

import re


GSM8K_RE = re.compile(r"####\s*([^\n\r]+)")
NUMBER_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")


def normalize_answer(text: str) -> str:
    value = text.strip()
    if value.startswith("$") and value.endswith("$") and len(value) > 1:
        value = value[1:-1].strip()
    value = value.replace(",", "")
    value = value.rstrip(".")
    return value


def _extract_latex_boxed(text: str) -> str:
    marker = r"\boxed"
    pos = 0
    last = ""
    while True:
        start = text.find(marker, pos)
        if start < 0:
            break
        i = start + len(marker)
        while i < len(text) and text[i].isspace():
            i += 1
        if i >= len(text):
            break
        if text[i] != "{":
            j = i
            while j < len(text) and not text[j].isspace():
                j += 1
            if j > i:
                last = text[i:j]
            pos = j
            continue
        depth = 0
        j = i
        while j < len(text):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[i + 1 : j].strip()
                    if candidate:
                        last = candidate
                    pos = j + 1
                    break
            j += 1
        else:
            break
    return last


def extract_final_answer(text: str) -> str:
    if not text:
        return ""
    m = GSM8K_RE.search(text)
    if m:
        return normalize_answer(m.group(1))
    boxed = _extract_latex_boxed(text)
    if boxed:
        return normalize_answer(boxed)

    # Fallback: use last detected number.
    matches = NUMBER_RE.findall(text)
    if matches:
        return normalize_answer(matches[-1])
    return normalize_answer(text)


def exact_match_reward(prediction_text: str, reference_text: str) -> float:
    pred = extract_final_answer(prediction_text)
    ref = extract_final_answer(reference_text)
    return 1.0 if pred == ref and pred != "" else 0.0
