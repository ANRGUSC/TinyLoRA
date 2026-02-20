from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tinylora.data import QADatasetSpec, build_prompt, build_sft_target, load_qa_split


@dataclass(frozen=True)
class GSM8KExample:
    question: str
    answer: str
    final_answer: str


def load_gsm8k_split(split: str = "train", limit: int = 0) -> list[GSM8KExample]:
    spec = QADatasetSpec(
        dataset_name="gsm8k",
        dataset_config="main",
        question_field="question",
        answer_field="answer",
        train_split="train",
        eval_split="test",
    )
    base = load_qa_split(spec=spec, split=split, limit=limit)
    return [GSM8KExample(question=x.question, answer=x.answer, final_answer=x.final_answer) for x in base]


def to_json_ready(examples: list[GSM8KExample]) -> list[dict[str, Any]]:
    return [{"question": e.question, "answer": e.answer, "final_answer": e.final_answer} for e in examples]
