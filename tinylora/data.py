from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tinylora.reward import extract_final_answer


@dataclass(frozen=True)
class QAExample:
    question: str
    answer: str
    final_answer: str


@dataclass(frozen=True)
class QADatasetSpec:
    dataset_name: str = "gsm8k"
    dataset_config: str = "main"
    question_field: str = "question"
    answer_field: str = "answer"
    train_split: str = "train"
    eval_split: str = "test"
    answer_prefix: str = "####"


def build_prompt(question: str) -> str:
    return (
        "You are a careful math reasoning assistant.\n"
        "Solve the problem step by step.\n"
        "At the end, provide your final answer in the format: #### <answer>\n\n"
        f"Question: {question}\n"
        "Answer:"
    )


def build_sft_target(answer: str) -> str:
    return answer.strip()


def load_qa_split(spec: QADatasetSpec, split: str, limit: int = 0) -> list[QAExample]:
    try:
        from datasets import load_dataset
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"datasets package is required: {exc}") from exc

    cfg = spec.dataset_config if spec.dataset_config else None
    raw = load_dataset(spec.dataset_name, cfg, split=split)
    out: list[QAExample] = []
    for idx, row in enumerate(raw):
        question = str(row[spec.question_field])
        answer = str(row[spec.answer_field])
        out.append(
            QAExample(
                question=question,
                answer=answer,
                final_answer=extract_final_answer(answer),
            )
        )
        if limit > 0 and idx + 1 >= limit:
            break
    return out


def to_json_ready(examples: list[QAExample]) -> list[dict[str, Any]]:
    return [{"question": e.question, "answer": e.answer, "final_answer": e.final_answer} for e in examples]

