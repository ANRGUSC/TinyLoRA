"""TinyLoRA core modules for parameter-efficient adaptation and small-budget training."""

from .adapter import (
    TinyLoRAConfig,
    choose_budget_plan,
    choose_tie_factor_for_budget,
    count_trainable_parameters,
    inject_tinylora,
    merge_tinylora,
    unmerge_tinylora,
)
from .checkpoint_manager import CheckpointManager
from .data import QAExample, QADatasetSpec, load_qa_split
from .gsm8k import GSM8KExample, build_prompt, build_sft_target, load_gsm8k_split
from .heartbeat import read_latest_heartbeat, write_heartbeat
from .notifier import notify_tinylora
from .reward import exact_match_reward, extract_final_answer

__all__ = [
    "TinyLoRAConfig",
    "choose_budget_plan",
    "choose_tie_factor_for_budget",
    "count_trainable_parameters",
    "inject_tinylora",
    "merge_tinylora",
    "unmerge_tinylora",
    "CheckpointManager",
    "QAExample",
    "QADatasetSpec",
    "load_qa_split",
    "GSM8KExample",
    "build_prompt",
    "build_sft_target",
    "load_gsm8k_split",
    "exact_match_reward",
    "extract_final_answer",
    "read_latest_heartbeat",
    "write_heartbeat",
    "notify_tinylora",
]
