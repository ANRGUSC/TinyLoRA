from __future__ import annotations

import io
import json
import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from tinylora.adapter import count_trainable_parameters


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_dtype(dtype_name: str) -> torch.dtype:
    key = dtype_name.strip().lower()
    if key in ("bf16", "bfloat16"):
        return torch.bfloat16
    if key in ("fp16", "float16", "half"):
        return torch.float16
    if key in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def grad_norm(parameters: list[torch.nn.Parameter]) -> float:
    total_sq = 0.0
    for p in parameters:
        if p.grad is None:
            continue
        total_sq += float((p.grad.detach().float().norm() ** 2).item())
    return float(total_sq ** 0.5)


def serialize_torch_state(state: dict[str, Any]) -> bytes:
    buffer = io.BytesIO()
    torch.save(state, buffer)
    return buffer.getvalue()


def safe_torch_load(source: Any, map_location: str | torch.device = "cpu") -> Any:
    try:
        return torch.load(source, map_location=map_location, weights_only=True)
    except TypeError:
        # Backward compatibility for torch versions without weights_only.
        return torch.load(source, map_location=map_location)


def deserialize_torch_state(blob: bytes) -> dict[str, Any]:
    buffer = io.BytesIO(blob)
    return safe_torch_load(buffer, map_location="cpu")


def trainable_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: param.detach().cpu() for name, param in model.named_parameters() if param.requires_grad}


def load_trainable_state_dict(model: torch.nn.Module, state: dict[str, torch.Tensor]) -> None:
    model_params = dict(model.named_parameters())
    for name, tensor in state.items():
        if name not in model_params:
            continue
        target = model_params[name]
        if target.requires_grad:
            with torch.no_grad():
                target.copy_(tensor.to(dtype=target.dtype, device=target.device))


def build_run_meta(
    *,
    run_id: str,
    model_name: str,
    trainable_params: int,
    budget_target: int,
    budget_actual: int,
    lr: float,
    seed: int,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "model_name": model_name,
        "trainable_params_counted": trainable_params,
        "budget_target": budget_target,
        "budget_actual": budget_actual,
        "lr": lr,
        "seed": seed,
    }


@dataclass(frozen=True)
class BudgetChoice:
    projection_dim: int
    tie_factor: int
    actual_params: int


def verify_trainable_budget(model: torch.nn.Module, expected: int) -> None:
    actual = count_trainable_parameters(model)
    if actual != expected:
        raise RuntimeError(f"Trainable parameter mismatch: expected={expected}, actual={actual}")
