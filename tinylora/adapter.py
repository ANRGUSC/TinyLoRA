from __future__ import annotations

"""
TinyLoRA adapter implementation.

Core idea:
- Freeze a base linear weight W.
- Compute frozen SVD factors (U, S, V).
- Learn only a tiny coefficient vector `v` that mixes fixed random projections.
- Apply the induced low-rank delta during forward pass.

This implementation is budget-first: it includes utilities to choose tie sharing
and projection dimension for tiny trainable-parameter targets (e.g., 13).
"""

import math
import re
import hashlib
from dataclasses import dataclass
from typing import Callable, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class TinyLoRAConfig:
    frozen_rank: int = 2
    projection_dim: int = 1
    alpha: float = 1.0
    tie_mode: str = "tiled"  # one of: none, all, tiled, structured
    tie_factor: int = 1
    seed: int = 13
    target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "up_proj",
        "down_proj",
        "gate_proj",
    )


def _extract_layer_index(name: str) -> int:
    for token in name.split("."):
        if token.isdigit():
            return int(token)
    m = re.search(r"(\d+)", name)
    return int(m.group(1)) if m else -1


def _module_type(name: str) -> str:
    return name.split(".")[-1]


def _chunk_ids(items: list[str], chunk_size: int) -> dict[str, int]:
    if chunk_size <= 1:
        return {name: idx for idx, name in enumerate(items)}
    result: dict[str, int] = {}
    gid = 0
    count = 0
    for name in items:
        result[name] = gid
        count += 1
        if count >= chunk_size:
            gid += 1
            count = 0
    return result


def build_tie_groups(module_names: list[str], tie_mode: str, tie_factor: int) -> dict[str, int]:
    if not module_names:
        return {}
    tie_mode = tie_mode.lower()
    tie_factor = max(1, tie_factor)

    if tie_mode == "none":
        return {name: idx for idx, name in enumerate(module_names)}
    if tie_mode == "all":
        return {name: 0 for name in module_names}

    if tie_mode == "tiled":
        ordered = sorted(module_names, key=lambda n: (_extract_layer_index(n), n))
        return _chunk_ids(ordered, tie_factor)

    if tie_mode == "structured":
        grouped: dict[str, list[str]] = {}
        for name in module_names:
            grouped.setdefault(_module_type(name), []).append(name)
        result: dict[str, int] = {}
        gid_offset = 0
        for key in sorted(grouped.keys()):
            ordered = sorted(grouped[key], key=lambda n: (_extract_layer_index(n), n))
            local = _chunk_ids(ordered, tie_factor)
            local_groups = (max(local.values()) + 1) if local else 0
            for n, g in local.items():
                result[n] = gid_offset + g
            gid_offset += local_groups
        return result

    raise ValueError(f"Unsupported tie_mode={tie_mode}")


def estimate_trainable_params(num_modules: int, projection_dim: int, tie_factor: int) -> int:
    groups = math.ceil(num_modules / max(1, tie_factor))
    return groups * max(1, projection_dim)


def choose_tie_factor_for_budget(num_modules: int, projection_dim: int, target_params: int) -> tuple[int, int]:
    if num_modules <= 0:
        return 1, 0
    best_tie = 1
    best_params = estimate_trainable_params(num_modules, projection_dim, best_tie)
    best_score = abs(best_params - target_params)
    for tie in range(1, num_modules + 1):
        actual = estimate_trainable_params(num_modules, projection_dim, tie)
        score = abs(actual - target_params)
        # Prefer fewer params when equally close, for cost control.
        if score < best_score or (score == best_score and actual < best_params):
            best_score = score
            best_params = actual
            best_tie = tie
    return best_tie, best_params


def choose_budget_plan(
    num_modules: int, target_params: int, projection_dim_candidates: Iterable[int] = (1, 2, 4, 8)
) -> dict[str, int]:
    best = {
        "projection_dim": 1,
        "tie_factor": 1,
        "actual_params": estimate_trainable_params(num_modules, 1, 1),
    }
    best_score = abs(best["actual_params"] - target_params)
    for u in projection_dim_candidates:
        tie, actual = choose_tie_factor_for_budget(num_modules, u, target_params)
        score = abs(actual - target_params)
        if score < best_score or (score == best_score and actual < best["actual_params"]):
            best = {"projection_dim": u, "tie_factor": tie, "actual_params": actual}
            best_score = score
    return best


class TinyLoRALinear(nn.Module):
    def __init__(
        self,
        linear: nn.Linear,
        frozen_rank: int,
        projection_dim: int,
        alpha: float,
        seed: int,
        shared_v: nn.Parameter | None = None,
    ) -> None:
        super().__init__()
        if frozen_rank < 1:
            raise ValueError("frozen_rank must be >= 1")
        if projection_dim < 1:
            raise ValueError("projection_dim must be >= 1")

        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.frozen_rank = min(frozen_rank, self.in_features, self.out_features)
        self.projection_dim = projection_dim
        self.alpha = float(alpha)
        self.scale = self.alpha / float(max(1, self.projection_dim))
        self.merged = False
        self.enabled = True

        weight = linear.weight.detach()
        device = weight.device
        self.weight = nn.Parameter(weight.clone(), requires_grad=False)
        if linear.bias is not None:
            self.bias = nn.Parameter(linear.bias.detach().clone(), requires_grad=False)
        else:
            self.bias = None

        # Frozen SVD factors.
        svd_weight = weight.to(torch.float32)
        u, s, vh = torch.linalg.svd(svd_weight, full_matrices=False)
        u = u[:, : self.frozen_rank].contiguous()
        s = s[: self.frozen_rank].contiguous()
        v = vh[: self.frozen_rank, :].transpose(0, 1).contiguous()  # [in, r]
        self.register_buffer("U", u)
        self.register_buffer("S", s)
        self.register_buffer("V", v)

        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        p = torch.randn(
            self.projection_dim,
            self.frozen_rank,
            self.frozen_rank,
            generator=generator,
            dtype=torch.float32,
        ).to(device=device) / math.sqrt(float(self.frozen_rank))
        self.register_buffer("P", p)

        if shared_v is None:
            self.v = nn.Parameter(
                torch.zeros(self.projection_dim, dtype=torch.float32, device=device),
                requires_grad=True,
            )
        else:
            if shared_v.device != device:
                raise RuntimeError(
                    f"shared_v device mismatch: shared_v={shared_v.device}, expected={device}"
                )
            self.v = shared_v

    def _mix_matrix(self) -> torch.Tensor:
        # [u] x [u, r, r] -> [r, r]
        return torch.tensordot(self.v, self.P, dims=([0], [0]))

    def delta_weight(self) -> torch.Tensor:
        m = self._mix_matrix()
        us = self.U * self.S.unsqueeze(0)
        return (us @ m @ self.V.transpose(0, 1)).to(self.weight.dtype)

    def merge(self) -> None:
        if self.merged:
            return
        with torch.no_grad():
            self.weight.add_(self.delta_weight() * self.scale)
        self.merged = True

    def unmerge(self) -> None:
        if not self.merged:
            return
        with torch.no_grad():
            self.weight.sub_(self.delta_weight() * self.scale)
        self.merged = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.linear(x, self.weight, self.bias)
        if self.merged or (not self.enabled):
            return base

        x_flat = x.reshape(-1, x.shape[-1]).to(torch.float32)
        m = self._mix_matrix()  # [r, r]
        proj = x_flat @ self.V  # [n, r]
        proj = proj @ m.transpose(0, 1)
        proj = proj * self.S.unsqueeze(0)
        delta = proj @ self.U.transpose(0, 1)  # [n, out]
        delta = delta.reshape(*x.shape[:-1], self.out_features).to(base.dtype)
        return base + (delta * self.scale)


def _set_child_module(parent: nn.Module, child_name: str, new_module: nn.Module) -> None:
    setattr(parent, child_name, new_module)


def _resolve_parent(model: nn.Module, module_path: str) -> tuple[nn.Module, str]:
    tokens = module_path.split(".")
    parent = model
    for token in tokens[:-1]:
        parent = getattr(parent, token)
    return parent, tokens[-1]


def _matches_target(name: str, targets: tuple[str, ...]) -> bool:
    suffix = name.split(".")[-1]
    return suffix in set(targets)


def iter_tinylora_modules(model: nn.Module):
    for name, module in model.named_modules():
        if isinstance(module, TinyLoRALinear):
            yield name, module


def merge_tinylora(model: nn.Module) -> None:
    for _, module in iter_tinylora_modules(model):
        module.merge()


def unmerge_tinylora(model: nn.Module) -> None:
    for _, module in iter_tinylora_modules(model):
        module.unmerge()


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def inject_tinylora(
    model: nn.Module,
    config: TinyLoRAConfig,
    module_filter: Callable[[str, nn.Module], bool] | None = None,
    freeze_base: bool = True,
) -> dict[str, int]:
    if freeze_base:
        for param in model.parameters():
            param.requires_grad_(False)

    candidates: list[tuple[str, nn.Linear]] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if module_filter is not None:
            if not module_filter(name, module):
                continue
        else:
            if not _matches_target(name, config.target_modules):
                continue
        candidates.append((name, module))

    names = [name for name, _ in candidates]
    groups = build_tie_groups(names, tie_mode=config.tie_mode, tie_factor=config.tie_factor)

    group_devices: dict[int, torch.device] = {}
    for name, linear in candidates:
        group_id = groups[name]
        dev = linear.weight.device
        if group_id in group_devices and group_devices[group_id] != dev:
            raise RuntimeError(
                f"TinyLoRA tie group {group_id} spans multiple devices: "
                f"{group_devices[group_id]} vs {dev}"
            )
        group_devices[group_id] = dev

    shared_bank: dict[int, nn.Parameter] = {}
    for group_id in sorted(set(groups.values())):
        # Deterministic per-group initialization.
        seed = int(config.seed + (group_id * 9973))
        g = torch.Generator(device="cpu")
        g.manual_seed(seed)
        dev = group_devices[group_id]
        v = torch.zeros(config.projection_dim, dtype=torch.float32, device=dev)
        # Small random init works better than exact zero to start updates.
        v += 1e-3 * torch.randn(config.projection_dim, generator=g, dtype=torch.float32).to(device=dev)
        shared_bank[group_id] = nn.Parameter(v, requires_grad=True)

    replaced = 0
    for name, linear in candidates:
        parent, child_name = _resolve_parent(model, name)
        group_id = groups[name]
        digest = hashlib.sha256(name.encode("utf-8")).hexdigest()
        module_seed = int(config.seed + (int(digest[:8], 16) % 100000))
        wrapped = TinyLoRALinear(
            linear=linear,
            frozen_rank=config.frozen_rank,
            projection_dim=config.projection_dim,
            alpha=config.alpha,
            seed=module_seed,
            shared_v=shared_bank[group_id],
        )
        _set_child_module(parent, child_name, wrapped)
        replaced += 1

    unique_params = len(set(groups.values()))
    trainable_params = unique_params * config.projection_dim
    return {
        "replaced_modules": replaced,
        "unique_shared_vectors": unique_params,
        "trainable_params_expected": trainable_params,
    }
