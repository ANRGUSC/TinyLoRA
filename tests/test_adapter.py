from __future__ import annotations

import unittest

import torch
import torch.nn as nn

from tinylora.adapter import (
    TinyLoRAConfig,
    TinyLoRALinear,
    build_tie_groups,
    choose_budget_plan,
    choose_tie_factor_for_budget,
    count_trainable_parameters,
    inject_tinylora,
    iter_tinylora_modules,
    merge_tinylora,
    unmerge_tinylora,
)


class ToyBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(8, 8)
        self.k_proj = nn.Linear(8, 8)
        self.v_proj = nn.Linear(8, 8)
        self.o_proj = nn.Linear(8, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.o_proj(self.q_proj(x) + self.k_proj(x) + self.v_proj(x))


class ToyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([ToyBlock(), ToyBlock()])
        self.final = nn.Linear(8, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.final(x)


class TinyLoRATests(unittest.TestCase):
    def test_tie_groups(self) -> None:
        names = [
            "layers.0.q_proj",
            "layers.0.k_proj",
            "layers.1.q_proj",
            "layers.1.k_proj",
        ]
        groups = build_tie_groups(names, tie_mode="all", tie_factor=99)
        self.assertEqual(set(groups.values()), {0})
        groups2 = build_tie_groups(names, tie_mode="none", tie_factor=1)
        self.assertEqual(len(set(groups2.values())), len(names))

    def test_budget_plan(self) -> None:
        tie, actual = choose_tie_factor_for_budget(num_modules=224, projection_dim=1, target_params=13)
        self.assertGreaterEqual(tie, 1)
        self.assertLessEqual(abs(actual - 13), 1)
        plan = choose_budget_plan(num_modules=224, target_params=49, projection_dim_candidates=[1, 2, 4])
        self.assertIn("tie_factor", plan)
        self.assertIn("actual_params", plan)

    def test_inject_forward_merge_unmerge(self) -> None:
        torch.manual_seed(1)
        model = ToyModel()
        x = torch.randn(2, 8)
        baseline = model(x).detach()

        cfg = TinyLoRAConfig(
            frozen_rank=2,
            projection_dim=1,
            tie_mode="all",
            tie_factor=999,
            seed=42,
            target_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
        )
        summary = inject_tinylora(model, cfg)
        self.assertEqual(summary["replaced_modules"], 8)
        self.assertEqual(summary["trainable_params_expected"], 1)
        self.assertEqual(count_trainable_parameters(model), 1)

        y1 = model(x)
        merge_tinylora(model)
        y2 = model(x)
        self.assertTrue(torch.allclose(y1, y2, atol=1e-4, rtol=1e-4))
        unmerge_tinylora(model)
        y3 = model(x)
        self.assertTrue(torch.allclose(y1, y3, atol=1e-4, rtol=1e-4))

        # With tiny random init, output should be close to baseline but not always identical.
        self.assertEqual(y1.shape, baseline.shape)

    def test_inject_tiled_unique_vectors(self) -> None:
        model = ToyModel()
        cfg = TinyLoRAConfig(
            frozen_rank=2,
            projection_dim=2,
            tie_mode="tiled",
            tie_factor=2,
            seed=7,
            target_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
        )
        summary = inject_tinylora(model, cfg)
        self.assertEqual(summary["replaced_modules"], 8)
        # 8 modules chunked by 2 => 4 shared vectors, each size 2 => 8 params.
        self.assertEqual(summary["trainable_params_expected"], 8)
        self.assertEqual(count_trainable_parameters(model), 8)

    def test_injected_weights_preserve_base_dtype(self) -> None:
        model = ToyModel().to(dtype=torch.bfloat16)
        cfg = TinyLoRAConfig(
            frozen_rank=2,
            projection_dim=1,
            tie_mode="all",
            tie_factor=999,
            seed=11,
            target_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
        )
        inject_tinylora(model, cfg)
        for _, module in iter_tinylora_modules(model):
            self.assertEqual(module.weight.dtype, torch.bfloat16)
            if module.bias is not None:
                self.assertEqual(module.bias.dtype, torch.bfloat16)
            self.assertEqual(module.v.device, module.weight.device)
            self.assertEqual(module.P.device, module.weight.device)

    def test_direct_tinylora_init_is_deterministic_and_nonzero(self) -> None:
        linear = nn.Linear(8, 8)
        lora1 = TinyLoRALinear(linear=linear, frozen_rank=2, projection_dim=2, alpha=1.0, seed=123)
        lora2 = TinyLoRALinear(linear=linear, frozen_rank=2, projection_dim=2, alpha=1.0, seed=123)
        self.assertTrue(torch.allclose(lora1.v.detach(), lora2.v.detach()))
        self.assertGreater(float(lora1.v.detach().abs().sum().item()), 0.0)


if __name__ == "__main__":
    unittest.main()
