# Technical Audit and Fidelity Notes

This document summarizes a code-level audit of the public TinyLoRA pack.

## 1. TinyLoRA Core (`tinylora/adapter.py`)

### Confirmed properties

- Base linear weights are frozen (`requires_grad=False`).
- Frozen SVD factors (`U`, `S`, `V`) are computed once at injection.
- Trainable parameters are only tiny coefficient vectors (`v`) shared by tie groups.
- Budget planner (`choose_budget_plan`) selects projection dimension + tie factor to match tiny trainable budgets (including 13).
- Merge/unmerge path is available for inference-time weight materialization.

### Correctness fixes included

- Dtype consistency:
  - Wrapped linear layer keeps original base weight dtype.
  - Delta projection is cast safely for accumulation.
- Device consistency:
  - Shared trainable vectors and random projection buffers are created on the same device as their target modules.
  - Guard rails raise clear errors if a tie group spans inconsistent devices.

## 2. Training Loops

### SFT (`scripts/train_sft_tinylora.py`)

- Standard masked next-token CE objective.
- Prompt tokens masked out from supervision.
- Checkpoint + resume support.
- Final exact-match evaluation (numeric-answer extraction).

### GRPO-style RL (`scripts/train_grpo_tinylora.py`)

- Group sampling per prompt.
- Relative reward advantages inside each group.
- Policy-gradient-like loss using generated token log-prob sums.
- Checkpoint + resume support.

## 3. 13-Parameter Reproducibility

- `scripts/run_budget13_sweep.py` performs LR sweep for `budget=13`.
- Records per-run status/scores and best run JSON.
- Intended for reproducible benchmarking of the tiny-parameter regime.

## 4. 2D Loss Surface

- `scripts/analysis/scan_loss_surface.py`:
  - Enforces exactly 2 trainable scalars.
  - Scans full 2D grid.
  - Computes SFT CE loss without parameter updates.
  - Supports safe resume with row-wise incremental writes.

## 5. Known Deviations / Limits

- The RL script is **GRPO-style**, not a claim of bit-exact reproduction of any external training stack.
- Adapter target-module assumptions follow Qwen/LLaMA naming (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `up_proj`, `down_proj`, `gate_proj`).
- Performance/cost tradeoffs depend strongly on model size and decoding limits.

## 6. Test Coverage in This Public Pack

- Adapter behavior + budget planner + dtype/device regression checks.
- Reward extraction exact-match behavior.
- Checkpoint manager integrity behavior.
- Loss-surface helper logic (grid, resume, parameter assignment).

