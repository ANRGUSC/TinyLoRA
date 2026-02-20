# TinyLoRA Public Research Pack

Reference implementation of TinyLoRA-style ultra-low-parameter post-training, with:

- A reproducible **budget=13** sweep workflow (paper-target budget).
- A **2D SFT loss-surface** scanner/visualizer for budget=2.
- Unit tests for core adapter math, checkpoints, reward extraction, and loss-surface helpers.

## Attribution

- **Repository author:** Bhaskar Krishnamachari
- **Implementation support:** Developed with OpenAI Codex (GPT-5 family coding agent)

## Scope and Paper Alignment

This code is designed to be faithful to the TinyLoRA core parameterization from:

- John X. Morris, Niloofar Mireshghallah, Mark Ibrahim, Saeed Mahloujifar,  
  *Learning to Reason in 13 Parameters* (arXiv:2602.04118).
- Paper PDF: https://arxiv.org/pdf/2602.04118

Implemented core elements:

- Frozen base model weights.
- TinyLoRA adapter with frozen SVD factors and a tiny trainable coefficient vector.
- Tie-sharing across target projection modules.
- Budget planning to match very small trainable parameter counts (including 13).
- `\boxed{...}` and `#### ...` exact-match answer extraction for math-style outputs.

## Layout

- `tinylora/` core library
  - `adapter.py` TinyLoRA injection, budget planning, merge/unmerge
  - `data.py` configurable Hugging Face QA dataset loader
  - `reward.py` numeric-answer extraction + exact-match reward
  - `checkpoint_manager.py` atomic checkpoints with hash manifests
- `scripts/`
  - `train_grpo_tinylora.py` GRPO-style RL training
  - `train_sft_tinylora.py` SFT baseline
  - `run_budget13_sweep.py` LR sweep runner for budget=13
  - `analysis/scan_loss_surface.py` 2D SFT loss-surface scan
- `notebooks/`
  - `TinyLoRA_Replication_13p.ipynb`
  - `TinyLoRA_2D_Loss_Surface.ipynb`
- `tests/` unit tests

## Install

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

## Run Tests

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

## Config Presets

- Fast/dev preset: `configs/replication_13p_fast.yaml`
- Paper-aligned GSM8K preset: `configs/replication_13p_paper_gsm8k.yaml`
  - Uses large settings (`batch_prompts=64`, `max_new_tokens=4096`, full train/eval limits).
  - Requires substantially more compute than the fast preset.

## Reproduce 13-Parameter Sweep (Fast Preset)

```bash
python scripts/run_budget13_sweep.py \
  --root-dir ./outputs/replication_13p \
  --model-name Qwen/Qwen2.5-7B-Instruct \
  --budget 13 \
  --seed 1 \
  --epochs 1 \
  --batch-prompts 2 \
  --group-size 2 \
  --train-limit 64 \
  --eval-limit 64 \
  --max-prompt-tokens 512 \
  --max-new-tokens 128 \
  --resume
```

Outputs:

- `outputs/replication_13p/results/budget13_sweep.csv`
- `outputs/replication_13p/results/budget13_best.json`

For higher-fidelity runs, increase:

- `--train-limit`, `--eval-limit`
- `--epochs`
- decoding lengths

For a paper-aligned GSM8K run target, see `configs/replication_13p_paper_gsm8k.yaml`.

## Run 2D SFT Loss Surface (Budget=2)

```bash
python scripts/analysis/scan_loss_surface.py \
  --root-dir ./outputs/loss_surface_b2_20x20 \
  --model-name Qwen/Qwen2.5-7B-Instruct \
  --budget 2 \
  --grid-size 20 \
  --vmin -0.05 \
  --vmax 0.05 \
  --train-limit 128 \
  --batch-size 4 \
  --max-length 512 \
  --dtype bfloat16 \
  --seed 1 \
  --dataset-name gsm8k \
  --dataset-config main \
  --train-split train \
  --question-field question \
  --answer-field answer \
  --resume
```

Outputs:

- `loss_surface.csv`
- `meta.json`
- `best_point.json`

Use the notebook `notebooks/TinyLoRA_2D_Loss_Surface.ipynb` to visualize heatmap/contours.

## Reports

- Loss-surface writeup PDF: [`reports/tinylora_loss_surface_report.pdf`](reports/tinylora_loss_surface_report.pdf)

## Known Limitations / TODOs

- GRPO in this repo is simplified REINFORCE-style (group-relative baseline with summed log-probs), not a full PPO-style GRPO implementation.
- Full per-token policy-ratio clipping, explicit importance sampling ratios, and integrated KL-penalty machinery are not implemented in the current RL script.
- The paper's VERL + vLLM training/inference stack is not reproduced here; scripts use standard Hugging Face `transformers`.
- The GRPO training loop currently favors clarity over maximal memory efficiency and is not fully batched across all prompt/group axes.
- Adapter scaling uses `alpha / projection_dim`; this is an implementation choice and differs from a literal reading of the paper formula when `projection_dim > 1`.

## Swapping Model or Dataset

Both training scripts and the loss-surface scanner support:

- `--model-name` (any Hugging Face causal LM with compatible linear projection names)
- `--dataset-name`, `--dataset-config`
- split names and QA field names

Example:

```bash
python scripts/train_sft_tinylora.py \
  --root-dir ./outputs/custom \
  --run-id custom_sft_b13 \
  --model-name <your-model> \
  --budget 13 \
  --lr 1e-5 \
  --seed 1 \
  --dataset-name <hf-dataset> \
  --dataset-config <config-or-empty-string> \
  --train-split train \
  --eval-split validation \
  --question-field prompt \
  --answer-field response
```

## License

Apache License 2.0 (see `LICENSE`).
