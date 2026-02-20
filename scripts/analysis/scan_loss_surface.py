from __future__ import annotations

"""Scan a 2D TinyLoRA SFT-loss surface for budget=2 adapters."""

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tinylora.adapter import TinyLoRAConfig, choose_budget_plan, count_trainable_parameters, inject_tinylora
from tinylora.data import QADatasetSpec, build_prompt, build_sft_target, load_qa_split
from tinylora.training_utils import choose_dtype, set_global_seed


@dataclass(frozen=True)
class OutputPaths:
    csv_path: Path
    meta_path: Path
    best_path: Path


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan 2-parameter TinyLoRA SFT-loss landscape on a fixed QA data slice.")
    parser.add_argument("--root-dir", required=True, help="Output folder for CSV/metadata.")
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--budget", type=int, default=2)
    parser.add_argument("--grid-size", type=int, default=20)
    parser.add_argument("--vmin", type=float, default=-0.05)
    parser.add_argument("--vmax", type=float, default=0.05)
    parser.add_argument("--train-limit", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--frozen-rank", type=int, default=2)
    parser.add_argument("--projection-candidates", default="1,2,4")
    parser.add_argument("--dataset-name", default="gsm8k")
    parser.add_argument("--dataset-config", default="main")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--question-field", default="question")
    parser.add_argument("--answer-field", default="answer")
    return parser.parse_args(argv)


def build_grid(vmin: float, vmax: float, grid_size: int) -> list[float]:
    if grid_size < 2:
        return [float(vmin)]
    step = (vmax - vmin) / float(grid_size - 1)
    return [float(vmin + (step * i)) for i in range(grid_size)]


def choose_output_paths(root_dir: Path, resume: bool) -> OutputPaths:
    root_dir.mkdir(parents=True, exist_ok=True)
    base_csv = root_dir / "loss_surface.csv"
    base_meta = root_dir / "meta.json"
    base_best = root_dir / "best_point.json"

    if resume:
        return OutputPaths(csv_path=base_csv, meta_path=base_meta, best_path=base_best)

    if base_csv.exists() or base_meta.exists() or base_best.exists():
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        return OutputPaths(
            csv_path=root_dir / f"loss_surface_{stamp}.csv",
            meta_path=root_dir / f"meta_{stamp}.json",
            best_path=root_dir / f"best_point_{stamp}.json",
        )

    return OutputPaths(csv_path=base_csv, meta_path=base_meta, best_path=base_best)


def load_completed_points(csv_path: Path) -> set[tuple[int, int]]:
    done: set[tuple[int, int]] = set()
    if not csv_path.exists():
        return done
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                i = int(row["i"])
                j = int(row["j"])
            except Exception:
                continue
            done.add((i, j))
    return done


def append_surface_row(csv_path: Path, row: dict[str, Any]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    exists = csv_path.exists()
    with csv_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["i", "j", "p1", "p2", "loss"])
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def read_surface_rows(csv_path: Path) -> list[dict[str, Any]]:
    if not csv_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                rows.append(
                    {
                        "i": int(row["i"]),
                        "j": int(row["j"]),
                        "p1": float(row["p1"]),
                        "p2": float(row["p2"]),
                        "loss": float(row["loss"]),
                    }
                )
            except Exception:
                continue
    return rows


def _target_linear_modules(model: torch.nn.Module, targets: tuple[str, ...]) -> list[str]:
    names: list[str] = []
    target_set = set(targets)
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and name.split(".")[-1] in target_set:
            names.append(name)
    return names


def encode_sft_examples(tokenizer, examples, max_length: int) -> list[dict[str, torch.Tensor]]:
    encoded: list[dict[str, torch.Tensor]] = []
    for ex in examples:
        prompt = build_prompt(ex.question)
        target = build_sft_target(ex.answer)
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        target_ids = tokenizer(target, add_special_tokens=False)["input_ids"]
        full_ids = (prompt_ids + target_ids)[:max_length]

        prompt_cut = min(len(prompt_ids), len(full_ids))
        labels = full_ids.copy()
        for idx in range(prompt_cut):
            labels[idx] = -100

        encoded.append(
            {
                "input_ids": torch.tensor(full_ids, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
                "attention_mask": torch.ones(len(full_ids), dtype=torch.long),
            }
        )
    return encoded


def _to_batch(samples: list[dict[str, torch.Tensor]], pad_token_id: int, device: torch.device) -> dict[str, torch.Tensor]:
    inputs = [s["input_ids"] for s in samples]
    labels = [s["labels"] for s in samples]
    attn = [s["attention_mask"] for s in samples]
    return {
        "input_ids": torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=pad_token_id).to(device),
        "labels": torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100).to(device),
        "attention_mask": torch.nn.utils.rnn.pad_sequence(attn, batch_first=True, padding_value=0).to(device),
    }


def mean_ce_loss(
    model: torch.nn.Module,
    encoded: list[dict[str, torch.Tensor]],
    batch_size: int,
    pad_token_id: int,
    device: torch.device,
) -> float:
    total_nll = 0.0
    total_tokens = 0
    model.eval()
    with torch.no_grad():
        for start in range(0, len(encoded), max(1, batch_size)):
            batch = _to_batch(encoded[start : start + max(1, batch_size)], pad_token_id=pad_token_id, device=device)
            logits = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            ).logits
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = batch["labels"][:, 1:].contiguous()
            nll = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
                reduction="sum",
            )
            token_count = int((shift_labels != -100).sum().item())
            total_nll += float(nll.detach().cpu().item())
            total_tokens += token_count
    if total_tokens == 0:
        return float("nan")
    return float(total_nll / float(total_tokens))


def collect_trainable_params(model: torch.nn.Module) -> list[tuple[str, torch.nn.Parameter]]:
    return [(name, param) for name, param in model.named_parameters() if param.requires_grad]


def trainable_scalar_count(params: list[tuple[str, torch.nn.Parameter]]) -> int:
    return sum(param.numel() for _, param in params)


def set_trainable_from_values(params: list[tuple[str, torch.nn.Parameter]], values: list[float]) -> None:
    expected = trainable_scalar_count(params)
    if len(values) != expected:
        raise ValueError(f"value length mismatch: got={len(values)}, expected={expected}")
    offset = 0
    with torch.no_grad():
        for _, param in params:
            n = param.numel()
            chunk = values[offset : offset + n]
            value_tensor = torch.tensor(chunk, dtype=param.dtype, device=param.device).view_as(param)
            param.copy_(value_tensor)
            offset += n


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    set_global_seed(args.seed)

    if args.grid_size <= 0:
        raise ValueError("--grid-size must be >= 1")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be >= 1")

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:  # pragma: no cover
        print(f"Missing transformers dependency: {exc}")
        return 2

    root_dir = Path(args.root_dir).resolve()
    outputs = choose_output_paths(root_dir=root_dir, resume=args.resume)

    dtype = choose_dtype(args.dtype)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=dtype)
    model.to(device)
    model.eval()

    module_names = _target_linear_modules(model, TinyLoRAConfig().target_modules)
    if not module_names:
        print("No target linear modules found for TinyLoRA injection.")
        return 2
    proj_candidates = [int(x) for x in args.projection_candidates.split(",") if x.strip()]
    plan = choose_budget_plan(
        num_modules=len(module_names),
        target_params=args.budget,
        projection_dim_candidates=proj_candidates,
    )
    config = TinyLoRAConfig(
        frozen_rank=args.frozen_rank,
        projection_dim=int(plan["projection_dim"]),
        tie_mode="tiled",
        tie_factor=int(plan["tie_factor"]),
        seed=args.seed,
    )
    inject_summary = inject_tinylora(model, config)
    counted_trainable = count_trainable_parameters(model)
    params = collect_trainable_params(model)
    scalar_count = trainable_scalar_count(params)

    if counted_trainable != inject_summary["trainable_params_expected"]:
        raise RuntimeError(
            "Trainable parameter mismatch after injection: "
            f"counted={counted_trainable}, expected={inject_summary['trainable_params_expected']}"
        )
    if scalar_count != args.budget:
        raise RuntimeError(f"Expected exactly budget={args.budget} trainable scalars, got {scalar_count}")
    if scalar_count != 2:
        raise RuntimeError(f"This script expects exactly 2 trainable scalars, got {scalar_count}")

    ds = QADatasetSpec(
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        question_field=args.question_field,
        answer_field=args.answer_field,
        train_split=args.train_split,
        eval_split=args.train_split,
    )
    examples = load_qa_split(ds, split=ds.train_split, limit=args.train_limit)
    if not examples:
        raise RuntimeError("No training examples loaded.")
    encoded = encode_sft_examples(tokenizer, examples, max_length=args.max_length)

    grid_values = build_grid(args.vmin, args.vmax, args.grid_size)
    completed = load_completed_points(outputs.csv_path) if args.resume else set()

    meta = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_name": args.model_name,
        "budget_target": args.budget,
        "grid_size": args.grid_size,
        "vmin": args.vmin,
        "vmax": args.vmax,
        "train_limit": args.train_limit,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "dtype": str(dtype),
        "seed": args.seed,
        "device": str(device),
        "projection_candidates": proj_candidates,
        "dataset": {
            "dataset_name": ds.dataset_name,
            "dataset_config": ds.dataset_config,
            "train_split": ds.train_split,
            "question_field": ds.question_field,
            "answer_field": ds.answer_field,
        },
        "budget_plan": plan,
        "inject_summary": inject_summary,
        "counted_trainable": counted_trainable,
        "trainable_names": [name for name, _ in params],
        "output_csv": str(outputs.csv_path),
    }
    outputs.meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")

    total_points = args.grid_size * args.grid_size
    scanned = 0
    for i, p1 in enumerate(grid_values):
        for j, p2 in enumerate(grid_values):
            if (i, j) in completed:
                continue
            set_trainable_from_values(params, [p1, p2])
            loss_value = mean_ce_loss(
                model=model,
                encoded=encoded,
                batch_size=args.batch_size,
                pad_token_id=tokenizer.pad_token_id,
                device=device,
            )
            append_surface_row(
                outputs.csv_path,
                {
                    "i": i,
                    "j": j,
                    "p1": f"{p1:.10f}",
                    "p2": f"{p2:.10f}",
                    "loss": f"{loss_value:.10f}",
                },
            )
            scanned += 1
            print(f"[{i},{j}] loss={loss_value:.6f}")

    rows = read_surface_rows(outputs.csv_path)
    if not rows:
        raise RuntimeError("No surface rows found after scan.")
    best = min(rows, key=lambda x: x["loss"])
    worst = max(rows, key=lambda x: x["loss"])
    best_payload = {
        "best_i": best["i"],
        "best_j": best["j"],
        "best_p1": best["p1"],
        "best_p2": best["p2"],
        "best_loss": best["loss"],
        "worst_loss": worst["loss"],
        "dynamic_range": float(worst["loss"] - best["loss"]),
        "rows": len(rows),
        "expected_rows": total_points,
        "newly_scanned_rows": scanned,
        "csv_path": str(outputs.csv_path),
    }
    outputs.best_path.write_text(json.dumps(best_payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(best_payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
