from __future__ import annotations

"""SFT TinyLoRA training entrypoint for causal language models."""

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tinylora.adapter import TinyLoRAConfig, choose_budget_plan, count_trainable_parameters, inject_tinylora
from tinylora.checkpoint_manager import CheckpointManager
from tinylora.data import QADatasetSpec, build_prompt, build_sft_target, load_qa_split
from tinylora.heartbeat import write_heartbeat
from tinylora.reward import exact_match_reward
from tinylora.training_utils import (
    build_run_meta,
    choose_dtype,
    grad_norm,
    load_trainable_state_dict,
    safe_torch_load,
    serialize_torch_state,
    set_global_seed,
    trainable_state_dict,
    verify_trainable_budget,
)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TinyLoRA with SFT on a HF question-answer dataset.")
    parser.add_argument("--root-dir", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--frozen-rank", type=int, default=2)
    parser.add_argument("--projection-candidates", default="1,2,4")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--eval-max-new-tokens", type=int, default=256)
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--eval-limit", type=int, default=256)
    parser.add_argument("--checkpoint-every-steps", type=int, default=20)
    parser.add_argument("--heartbeat-every-steps", type=int, default=5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dataset-name", default="gsm8k")
    parser.add_argument("--dataset-config", default="main")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="test")
    parser.add_argument("--question-field", default="question")
    parser.add_argument("--answer-field", default="answer")
    return parser.parse_args(argv)


def _target_linear_modules(model: torch.nn.Module, targets: tuple[str, ...]) -> list[str]:
    names = []
    target_set = set(targets)
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and name.split(".")[-1] in target_set:
            names.append(name)
    return names


def _build_batch(tokenizer, examples, max_length: int, device: torch.device) -> dict[str, torch.Tensor]:
    input_ids_list = []
    label_ids_list = []
    attn_list = []

    for ex in examples:
        prompt = build_prompt(ex.question)
        target = build_sft_target(ex.answer)
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        target_ids = tokenizer(target, add_special_tokens=False)["input_ids"]
        full_ids = (prompt_ids + target_ids)[:max_length]
        prompt_cut = min(len(prompt_ids), len(full_ids))
        labels = full_ids.copy()
        for i in range(prompt_cut):
            labels[i] = -100

        input_ids_list.append(torch.tensor(full_ids, dtype=torch.long))
        label_ids_list.append(torch.tensor(labels, dtype=torch.long))
        attn_list.append(torch.ones(len(full_ids), dtype=torch.long))

    padded_input = torch.nn.utils.rnn.pad_sequence(
        input_ids_list, batch_first=True, padding_value=tokenizer.pad_token_id
    ).to(device)
    padded_labels = torch.nn.utils.rnn.pad_sequence(label_ids_list, batch_first=True, padding_value=-100).to(device)
    padded_attn = torch.nn.utils.rnn.pad_sequence(attn_list, batch_first=True, padding_value=0).to(device)
    return {"input_ids": padded_input, "labels": padded_labels, "attention_mask": padded_attn}


def _generate_with_eval_mode(model: torch.nn.Module, **kwargs) -> torch.Tensor:
    was_training = bool(model.training)
    model.eval()
    with torch.no_grad():
        outputs = model.generate(**kwargs)
    if was_training:
        model.train()
    return outputs


@torch.no_grad()
def evaluate_pass_at_1(model, tokenizer, examples, max_length: int, max_new_tokens: int, device: torch.device) -> float:
    if not examples:
        return 0.0
    correct = 0.0
    total = 0
    for ex in examples:
        prompt = build_prompt(ex.question)
        enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        out = _generate_with_eval_mode(
            model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        completion = out[0, input_ids.shape[1] :]
        pred_text = tokenizer.decode(completion, skip_special_tokens=True)
        correct += exact_match_reward(pred_text, ex.answer)
        total += 1
    return float(correct / max(1, total))


def _save_training_checkpoint(
    manager: CheckpointManager,
    run_id: str,
    step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_state: dict[str, Any],
    metadata: dict[str, Any],
) -> None:
    files = {
        "train_state.json": json.dumps(train_state, indent=2, sort_keys=True).encode("utf-8"),
        "adapter_state.pt": serialize_torch_state(trainable_state_dict(model)),
        "optim_state.pt": serialize_torch_state(optimizer.state_dict()),
    }
    manager.save_files_checkpoint(run_id=run_id, step=step, files=files, metadata=metadata)


def _try_resume(
    manager: CheckpointManager,
    run_id: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> dict[str, Any] | None:
    latest = manager.find_latest_valid_checkpoint(run_id)
    if latest is None:
        return None
    ckpt_dir = latest.checkpoint_dir
    train_state_path = ckpt_dir / "train_state.json"
    adapter_state_path = ckpt_dir / "adapter_state.pt"
    optim_state_path = ckpt_dir / "optim_state.pt"
    if not train_state_path.exists() or not adapter_state_path.exists() or not optim_state_path.exists():
        return None
    train_state = json.loads(train_state_path.read_text(encoding="utf-8"))
    adapter_state = safe_torch_load(adapter_state_path, map_location="cpu")
    load_trainable_state_dict(model, adapter_state)
    optim_state = safe_torch_load(optim_state_path, map_location="cpu")
    optimizer.load_state_dict(optim_state)
    return train_state


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    set_global_seed(args.seed)

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:  # pragma: no cover
        print(f"Missing transformers dependency: {exc}")
        return 2

    root_dir = Path(args.root_dir).resolve()
    run_root = root_dir / "runs" / args.run_id
    metrics_dir = run_root / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_dir / "metrics.jsonl"
    summary_path = metrics_dir / "summary.csv"

    dtype = choose_dtype(args.dtype)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=dtype)
    model.to(device)
    model.train()

    module_names = _target_linear_modules(model, TinyLoRAConfig().target_modules)
    if len(module_names) == 0:
        print(
            "No target linear modules found for TinyLoRA injection. "
            "This script expects Qwen/LLaMA-style projection names (q_proj/k_proj/v_proj/o_proj/up_proj/down_proj/gate_proj)."
        )
        return 2
    proj_candidates = [int(x) for x in args.projection_candidates.split(",") if x.strip()]
    plan = choose_budget_plan(num_modules=len(module_names), target_params=args.budget, projection_dim_candidates=proj_candidates)
    cfg = TinyLoRAConfig(
        frozen_rank=args.frozen_rank,
        projection_dim=int(plan["projection_dim"]),
        tie_mode="tiled",
        tie_factor=int(plan["tie_factor"]),
        seed=args.seed,
    )
    inject_summary = inject_tinylora(model, cfg)
    verify_trainable_budget(model, inject_summary["trainable_params_expected"])

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr)
    manager = CheckpointManager(root_dir=root_dir, keep_last_n=8)

    ds = QADatasetSpec(
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        question_field=args.question_field,
        answer_field=args.answer_field,
        train_split=args.train_split,
        eval_split=args.eval_split,
    )
    train_data = load_qa_split(ds, split=ds.train_split, limit=args.train_limit)
    eval_data = load_qa_split(ds, split=ds.eval_split, limit=args.eval_limit)
    if not train_data:
        print("No training data loaded.")
        return 2

    steps_per_epoch = math.ceil(len(train_data) / max(1, args.batch_size))
    max_steps = args.epochs * steps_per_epoch
    train_state = {"global_step": 0, "epoch": 0, "cursor": 0}
    if args.resume:
        restored = _try_resume(manager, args.run_id, model, optimizer)
        if restored is not None:
            train_state.update(restored)

    run_meta = build_run_meta(
        run_id=args.run_id,
        model_name=args.model_name,
        trainable_params=count_trainable_parameters(model),
        budget_target=args.budget,
        budget_actual=inject_summary["trainable_params_expected"],
        lr=args.lr,
        seed=args.seed,
    )
    run_meta["inject_summary"] = inject_summary
    run_meta["dataset"] = {
        "dataset_name": ds.dataset_name,
        "dataset_config": ds.dataset_config,
        "train_split": ds.train_split,
        "eval_split": ds.eval_split,
        "question_field": ds.question_field,
        "answer_field": ds.answer_field,
    }
    run_meta_path = run_root / "run_meta.json"
    run_meta_path.write_text(json.dumps(run_meta, indent=2, sort_keys=True), encoding="utf-8")

    rng = random.Random(args.seed + 17)
    order = list(range(len(train_data)))
    rng.shuffle(order)

    for step_idx in range(train_state["global_step"], max_steps):
        if train_state["cursor"] >= len(order):
            train_state["cursor"] = 0
            train_state["epoch"] += 1
            rng.shuffle(order)

        ids = order[train_state["cursor"] : train_state["cursor"] + args.batch_size]
        train_state["cursor"] += len(ids)
        batch = [train_data[i] for i in ids]

        optimizer.zero_grad(set_to_none=True)
        batch_tensors = _build_batch(tokenizer, batch, max_length=args.max_length, device=device)
        out = model(
            input_ids=batch_tensors["input_ids"],
            attention_mask=batch_tensors["attention_mask"],
            labels=batch_tensors["labels"],
        )
        loss = out.loss
        if torch.isnan(loss) or torch.isinf(loss):
            print("Encountered invalid loss; stopping run.")
            return 2

        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
        gnorm = grad_norm(trainable)
        optimizer.step()

        train_state["global_step"] = step_idx + 1
        metric_row = {
            "step": train_state["global_step"],
            "epoch": train_state["epoch"],
            "loss": float(loss.detach().cpu().item()),
            "reward": 0.0,
            "grad_norm": gnorm,
            "budget": args.budget,
            "lr": args.lr,
            "seed": args.seed,
        }
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(metric_row, sort_keys=True))
            handle.write("\n")

        if train_state["global_step"] % max(1, args.heartbeat_every_steps) == 0:
            write_heartbeat(root_dir=root_dir, run_id=args.run_id, payload={"step": train_state["global_step"]})
        if train_state["global_step"] % max(1, args.checkpoint_every_steps) == 0:
            _save_training_checkpoint(
                manager=manager,
                run_id=args.run_id,
                step=train_state["global_step"],
                model=model,
                optimizer=optimizer,
                train_state=train_state,
                metadata=run_meta,
            )

    _save_training_checkpoint(
        manager=manager,
        run_id=args.run_id,
        step=train_state["global_step"],
        model=model,
        optimizer=optimizer,
        train_state=train_state,
        metadata=run_meta,
    )
    pass_at_1 = evaluate_pass_at_1(
        model=model,
        tokenizer=tokenizer,
        examples=eval_data,
        max_length=args.max_length,
        max_new_tokens=args.eval_max_new_tokens,
        device=device,
    )
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["trainable_params", "pass_at_1", "lr", "seed"])
        writer.writeheader()
        writer.writerow(
            {
                "trainable_params": inject_summary["trainable_params_expected"],
                "pass_at_1": f"{pass_at_1:.6f}",
                "lr": args.lr,
                "seed": args.seed,
            }
        )
    print(json.dumps({"run_id": args.run_id, "pass_at_1": pass_at_1}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
