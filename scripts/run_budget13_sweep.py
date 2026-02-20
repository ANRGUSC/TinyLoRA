from __future__ import annotations

"""Convenience sweep runner for budget=13 TinyLoRA GRPO experiments."""

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run TinyLoRA GRPO sweeps for budget=13 (paper-aligned target budget)."
    )
    parser.add_argument("--root-dir", required=True, help="Experiment root directory.")
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--budget", type=int, default=13)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--lrs", default="1e-7,5e-7,1e-6,5e-6,1e-5,1e-4,2e-4")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-prompts", type=int, default=2)
    parser.add_argument("--group-size", type=int, default=2)
    parser.add_argument("--train-limit", type=int, default=64)
    parser.add_argument("--eval-limit", type=int, default=64)
    parser.add_argument("--max-prompt-tokens", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--checkpoint-every-steps", type=int, default=20)
    parser.add_argument("--heartbeat-every-steps", type=int, default=5)
    parser.add_argument("--dataset-name", default="gsm8k")
    parser.add_argument("--dataset-config", default="main")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="test")
    parser.add_argument("--question-field", default="question")
    parser.add_argument("--answer-field", default="answer")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def _run_id(budget: int, lr: float, seed: int) -> str:
    return f"grpo_b{budget}_lr{lr:.1e}_s{seed}".replace("+0", "").replace("+", "")


def _read_pass_at_1(root: Path, run_id: str) -> float | None:
    path = root / "runs" / run_id / "metrics" / "summary.csv"
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return None
    try:
        return float(rows[0]["pass_at_1"])
    except Exception:
        return None


def _load_done(summary_csv: Path) -> set[str]:
    if not summary_csv.exists():
        return set()
    done: set[str] = set()
    with summary_csv.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("status") == "ok":
                rid = row.get("run_id")
                if rid:
                    done.add(rid)
    return done


def _append_row(path: Path, row: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["run_id", "budget", "lr", "seed", "status", "pass_at_1"])
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _write_best(path: Path, rows_path: Path) -> None:
    if not rows_path.exists():
        return
    best_score = float("-inf")
    best_row: dict[str, str] | None = None
    with rows_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("status") != "ok":
                continue
            try:
                score = float(row["pass_at_1"])
            except Exception:
                continue
            if score > best_score:
                best_score = score
                best_row = row
    if best_row is None:
        return
    payload = {
        "run_id": best_row["run_id"],
        "budget": int(best_row["budget"]),
        "lr": float(best_row["lr"]),
        "seed": int(best_row["seed"]),
        "pass_at_1": float(best_row["pass_at_1"]),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    root = Path(args.root_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)

    lrs = [float(x.strip()) for x in args.lrs.split(",") if x.strip()]
    summary_csv = root / "results" / "budget13_sweep.csv"
    best_json = root / "results" / "budget13_best.json"
    done = _load_done(summary_csv) if args.resume else set()

    for idx, lr in enumerate(lrs, start=1):
        run_id = _run_id(args.budget, lr, args.seed)
        if run_id in done:
            print(f"[{idx}/{len(lrs)}] SKIP {run_id} (already status=ok)")
            continue

        cmd = [
            sys.executable,
            str(Path(__file__).resolve().parent / "train_grpo_tinylora.py"),
            "--root-dir",
            str(root),
            "--run-id",
            run_id,
            "--model-name",
            args.model_name,
            "--budget",
            str(args.budget),
            "--lr",
            str(lr),
            "--seed",
            str(args.seed),
            "--epochs",
            str(args.epochs),
            "--batch-prompts",
            str(args.batch_prompts),
            "--group-size",
            str(args.group_size),
            "--train-limit",
            str(args.train_limit),
            "--eval-limit",
            str(args.eval_limit),
            "--max-prompt-tokens",
            str(args.max_prompt_tokens),
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--checkpoint-every-steps",
            str(args.checkpoint_every_steps),
            "--heartbeat-every-steps",
            str(args.heartbeat_every_steps),
            "--dataset-name",
            args.dataset_name,
            "--dataset-config",
            args.dataset_config,
            "--train-split",
            args.train_split,
            "--eval-split",
            args.eval_split,
            "--question-field",
            args.question_field,
            "--answer-field",
            args.answer_field,
            "--resume",
        ]
        print(f"[{idx}/{len(lrs)}] RUN {run_id}")
        proc = subprocess.run(cmd, check=False)
        if proc.returncode != 0:
            _append_row(
                summary_csv,
                {
                    "run_id": run_id,
                    "budget": str(args.budget),
                    "lr": str(lr),
                    "seed": str(args.seed),
                    "status": "failed",
                    "pass_at_1": "",
                },
            )
            _write_best(best_json, summary_csv)
            return proc.returncode

        score = _read_pass_at_1(root, run_id)
        _append_row(
            summary_csv,
            {
                "run_id": run_id,
                "budget": str(args.budget),
                "lr": str(lr),
                "seed": str(args.seed),
                "status": "ok",
                "pass_at_1": "" if score is None else f"{score:.6f}",
            },
        )
        _write_best(best_json, summary_csv)

    print(f"Done. Summary: {summary_csv}")
    if best_json.exists():
        print(f"Best: {best_json}")
        print(best_json.read_text(encoding='utf-8'))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
