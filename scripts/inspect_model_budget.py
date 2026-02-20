from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tinylora.adapter import TinyLoRAConfig, choose_budget_plan, inject_tinylora


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect TinyLoRA budget fit on a model.")
    parser.add_argument("--model-name", required=True, help="HF model id (e.g. Qwen/Qwen2.5-7B-Instruct)")
    parser.add_argument("--target-params", type=int, default=13)
    parser.add_argument("--frozen-rank", type=int, default=2)
    parser.add_argument("--projection-candidates", default="1,2,4")
    parser.add_argument("--dtype", default="float16")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    try:
        import torch
        from transformers import AutoModelForCausalLM
    except Exception as exc:
        print(f"ERROR: missing dependency: {exc}")
        return 2

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map.get(args.dtype, torch.float16)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=dtype, low_cpu_mem_usage=True)

    cfg0 = TinyLoRAConfig(frozen_rank=args.frozen_rank, projection_dim=1, tie_mode="none", tie_factor=1)
    # Discover target modules by dry injection filter.
    candidates = []
    for name, module in model.named_modules():
        if module.__class__.__name__ == "Linear" and name.split(".")[-1] in cfg0.target_modules:
            candidates.append(name)

    plan = choose_budget_plan(
        num_modules=len(candidates),
        target_params=args.target_params,
        projection_dim_candidates=[int(x) for x in args.projection_candidates.split(",") if x.strip()],
    )
    cfg = TinyLoRAConfig(
        frozen_rank=args.frozen_rank,
        projection_dim=plan["projection_dim"],
        tie_mode="tiled",
        tie_factor=plan["tie_factor"],
        target_modules=cfg0.target_modules,
    )
    summary = inject_tinylora(model, cfg)
    report = {
        "model_name": args.model_name,
        "target_params": args.target_params,
        "candidate_modules": len(candidates),
        "budget_plan": plan,
        "inject_summary": summary,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
