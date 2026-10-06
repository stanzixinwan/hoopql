"""
run_experiment_matrix.py — Launch reproducible training/eval sweeps.

This helper does not change model logic. It standardizes command templates for:
  - LoRA rank sweep (r=4/8/16/32)
  - Method comparison (full/lora/qlora)
  - NBA adaptation points (n=10/20/70/all)

Usage examples:
    python -m src.run_experiment_matrix --stage rank_sweep --dry-run
    python -m src.run_experiment_matrix --stage method_compare --run
"""

import argparse
import shlex
import subprocess
import sys


def _print_and_run(commands: list[str], run: bool) -> None:
    for cmd in commands:
        print(cmd)
        if run:
            argv = shlex.split(cmd, posix=(sys.platform != "win32"))
            completed = subprocess.run(argv, check=False)
            if completed.returncode != 0:
                raise RuntimeError(f"Command failed ({completed.returncode}): {cmd}")


def build_rank_sweep(model: str, epochs: int, lr: float, seed: int) -> list[str]:
    return [
        (
            f"{sys.executable} -m src.train --method lora --model {model} "
            f"--rank {rank} --epochs {epochs} --lr {lr} --seed {seed}"
        )
        for rank in (4, 8, 16, 32)
    ]


def build_method_compare(model: str, epochs: int, lr: float, seed: int) -> list[str]:
    methods = [
        ("full", ""),
        ("lora", "--rank 16"),
        ("qlora", "--rank 16"),
    ]
    return [
        (
            f"{sys.executable} -m src.train --method {method} --model {model} "
            f"{extra} --epochs {epochs} --lr {lr} --seed {seed}"
        ).strip()
        for method, extra in methods
    ]


def build_adaptation_points(base_checkpoint: str, base_model: str, seed: int) -> list[str]:
    points = ["10", "20", "70", "all"]
    return [
        (
            f"{sys.executable} -m src.train_nba --base-checkpoint {base_checkpoint} "
            f"--base-model {base_model} --n-train {n} --seed {seed}"
        )
        for n in points
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["rank_sweep", "method_compare", "adaptation"], required=True)
    parser.add_argument("--model", default="t5-base")
    parser.add_argument("--base-checkpoint", default="models/lora_t5-base_r16/final")
    parser.add_argument("--base-model", default="t5-base")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.run and args.dry_run:
        raise ValueError("Use either --run or --dry-run, not both.")
    should_run = args.run and not args.dry_run

    if args.stage == "rank_sweep":
        commands = build_rank_sweep(args.model, args.epochs, args.lr, args.seed)
    elif args.stage == "method_compare":
        commands = build_method_compare(args.model, args.epochs, args.lr, args.seed)
    else:
        commands = build_adaptation_points(args.base_checkpoint, args.base_model, args.seed)

    _print_and_run(commands, run=should_run)


if __name__ == "__main__":
    main()
