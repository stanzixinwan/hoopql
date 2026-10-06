"""
build_results_report.py — Build comparable result tables from eval JSON files.

Outputs:
  - eval/results_summary.csv
  - eval/results_summary.md
  - eval/fewshot_curve.csv
  - eval/rag_ablation.csv
  - eval/spider_summary.csv
  - eval/spider_summary.md
"""

import argparse
import csv
import json
import random
import re
from pathlib import Path


def _quantile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    if q <= 0:
        return sorted_values[0]
    if q >= 1:
        return sorted_values[-1]
    pos = q * (len(sorted_values) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = pos - lo
    return sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac


def _bootstrap_ci(
    hits: list[int],
    n_bootstrap: int,
    ci_level: float,
    rng: random.Random,
) -> tuple[float, float]:
    n = len(hits)
    if n == 0:
        return 0.0, 0.0

    means = []
    for _ in range(max(n_bootstrap, 1)):
        sample_sum = 0
        for _ in range(n):
            sample_sum += hits[rng.randrange(n)]
        means.append(sample_sum / n)

    means.sort()
    alpha = 1.0 - ci_level
    low_q = alpha / 2
    high_q = 1.0 - alpha / 2
    return _quantile(means, low_q), _quantile(means, high_q)


def _load_scores(
    path: Path,
    n_bootstrap: int,
    ci_level: float,
    rng: random.Random,
) -> tuple[float, float, int, float, float, float, float]:
    with open(path, "r", encoding="utf-8") as f:
        items = json.load(f)

    exec_hits = [1 if bool(r.get("exec_match")) else 0 for r in items]
    exact_hits = [1 if bool(r.get("exact_match")) else 0 for r in items]
    n = len(items)

    if n == 0:
        return 0.0, 0.0, 0, 0.0, 0.0, 0.0, 0.0

    exec_acc = sum(exec_hits) / n
    exact_acc = sum(exact_hits) / n
    exec_low, exec_high = _bootstrap_ci(exec_hits, n_bootstrap, ci_level, rng)
    exact_low, exact_high = _bootstrap_ci(exact_hits, n_bootstrap, ci_level, rng)
    return exec_acc, exact_acc, n, exec_low, exec_high, exact_low, exact_high


def _label_from_filename(name: str) -> dict:
    return {
        "run": name.replace(".json", ""),
        "model": "codet5p-220m" if "codet5p-220m" in name else "t5-base" if "t5-base" in name else "unknown",
        "method": (
            "full" if name.startswith("full_")
            else "lora" if name.startswith("lora_")
            else "qlora" if name.startswith("qlora_")
            else "baseline"
        ),
        "mode": "rag" if "_rag_" in name else "oracle" if "_oracle_" in name else "full_schema" if "_full_" in name else "other",
        "split": "test" if name.endswith("_test.json") else "all",
    }


def _parse_n_train(name: str) -> str:
    m = re.search(r"_nba_n(10|20|70|all)(?:_s\d+)?_", name)
    if m:
        return m.group(1)
    if "_nba_oracle_test" in name:
        return "0"
    return ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dir", default="eval")
    parser.add_argument("--glob", default="*_test.json")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument("--ci-level", type=float, default=0.95)
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    files = sorted(eval_dir.glob(args.glob))
    rng = random.Random(args.bootstrap_seed)
    rows = []
    for path in files:
        exec_acc, exact_acc, n, exec_low, exec_high, exact_low, exact_high = _load_scores(
            path,
            n_bootstrap=args.bootstrap_samples,
            ci_level=args.ci_level,
            rng=rng,
        )
        meta = _label_from_filename(path.name)
        n_train = _parse_n_train(path.name)
        rows.append({
            **meta,
            "file": path.name,
            "n_examples": n,
            "n_train": n_train,
            "exec_acc": round(exec_acc, 4),
            "exact_acc": round(exact_acc, 4),
            "exec_ci_low": round(exec_low, 4),
            "exec_ci_high": round(exec_high, 4),
            "exact_ci_low": round(exact_low, 4),
            "exact_ci_high": round(exact_high, 4),
        })

    out_csv = eval_dir / "results_summary.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "file",
                "run",
                "model",
                "method",
                "mode",
                "split",
                "n_examples",
                "n_train",
                "exec_acc",
                "exec_ci_low",
                "exec_ci_high",
                "exact_acc",
                "exact_ci_low",
                "exact_ci_high",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    out_md = eval_dir / "results_summary.md"
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("| file | model | method | mode | n_train | exec_acc | exec_ci | exact_acc | exact_ci |\n")
        f.write("|---|---|---|---|---:|---:|---|---:|---|\n")
        for r in rows:
            f.write(
                f"| {r['file']} | {r['model']} | {r['method']} | {r['mode']} | {r['n_train'] or '-'} | "
                f"{r['exec_acc']:.4f} | [{r['exec_ci_low']:.4f}, {r['exec_ci_high']:.4f}] | "
                f"{r['exact_acc']:.4f} | [{r['exact_ci_low']:.4f}, {r['exact_ci_high']:.4f}] |\n"
            )

    fewshot = [r for r in rows if r["n_train"] in {"0", "10", "20", "70", "all"} and r["mode"] == "oracle"]
    fewshot = sorted(fewshot, key=lambda x: {"0": 0, "10": 1, "20": 2, "70": 3, "all": 4}[x["n_train"]])
    with open(eval_dir / "fewshot_curve.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "file",
                "model",
                "n_train",
                "exec_acc",
                "exec_ci_low",
                "exec_ci_high",
                "exact_acc",
                "exact_ci_low",
                "exact_ci_high",
            ],
        )
        writer.writeheader()
        for r in fewshot:
            writer.writerow({
                k: r[k]
                for k in [
                    "file",
                    "model",
                    "n_train",
                    "exec_acc",
                    "exec_ci_low",
                    "exec_ci_high",
                    "exact_acc",
                    "exact_ci_low",
                    "exact_ci_high",
                ]
            })

    rag = [r for r in rows if r["mode"] == "rag"]
    with open(eval_dir / "rag_ablation.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "file",
                "model",
                "exec_acc",
                "exec_ci_low",
                "exec_ci_high",
                "exact_acc",
                "exact_ci_low",
                "exact_ci_high",
            ],
        )
        writer.writeheader()
        for r in rag:
            writer.writerow({
                k: r[k]
                for k in [
                    "file",
                    "model",
                    "exec_acc",
                    "exec_ci_low",
                    "exec_ci_high",
                    "exact_acc",
                    "exact_ci_low",
                    "exact_ci_high",
                ]
            })

    spider_files = sorted(eval_dir.glob("*_spider.json"))
    spider_rows = []
    for path in spider_files:
        exec_acc, exact_acc, n, exec_low, exec_high, exact_low, exact_high = _load_scores(
            path,
            n_bootstrap=args.bootstrap_samples,
            ci_level=args.ci_level,
            rng=rng,
        )
        lower = path.name.lower()
        if "codet5p-220m" in lower:
            model = "codet5p-220m"
        elif "flan-t5-base" in lower:
            model = "flan-t5-base"
        elif "t5-base" in lower:
            model = "t5-base"
        else:
            model = "unknown"
        spider_rows.append({
            "file": path.name,
            "run": path.stem,
            "model": model,
            "n_examples": n,
            "exec_acc": round(exec_acc, 4),
            "exact_acc": round(exact_acc, 4),
            "exec_ci_low": round(exec_low, 4),
            "exec_ci_high": round(exec_high, 4),
            "exact_ci_low": round(exact_low, 4),
            "exact_ci_high": round(exact_high, 4),
        })

    spider_rows = sorted(spider_rows, key=lambda r: r["exact_acc"], reverse=True)
    spider_csv = eval_dir / "spider_summary.csv"
    with open(spider_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "file",
                "run",
                "model",
                "n_examples",
                "exec_acc",
                "exec_ci_low",
                "exec_ci_high",
                "exact_acc",
                "exact_ci_low",
                "exact_ci_high",
            ],
        )
        writer.writeheader()
        writer.writerows(spider_rows)

    spider_md = eval_dir / "spider_summary.md"
    with open(spider_md, "w", encoding="utf-8") as f:
        f.write("| file | model | n_examples | exec_acc | exec_ci | exact_acc | exact_ci |\n")
        f.write("|---|---|---:|---:|---|---:|---|\n")
        for r in spider_rows:
            f.write(
                f"| {r['file']} | {r['model']} | {r['n_examples']} | "
                f"{r['exec_acc']:.4f} | [{r['exec_ci_low']:.4f}, {r['exec_ci_high']:.4f}] | "
                f"{r['exact_acc']:.4f} | [{r['exact_ci_low']:.4f}, {r['exact_ci_high']:.4f}] |\n"
            )

    print(f"Saved -> {out_csv}")
    print(f"Saved -> {out_md}")
    print(f"Saved -> {eval_dir / 'fewshot_curve.csv'}")
    print(f"Saved -> {eval_dir / 'rag_ablation.csv'}")
    print(f"Saved -> {spider_csv}")
    print(f"Saved -> {spider_md}")


if __name__ == "__main__":
    main()
