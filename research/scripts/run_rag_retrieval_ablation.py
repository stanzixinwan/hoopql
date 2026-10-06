"""
Run retrieval-only + end-to-end NBA (test split) for dense / bm25 / hybrid × k ∈ {1,3,5}.

Outputs:
  - eval/rag_retrieval_summary.csv
  - eval/rag_e2e_summary.csv
  - eval/rag_recall_bins_summary.csv
  - eval/*.json (via src.evaluate)
  - internal_docs/rag_failure_cases.md (10 failure examples from one config)

Usage:
  python scripts/run_rag_retrieval_ablation.py
  python scripts/run_rag_retrieval_ablation.py --smoke   # max-examples 5, skip some combos optional
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_utils import NBA_SPLIT_PATH, load_nba_split_ids  # noqa: E402
from src.rag import evaluate_retrieval  # noqa: E402

BACKENDS = ["dense", "bm25", "hybrid"]
KS = [1, 3, 5]

DEFAULT_CHECKPOINTS = [
    (
        "models/lora_codet5p-220m_r16_bs4_r16_lr0.0001_s42/final",
        "Salesforce/codet5p-220m",
        "codet5p_spider_bs4",
    ),
    (
        "models/lora_codet5p-220m_r16_nba_nall_s42/final",
        "Salesforce/codet5p-220m",
        "codet5p_nba_nall",
    ),
]


def _test_questions(questions_path: Path, split_path: Path) -> list[dict]:
    _, test_ids = load_nba_split_ids(str(split_path))
    with open(questions_path, encoding="utf-8") as f:
        allq = json.load(f)
    return [allq[i] for i in sorted(test_ids)]


def _mean_recall_from_json(path: Path) -> float:
    with open(path, encoding="utf-8") as f:
        rows = json.load(f)
    vals = [r.get("retrieval_recall") for r in rows]
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else 0.0


def _exec_exact_from_json(path: Path) -> tuple[float, float]:
    with open(path, encoding="utf-8") as f:
        rows = json.load(f)
    n = len(rows) or 1
    ex = sum(bool(r.get("exec_match")) for r in rows) / n
    exa = sum(bool(r.get("exact_match")) for r in rows) / n
    return ex, exa


def _recall_bins(rows: list[dict]) -> list[dict]:
    bins = [
        ("lt_0.4", lambda r: r.get("retrieval_recall") is not None and r["retrieval_recall"] < 0.4),
        ("0.4_0.7", lambda r: r.get("retrieval_recall") is not None and 0.4 <= r["retrieval_recall"] <= 0.7),
        ("gt_0.7", lambda r: r.get("retrieval_recall") is not None and r["retrieval_recall"] > 0.7),
    ]
    out = []
    for name, pred in bins:
        sub = [r for r in rows if pred(r)]
        if not sub:
            out.append({"bin": name, "n": 0, "exec_rate": None})
            continue
        ex = sum(bool(r.get("exec_match")) for r in sub) / len(sub)
        out.append({"bin": name, "n": len(sub), "exec_rate": round(ex, 4)})
    return out


def _checkpoint_tag_from_run_name(run_name: str) -> str:
    if "nba_nall" in run_name:
        return "codet5p_nba_nall"
    return "codet5p_spider_bs4"


def aggregate_from_eval_json(eval_dir: Path) -> None:
    """Build rag_e2e_summary.csv / rag_recall_bins_summary.md from existing eval JSON."""
    pattern = re.compile(r"(.+)_nba_rag_(\w+)_k(\d+)_test\.json$")
    e2e_rows = []
    bins_rows = []
    for path in sorted(eval_dir.glob("*_nba_rag_*_test.json")):
        m = pattern.match(path.name)
        if not m:
            continue
        run_name, backend, k = m.group(1), m.group(2), int(m.group(3))
        tag = _checkpoint_tag_from_run_name(run_name)
        mr = _mean_recall_from_json(path)
        ex_acc, exact_acc = _exec_exact_from_json(path)
        e2e_rows.append(
            {
                "checkpoint_tag": tag,
                "run_name": run_name,
                "backend": backend,
                "k": k,
                "mean_retrieval_recall": round(mr, 4),
                "exec_acc": round(ex_acc, 4),
                "exact_acc": round(exact_acc, 4),
                "json": path.name,
            }
        )
        with open(path, encoding="utf-8") as f:
            jrows = json.load(f)
        for b in _recall_bins(jrows):
            bins_rows.append(
                {"checkpoint_tag": tag, "backend": backend, "k": k, **b}
            )

    e2e_path = eval_dir / "rag_e2e_summary.csv"
    with open(e2e_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "checkpoint_tag",
                "run_name",
                "backend",
                "k",
                "mean_retrieval_recall",
                "exec_acc",
                "exact_acc",
                "json",
            ],
        )
        w.writeheader()
        w.writerows(sorted(e2e_rows, key=lambda r: (r["checkpoint_tag"], r["backend"], r["k"])))
    print(f"Saved {e2e_path} ({len(e2e_rows)} rows)")

    bins_path = eval_dir / "rag_recall_bins_summary.csv"
    with open(bins_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "checkpoint_tag",
                "backend",
                "k",
                "bin",
                "n",
                "exec_rate",
            ],
        )
        w.writeheader()
        w.writerows(bins_rows)
    print(f"Saved {bins_path}")

    primary = eval_dir / "lora_codet5p-220m_r16_nba_nall_s42_nba_rag_dense_k3_test.json"
    if primary.exists():
        with open(primary, encoding="utf-8") as f:
            pr = json.load(f)
        _write_failure_cases_md(ROOT / "internal_docs" / "rag_failure_cases.md", pr)


def _write_failure_cases_md(
    path: Path,
    rows: list[dict],
    n: int = 10,
) -> None:
    failures = [r for r in rows if not r.get("exec_match")]
    miss = [r for r in failures if not r.get("perfect_retrieval")]
    gen = [r for r in failures if r.get("perfect_retrieval")]
    picked = (miss[: max(1, n // 2)] + gen[: max(1, n - n // 2)])[:n]
    lines = [
        "# RAG failure cases (sample)",
        "",
        "Rule: **retrieval_miss** = not all gold tables in top-k (`perfect_retrieval` false). "
        "**gen_error** = gold tables retrieved but execution still wrong.",
        "",
        "| # | category | question | tables_used | retrieved | recall | pred (trunc) |",
        "|---|----------|----------|-------------|-----------|--------|--------------|",
    ]
    for i, r in enumerate(picked, 1):
        cat = "retrieval_miss" if not r.get("perfect_retrieval") else "gen_error"
        q = (r.get("question") or "")[:80].replace("|", "\\|")
        tu = str(r.get("tables_used") or "")[:60].replace("|", "\\|")
        rt = str(r.get("retrieved_tables") or "")[:60].replace("|", "\\|")
        rec = r.get("retrieval_recall", "")
        pred = (r.get("pred") or "")[:100].replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {i} | {cat} | {q} | {tu} | {rt} | {rec} | {pred} |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Small e2e only (5 examples)")
    parser.add_argument("--skip-e2e", action="store_true", help="Only retrieval summary")
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Only rebuild CSVs + failure cases from eval/*_nba_rag_*_test.json",
    )
    parser.add_argument("--questions", default="data/nba/nba_questions.json")
    parser.add_argument("--split-path", default=str(NBA_SPLIT_PATH))
    parser.add_argument("--eval-dir", default="eval")
    args = parser.parse_args()

    eval_dir = ROOT / args.eval_dir
    if args.aggregate_only:
        aggregate_from_eval_json(eval_dir)
        return

    questions_path = ROOT / args.questions
    split_path = ROOT / args.split_path
    eval_dir.mkdir(parents=True, exist_ok=True)

    test_q = _test_questions(questions_path, split_path)

    retr_path = eval_dir / "rag_retrieval_summary.csv"
    e2e_path = eval_dir / "rag_e2e_summary.csv"
    bins_path = eval_dir / "rag_recall_bins_summary.csv"

    retr_rows = []
    for backend in BACKENDS:
        for k in KS:
            stats = evaluate_retrieval(
                top_k=k,
                backend=backend,
                questions=test_q,
            )
            retr_rows.append(
                {
                    "backend": backend,
                    "k": k,
                    "mean_recall": round(stats["overall_recall"], 4),
                    "perfect_rate": round(stats["perfect_rate"], 4),
                    "n": stats["n"],
                }
            )

    with open(retr_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["backend", "k", "mean_recall", "perfect_rate", "n"],
        )
        w.writeheader()
        w.writerows(retr_rows)
    print(f"Saved {retr_path}")

    e2e_rows = []
    bins_rows = []

    if args.skip_e2e:
        e2e_path.write_text("", encoding="utf-8")
    else:
        max_ex = "5" if args.smoke else None
        for ckpt, base_model, tag in DEFAULT_CHECKPOINTS:
            ckpt_p = ROOT / ckpt
            if not ckpt_p.is_dir():
                print(f"Skip missing checkpoint: {ckpt_p}")
                continue
            run_name = ckpt_p.parent.name
            for backend in BACKENDS:
                for k in KS:
                    cmd = [
                        sys.executable,
                        "-m",
                        "src.evaluate",
                        "--checkpoint",
                        str(ckpt_p),
                        "--base-model",
                        base_model,
                        "--eval",
                        "nba",
                        "--use-rag",
                        "--rag-backend",
                        backend,
                        "--top-k",
                        str(k),
                        "--split",
                        "test",
                        "--split-path",
                        str(split_path),
                        "--output-dir",
                        str(eval_dir),
                    ]
                    if max_ex:
                        cmd.extend(["--max-examples", max_ex])
                    print("Running:", " ".join(cmd))
                    subprocess.run(cmd, cwd=str(ROOT), check=True)

                    out_json = eval_dir / f"{run_name}_nba_rag_{backend}_k{k}_test.json"
                    if not out_json.exists():
                        print(f"Missing output {out_json}")
                        continue
                    mr = _mean_recall_from_json(out_json)
                    ex_acc, exact_acc = _exec_exact_from_json(out_json)
                    e2e_rows.append(
                        {
                            "checkpoint_tag": tag,
                            "run_name": run_name,
                            "backend": backend,
                            "k": k,
                            "mean_retrieval_recall": round(mr, 4),
                            "exec_acc": round(ex_acc, 4),
                            "exact_acc": round(exact_acc, 4),
                            "json": out_json.name,
                        }
                    )
                    with open(out_json, encoding="utf-8") as f:
                        jrows = json.load(f)
                    for b in _recall_bins(jrows):
                        bins_rows.append(
                            {
                                "checkpoint_tag": tag,
                                "backend": backend,
                                "k": k,
                                **b,
                            }
                        )

        with open(e2e_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "checkpoint_tag",
                    "run_name",
                    "backend",
                    "k",
                    "mean_retrieval_recall",
                    "exec_acc",
                    "exact_acc",
                    "json",
                ],
            )
            w.writeheader()
            w.writerows(e2e_rows)
        print(f"Saved {e2e_path}")

        with open(bins_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "checkpoint_tag",
                    "backend",
                    "k",
                    "bin",
                    "n",
                    "exec_rate",
                ],
            )
            w.writeheader()
            w.writerows(bins_rows)
        print(f"Saved {bins_path}")

        # Failure cases: prefer nba_nall, k=3, dense if present
        primary = eval_dir / "lora_codet5p-220m_r16_nba_nall_s42_nba_rag_dense_k3_test.json"
        if not primary.exists() and e2e_rows:
            primary = eval_dir / e2e_rows[-1]["json"]
        if primary.exists():
            with open(primary, encoding="utf-8") as f:
                pr = json.load(f)
            _write_failure_cases_md(ROOT / "internal_docs" / "rag_failure_cases.md", pr)


if __name__ == "__main__":
    main()
