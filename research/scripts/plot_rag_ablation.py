"""
Plot RAG ablation summaries from eval/rag_e2e_summary.csv and eval/rag_retrieval_summary.csv.

Outputs (PDF + PNG):
  - reports/figures/rag_recall_vs_exec.pdf
  - reports/figures/rag_backend_bars_k3.pdf

Usage:
  python scripts/plot_rag_ablation.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
E2E_CSV = ROOT / "eval" / "rag_e2e_summary.csv"
RETR_CSV = ROOT / "eval" / "rag_retrieval_summary.csv"
FIG_DIR = ROOT / "reports" / "figures"

BACKENDS = ["dense", "bm25", "hybrid"]
MARKERS = {"dense": "o", "bm25": "s", "hybrid": "^"}
COLORS = {"codet5p_spider_bs4": "#1f77b4", "codet5p_nba_nall": "#ff7f0e"}


def _read_e2e(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _read_retr(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def plot_recall_vs_exec(rows: list[dict], out_base: Path) -> None:
    """One subplot per checkpoint_tag: x = mean retrieval recall, y = exec acc."""
    if not rows:
        print("No e2e rows; skip recall_vs_exec")
        return
    tags = sorted({r["checkpoint_tag"] for r in rows})
    fig, axes = plt.subplots(1, len(tags), figsize=(5 * len(tags), 4), squeeze=False)
    for ax, tag in zip(axes[0], tags):
        sub = [r for r in rows if r["checkpoint_tag"] == tag]
        for b in BACKENDS:
            pts = [r for r in sub if r["backend"] == b]
            if not pts:
                continue
            xs = [float(r["mean_retrieval_recall"]) for r in pts]
            ys = [float(r["exec_acc"]) for r in pts]
            ks = [int(r["k"]) for r in pts]
            ax.scatter(
                xs,
                ys,
                marker=MARKERS.get(b, "o"),
                label=b,
                s=80,
            )
            for x, y, k in zip(xs, ys, ks):
                ax.annotate(f"k={k}", (x, y), textcoords="offset points", xytext=(4, 4), fontsize=7)
        ax.set_xlabel("Mean retrieval recall (test)")
        ax.set_ylabel("Execution accuracy")
        ax.set_title(tag)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(0.0, 0.2)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Retrieval recall vs downstream exec accuracy")
    fig.tight_layout()
    for ext in (".png", ".pdf"):
        p = out_base.with_suffix(ext)
        fig.savefig(p, dpi=150, bbox_inches="tight")
        print(f"Saved {p}")
    plt.close(fig)


def plot_bars_k3(e2e_rows: list[dict], retr_rows: list[dict], out_base: Path) -> None:
    """Grouped bars: x = backend, y = exec acc at k=3; two groups for checkpoint."""
    k_target = 3
    sub = [r for r in e2e_rows if int(r["k"]) == k_target]
    if not sub:
        print("No k=3 e2e rows; skip bar chart")
        return
    tags = sorted({r["checkpoint_tag"] for r in sub})
    x = range(len(BACKENDS))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, tag in enumerate(tags):
        offset = (i - len(tags) / 2 + 0.5) * width
        ys = []
        for b in BACKENDS:
            row = next((r for r in sub if r["backend"] == b and r["checkpoint_tag"] == tag), None)
            ys.append(float(row["exec_acc"]) if row else 0.0)
        ax.bar([xi + offset for xi in x], ys, width, label=tag, color=COLORS.get(tag))
    ax.set_xticks(list(x))
    ax.set_xticklabels(BACKENDS)
    ax.set_ylabel("Execution accuracy")
    ax.set_title(f"RAG backends at k={k_target} (NBA test)")
    ax.legend()
    # Zoom into the low-accuracy range where all bars lie.
    ax.set_ylim(0, 0.2)
    ax.grid(True, axis="y", alpha=0.3)

    # Secondary text: retrieval recall from retr summary
    if retr_rows:
        lines = [f"Retrieval mean recall @ k={k_target}:"]
        for b in BACKENDS:
            r = next(
                (row for row in retr_rows if row["backend"] == b and int(row["k"]) == k_target),
                None,
            )
            if r:
                lines.append(f"  {b}: {float(r['mean_recall']):.3f}")
        # Place recall note in the upper-left empty area to avoid occluding bars.
        ax.text(
            0.02,
            0.98,
            "\n".join(lines),
            transform=ax.transAxes,
            fontsize=7,
            family="monospace",
            va="top",
            ha="left",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=2.0),
        )

    fig.tight_layout()
    for ext in (".png", ".pdf"):
        p = out_base.with_suffix(ext)
        fig.savefig(p, dpi=150, bbox_inches="tight")
        print(f"Saved {p}")
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    e2e = _read_e2e(E2E_CSV)
    retr = _read_retr(RETR_CSV)
    plot_recall_vs_exec(e2e, FIG_DIR / "rag_recall_vs_exec")
    plot_bars_k3(e2e, retr, FIG_DIR / "rag_backend_bars_k3")


if __name__ == "__main__":
    main()
