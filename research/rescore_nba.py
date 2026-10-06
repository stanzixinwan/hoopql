"""
rescore_nba.py — Re-score existing NBA evaluation JSON files on the held-out
test split, without re-running model generation.

Each file in eval/<run>_nba*.json is a list of 200 result dicts whose order
matches data/nba/nba_questions.json (indices 0..199). train_nba.py created a
deterministic 150/50 split in data/nba/nba_split.json. This script:

  1. Loads the test_ids from the split.
  2. For each input eval JSON, sanity-checks length == 200 and that a few
     sampled questions match nba_questions.json at the same index.
  3. Slices the result list down to the 50 test indices.
  4. Recomputes overall execution / exact-match accuracy and a per-difficulty
     breakdown — same format as evaluate.py's printout.
  5. With --save, writes the test subset back to eval/<basename>_test.json
     (preserving the original record schema) for downstream analysis.

Usage:
    python -m src.rescore_nba eval/lora_t5-base_r16_nba_oracle.json \
                              eval/lora_t5-base_r16_nba_n20_v2_nba_oracle.json \
                              eval/lora_t5-base_r16_nba_n70_v2_nba_oracle.json \
                              eval/lora_t5-base_r16_nba_nall_v2_nba_oracle.json \
                              --save
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

try:
    from src.data_utils import NBA_SPLIT_PATH, load_nba_split_ids
except ModuleNotFoundError:
    from data_utils import NBA_SPLIT_PATH, load_nba_split_ids


DIFFICULTY_ORDER = ["easy", "medium", "hard", "extra_hard", "unknown"]


def _load_questions(questions_path: str) -> list[dict]:
    with open(questions_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _sanity_check(results: list[dict], questions: list[dict],
                   path: Path, n_samples: int = 5) -> bool:
    """Verify that result[i]['question'] matches questions[i]['question'] for
    a few sampled indices. Returns True if everything aligns."""
    if len(results) != len(questions):
        print(f"  [SKIP] {path.name}: {len(results)} results vs "
              f"{len(questions)} questions — not the canonical 200-q file.")
        return False

    sample_indices = list(range(0, len(results), max(1, len(results) // n_samples)))[:n_samples]
    mismatches = []
    for i in sample_indices:
        r_q = (results[i].get("question") or "").strip()
        g_q = (questions[i].get("question") or "").strip()
        if r_q and g_q and r_q != g_q:
            mismatches.append((i, r_q[:60], g_q[:60]))

    if mismatches:
        print(f"  [WARN] {path.name}: question text mismatch at sampled indices:")
        for i, r, g in mismatches:
            print(f"          idx {i}: result='{r}' vs gold='{g}'")
        print("          Refusing to rescore — order may not align with nba_questions.json.")
        return False
    return True


def _summarize(results: list[dict], label: str) -> None:
    n = len(results)
    if n == 0:
        print(f"\n=== {label} (0 examples) ===\n  (empty)")
        return

    by_diff = defaultdict(lambda: {"total": 0, "exec": 0, "exact": 0})
    total_exec = 0
    total_exact = 0
    for r in results:
        ex_exec = bool(r.get("exec_match"))
        ex_exact = bool(r.get("exact_match"))
        total_exec += int(ex_exec)
        total_exact += int(ex_exact)
        diff = r.get("difficulty", "unknown") or "unknown"
        by_diff[diff]["total"] += 1
        by_diff[diff]["exec"] += int(ex_exec)
        by_diff[diff]["exact"] += int(ex_exact)

    print(f"\n=== {label} ({n} examples) ===")
    print(f"  Execution accuracy: {total_exec}/{n} = {total_exec/n:.1%}")
    print(f"  Exact match:        {total_exact}/{n} = {total_exact/n:.1%}")
    print(f"\n  By difficulty:")
    for diff in DIFFICULTY_ORDER:
        if diff in by_diff and by_diff[diff]["total"] > 0:
            d = by_diff[diff]
            print(f"    {diff:11s}: exec {d['exec']:3d}/{d['total']:3d} ({d['exec']/d['total']:5.1%}) | "
                  f"exact {d['exact']:3d}/{d['total']:3d} ({d['exact']/d['total']:5.1%})")


def rescore_file(path: Path, test_ids: set[int], questions: list[dict],
                 save: bool, output_dir: Path) -> None:
    with open(path, "r", encoding="utf-8") as f:
        results = json.load(f)

    if not isinstance(results, list):
        print(f"  [SKIP] {path.name}: not a list of result records.")
        return

    if not _sanity_check(results, questions, path):
        return

    test_subset = [r for i, r in enumerate(results) if i in test_ids]
    label = f"{path.stem} (test {len(test_subset)})"
    _summarize(test_subset, label)

    if save:
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"{path.stem}_test.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(test_subset, f, indent=2, ensure_ascii=False)
        print(f"  Saved -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("eval_files", nargs="+",
                        help="Existing eval JSON files (200-question NBA results)")
    parser.add_argument("--split-path", default=NBA_SPLIT_PATH,
                        help="Path to the NBA split JSON written by train_nba.py")
    parser.add_argument("--nba-questions", default="data/nba/nba_questions.json",
                        help="Used for sanity-checking that result order matches "
                             "the original question list")
    parser.add_argument("--save", action="store_true",
                        help="Save the test subset for each input as "
                             "<output-dir>/<basename>_test.json")
    parser.add_argument("--output-dir", default="eval",
                        help="Where to write _test.json files when --save is given")
    args = parser.parse_args()

    _, test_ids = load_nba_split_ids(args.split_path)
    print(f"Loaded test split: {len(test_ids)} indices from {args.split_path}")
    questions = _load_questions(args.nba_questions)
    print(f"Loaded {len(questions)} canonical NBA questions from {args.nba_questions}")

    output_dir = Path(args.output_dir)
    for f in args.eval_files:
        path = Path(f)
        if not path.exists():
            print(f"  [SKIP] {path}: not found")
            continue
        rescore_file(path, test_ids, questions, args.save, output_dir)


if __name__ == "__main__":
    main()
