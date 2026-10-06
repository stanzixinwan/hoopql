"""
error_analysis.py — Rule-based taxonomy for SQL generation errors.

Categories:
  - schema_linking: wrong/missing table or column references
  - structural: parse / clause-structure problems
  - value: literal mismatch in conditions
  - aggregation: incorrect aggregate/grouping behavior
"""

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


TABLE_COL_RE = re.compile(r"\bfrom\s+([a-zA-Z_][\w]*)|\bjoin\s+([a-zA-Z_][\w]*)", re.IGNORECASE)
VALUE_RE = re.compile(r"(=|<|>|<=|>=|!=)\s*('[^']*'|\"[^\"]*\"|\d+(\.\d+)?)")
AGG_RE = re.compile(r"\b(count|sum|avg|min|max)\s*\(", re.IGNORECASE)


def _extract_tables(sql: str) -> set[str]:
    tables = set()
    for left, right in TABLE_COL_RE.findall(sql or ""):
        if left:
            tables.add(left.lower())
        if right:
            tables.add(right.lower())
    return tables


def _extract_values(sql: str) -> set[str]:
    values = set()
    for match in VALUE_RE.findall(sql or ""):
        values.add(match[1].strip().lower())
    return values


def classify_error(gold_sql: str, pred_sql: str, exec_match: bool) -> str:
    if exec_match:
        return "correct"
    g = (gold_sql or "").lower()
    p = (pred_sql or "").lower()
    g_tables = _extract_tables(g)
    p_tables = _extract_tables(p)
    if g_tables and (not p_tables or g_tables != p_tables):
        return "schema_linking"
    if AGG_RE.search(g) and not AGG_RE.search(p):
        return "aggregation"
    if _extract_values(g) and _extract_values(g) != _extract_values(p):
        return "value"
    if "select" not in p or ("from" in g and "from" not in p):
        return "structural"
    return "structural"


def analyze_file(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    counts = Counter()
    by_diff = defaultdict(Counter)
    annotated = []
    for row in rows:
        label = classify_error(row.get("gold", ""), row.get("pred", ""), bool(row.get("exec_match")))
        counts[label] += 1
        by_diff[row.get("difficulty", "unknown")][label] += 1
        row_out = dict(row)
        row_out["error_type"] = label
        annotated.append(row_out)
    n = len(rows) or 1
    return {
        "file": str(path),
        "total": len(rows),
        "counts": dict(counts),
        "rates": {k: v / n for k, v in counts.items()},
        "by_difficulty": {k: dict(v) for k, v in by_diff.items()},
        "annotated": annotated,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("eval_files", nargs="+")
    parser.add_argument("--output", default="eval/error_analysis_summary.json")
    parser.add_argument("--save-annotated", action="store_true")
    args = parser.parse_args()

    summaries = []
    for eval_file in args.eval_files:
        summary = analyze_file(Path(eval_file))
        summaries.append({k: v for k, v in summary.items() if k != "annotated"})
        print(f"\n=== {eval_file} ===")
        print(f"Total: {summary['total']}")
        for label, count in sorted(summary["counts"].items(), key=lambda x: (-x[1], x[0])):
            print(f"  {label:14s} {count:4d} ({summary['rates'][label]:.1%})")
        if args.save_annotated:
            out_path = Path(eval_file).with_name(Path(eval_file).stem + "_annotated.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(summary["annotated"], f, indent=2, ensure_ascii=False)
            print(f"  Saved annotated -> {out_path}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2, ensure_ascii=False)
    print(f"\nSaved summary -> {args.output}")


if __name__ == "__main__":
    main()
