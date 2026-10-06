"""
evaluate.py — Evaluate a fine-tuned checkpoint on NBA and/or Spider.

Supports any encoder-decoder model (t5, flan-t5, codet5) for both full
fine-tuning and PEFT/LoRA checkpoints. Three NBA schema-input modes:
  - default:        full NBA schema in prompt
  - --oracle-tables: only schemas of gold tables (perfect retrieval upper bound)
  - --use-rag:       schema retrieval (top-k tables): --rag-backend dense|bm25|hybrid

NBA evaluation defaults to `--split test` (the 50-example hold-out from
train_nba.py), so few-shot training samples never leak into the report.
Pass `--split all` to reproduce the legacy 200-question evaluation.

Usage:
    # T5-base LoRA, RAG retrieval, on the held-out test split
    python -m src.evaluate --checkpoint models/lora_t5-base_r16/final \
        --eval nba --use-rag --top-k 3

    # Flan-T5-large LoRA — must pass --base-model so the adapter loads
    python -m src.evaluate --checkpoint models/lora_flan-t5-large_r16/final \
        --base-model google/flan-t5-large --eval nba --oracle-tables

    # Reproduce the old 200-question oracle eval (legacy)
    python -m src.evaluate --checkpoint models/full_flan-t5-base/final \
        --eval nba --oracle-tables --split all
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm

from src.data_utils import (
    NBA_SPLIT_PATH, load_nba_dataset, load_nba_split_ids, load_spider_splits,
)
from src.gpu_env import require_cuda
from src.prompt_baseline import (
    execution_accuracy,
    exact_match,
    generate_sql,
    load_tokenizer_with_fallback,
)


def load_checkpoint(checkpoint_path: str, base_model: str = "t5-base"):
    """
    Load a checkpoint. Auto-detects PEFT vs full fine-tune.

    For PEFT: base_model must match the model the adapter was trained on
    (e.g. google/flan-t5-large for a flan-t5-large LoRA adapter).
    """
    ckpt = Path(checkpoint_path)
    is_peft = (ckpt / "adapter_config.json").exists()

    if is_peft:
        from peft import PeftModel
        print(f"Loading PEFT adapter from {checkpoint_path} on top of {base_model}")
        tokenizer = load_tokenizer_with_fallback(checkpoint_path, base_model)
        model = AutoModelForSeq2SeqLM.from_pretrained(base_model)
        model = PeftModel.from_pretrained(model, checkpoint_path)
        model = model.merge_and_unload()
    else:
        print(f"Loading full model from {checkpoint_path}")
        tokenizer = load_tokenizer_with_fallback(checkpoint_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_path)

    return model, tokenizer


def evaluate(model, tokenizer, examples, db_path, device, eval_name: str):
    results = []
    by_diff = defaultdict(lambda: {"total": 0, "exec": 0, "exact": 0})

    for ex in tqdm(examples, desc=f"Eval {eval_name}"):
        pred_sql = generate_sql(model, tokenizer, ex["input_text"], device)
        gold_sql = ex.get("gold_sql", ex["target_text"])

        ex_exec = execution_accuracy(pred_sql, gold_sql, db_path) if db_path else False
        ex_exact = exact_match(pred_sql, gold_sql)

        diff = ex.get("difficulty", "unknown")
        by_diff[diff]["total"] += 1
        by_diff[diff]["exec"] += int(ex_exec)
        by_diff[diff]["exact"] += int(ex_exact)

        row = {
            "question": ex.get("question", ex["input_text"][:100]),
            "gold": gold_sql,
            "pred": pred_sql,
            "exec_match": ex_exec,
            "exact_match": ex_exact,
            "difficulty": diff,
            "retrieved_tables": ex.get("retrieved_tables"),
            "tables_used": ex.get("tables_used"),
            "retrieval_recall": ex.get("retrieval_recall"),
            "perfect_retrieval": ex.get("perfect_retrieval"),
        }
        results.append({k: v for k, v in row.items() if v is not None})

    n = len(results)
    total_exec = sum(r["exec_match"] for r in results)
    total_exact = sum(r["exact_match"] for r in results)

    print(f"\n=== {eval_name} ({n} examples) ===")
    print(f"  Execution accuracy: {total_exec}/{n} = {total_exec/n:.1%}")
    print(f"  Exact match:        {total_exact}/{n} = {total_exact/n:.1%}")
    print(f"\n  By difficulty:")
    for diff in ["easy", "medium", "hard", "extra_hard", "unknown"]:
        if diff in by_diff and by_diff[diff]["total"] > 0:
            d = by_diff[diff]
            print(f"    {diff:11s}: exec {d['exec']:3d}/{d['total']:3d} ({d['exec']/d['total']:5.1%}) | "
                  f"exact {d['exact']:3d}/{d['total']:3d} ({d['exact']/d['total']:5.1%})")

    return results


def load_nba_examples(args):
    """Pick the NBA loader based on flags. Mutually exclusive: oracle vs RAG."""
    if args.use_rag:
        from src.rag import load_nba_dataset_with_rag
        print(
            f"NBA mode: RAG backend={args.rag_backend!r} top-{args.top_k}"
        )
        return load_nba_dataset_with_rag(
            args.nba_questions,
            args.nba_db,
            backend=args.rag_backend,
            top_k=args.top_k,
        )
    elif args.oracle_tables:
        print("NBA mode: oracle tables (gold tables_used only)")
        return load_nba_dataset(args.nba_questions, args.nba_db,
                                use_oracle_tables=True)
    else:
        print("NBA mode: full schema")
        return load_nba_dataset(args.nba_questions, args.nba_db)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--base-model", default="t5-base",
                        help="For PEFT, the base model the adapter was trained on. "
                             "Examples: t5-base, google/flan-t5-base, google/flan-t5-large")
    parser.add_argument("--eval", choices=["nba", "spider", "both"], required=True)
    parser.add_argument("--nba-questions", default="data/nba/nba_questions.json")
    parser.add_argument("--nba-db", default="data/raw/nba.sqlite")
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--output-dir", default="eval")
    parser.add_argument("--oracle-tables", action="store_true",
                        help="Restrict NBA schema to gold tables (perfect retrieval)")
    parser.add_argument("--use-rag", action="store_true",
                        help="Use RAG retrieval instead of full/oracle schema")
    parser.add_argument(
        "--rag-backend",
        default="dense",
        choices=["dense", "bm25", "hybrid"],
        help="Retriever when --use-rag (build indices via python -m src.rag --build [--build-bm25])",
    )
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--split", choices=["test", "train", "all"], default="test",
                        help="Which NBA subset to evaluate on. Default 'test' "
                             "uses the 50-example hold-out from train_nba.py "
                             "(no train-set leakage).")
    parser.add_argument("--split-path", default=NBA_SPLIT_PATH,
                        help="Path to the NBA split JSON written by train_nba.py")
    args = parser.parse_args()

    if args.use_rag and args.oracle_tables:
        parser.error("--use-rag and --oracle-tables are mutually exclusive")

    require_cuda()
    device = "cuda"
    print(f"Device: {device}")

    model, tokenizer = load_checkpoint(args.checkpoint, args.base_model)
    model.to(device).eval()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    run_name = Path(args.checkpoint).parent.name

    # Filename suffix to distinguish runs
    if args.use_rag:
        suffix = f"_rag_{args.rag_backend}_k{args.top_k}"
    elif args.oracle_tables:
        suffix = "_oracle"
    else:
        suffix = "_full"

    # Append split tag so test/train results don't overwrite legacy 200-q files.
    split_suffix = "" if args.split == "all" else f"_{args.split}"

    if args.eval in ("nba", "both"):
        nba = load_nba_examples(args)
        if args.split != "all":
            train_ids, test_ids = load_nba_split_ids(args.split_path)
            keep = test_ids if args.split == "test" else train_ids
            nba = [ex for i, ex in enumerate(nba) if i in keep]
        print(f"NBA split: {args.split} ({len(nba)} examples)")
        if args.max_examples:
            nba = nba[:args.max_examples]
        results = evaluate(model, tokenizer, nba, args.nba_db, device,
                           f"NBA / {run_name}{suffix}{split_suffix}")
        out = f"{args.output_dir}/{run_name}_nba{suffix}{split_suffix}.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"  Saved -> {out}")

    if args.eval in ("spider", "both"):
        _, dev = load_spider_splits(max_examples=args.max_examples)
        # Spider exec acc would need per-db SQLite paths; using exact match only.
        results = evaluate(model, tokenizer, dev, None, device,
                           f"Spider / {run_name}")
        out = f"{args.output_dir}/{run_name}_spider.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"  Saved -> {out}")


if __name__ == "__main__":
    main()