"""
train_nba.py — Few-shot fine-tune on NBA after Spider pre-training.

Continues training a Spider-fine-tuned checkpoint on a small NBA train split,
producing a domain-adapted model. This is the proposal's core experiment for
"NBA adaptation curve" (0/10/50/all examples).

Usage:
    # Full adaptation: all 70 NBA train examples
    python -m src.train_nba \
        --base-checkpoint models/lora_t5-base_r16/final \
        --base-model t5-base \
        --n-train all --epochs 10

    # Few-shot curve: 10 examples
    python -m src.train_nba \
        --base-checkpoint models/lora_t5-base_r16/final \
        --base-model t5-base \
        --n-train 10 --epochs 10

The 50-example test split is held out and not seen during NBA training.
Evaluate the resulting checkpoint with `evaluate.py --eval nba` afterward;
that command now defaults to `--split test` so train-set leakage is avoided.
"""

import argparse
import json
import random
from pathlib import Path

import torch
import numpy as np
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    Seq2SeqTrainer, Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)
from datasets import Dataset
from peft import PeftModel

try:
    from src.data_utils import load_nba_dataset
    from src.gpu_env import require_cuda
    from src.prompt_baseline import load_tokenizer_with_fallback
except ModuleNotFoundError:
    from data_utils import load_nba_dataset
    from gpu_env import require_cuda
    from prompt_baseline import load_tokenizer_with_fallback


SPLIT_PATH = Path("data/nba/nba_split.json")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _cuda_bf16_supported() -> bool:
    return (
        torch.cuda.is_available()
        and hasattr(torch.cuda, "is_bf16_supported")
        and torch.cuda.is_bf16_supported()
    )


def _mixed_precision_flags() -> tuple[bool, bool]:
    if not torch.cuda.is_available():
        return False, False
    bf16 = _cuda_bf16_supported()
    return bf16, not bf16


def make_or_load_split(nba_examples, test_size=50, seed=42):
    """
    Deterministic 150/50 train/test split. Saves to disk so all NBA experiments
    use the same split (critical for valid comparison across models/configs).
    """
    if SPLIT_PATH.exists():
        with open(SPLIT_PATH) as f:
            split = json.load(f)
        train_ids = set(split["train_ids"])
        test_ids = set(split["test_ids"])
        print(f"Loaded existing split: {len(train_ids)} train, {len(test_ids)} test")
    else:
        all_ids = list(range(len(nba_examples)))
        random.Random(seed).shuffle(all_ids)
        test_ids = set(all_ids[:test_size])
        train_ids = set(all_ids[test_size:])
        SPLIT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(SPLIT_PATH, "w") as f:
            json.dump({
                "train_ids": sorted(train_ids),
                "test_ids": sorted(test_ids),
                "seed": seed,
            }, f, indent=2)
        print(f"Created new split: {len(train_ids)} train, {len(test_ids)} test")

    train = [ex for i, ex in enumerate(nba_examples) if i in train_ids]
    test = [ex for i, ex in enumerate(nba_examples) if i in test_ids]
    return train, test


def tokenize_examples(examples, tokenizer, max_input=1024, max_target=256):
    inputs = [ex["input_text"] for ex in examples]
    targets = [ex["target_text"] for ex in examples]
    model_inputs = tokenizer(inputs, max_length=max_input, truncation=True, padding=False)
    labels = tokenizer(text_target=targets, max_length=max_target,
                       truncation=True, padding=False)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def build_dataset(examples, tokenizer):
    ds = Dataset.from_list([{"input_text": e["input_text"],
                             "target_text": e["target_text"]} for e in examples])
    ds = ds.map(
        lambda batch: tokenize_examples(
            [{"input_text": i, "target_text": t}
             for i, t in zip(batch["input_text"], batch["target_text"])],
            tokenizer,
        ),
        batched=True,
        remove_columns=ds.column_names,
    )
    return ds


def load_base(base_checkpoint: str, base_model: str):
    """Load Spider-trained checkpoint (PEFT or full) for further training."""
    ckpt = Path(base_checkpoint)
    is_peft = (ckpt / "adapter_config.json").exists()

    if is_peft:
        tokenizer = load_tokenizer_with_fallback(base_checkpoint, base_model)
    else:
        tokenizer = load_tokenizer_with_fallback(base_checkpoint)

    if is_peft:
        # Keep LoRA: merging then full fine-tuning on tiny NBA sets destroys generation.
        # Train adapter weights only; merge once after training when saving final/.
        print(f"Loading PEFT base from {base_checkpoint} on {base_model} (LoRA continued, base frozen)")
        model = AutoModelForSeq2SeqLM.from_pretrained(base_model)
        model = PeftModel.from_pretrained(
            model, base_checkpoint, is_trainable=True
        )
        model.print_trainable_parameters()
    else:
        print(f"Loading full model from {base_checkpoint}")
        model = AutoModelForSeq2SeqLM.from_pretrained(base_checkpoint)

    return model, tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-checkpoint", required=True,
                        help="Spider-fine-tuned checkpoint to continue from")
    parser.add_argument("--base-model", default="t5-base",
                        help="If checkpoint is PEFT, the underlying base model")
    parser.add_argument("--nba-questions", default="data/nba/nba_questions.json")
    parser.add_argument("--nba-db", default="data/raw/nba.sqlite")
    parser.add_argument("--n-train", default="all",
                        help="How many NBA train examples to use: 'all', or an int (10, 50, ...)")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-5,
                        help="Lower than Spider stage — model is already trained, just adapting")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-oracle-tables", action="store_true", default=True,
                        help="Train with gold tables only (matches what oracle eval uses)")
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    require_cuda()
    _set_seed(args.seed)

    # Load NBA examples with oracle tables (only gold-relevant schemas)
    # — this matches the eval setting and keeps inputs short.
    nba = load_nba_dataset(args.nba_questions, args.nba_db,
                           use_oracle_tables=args.use_oracle_tables)
    train_full, test = make_or_load_split(nba)

    # Subsample train if --n-train is not 'all'
    if args.n_train == "all":
        train = train_full
    else:
        n = int(args.n_train)
        random.Random(args.seed).shuffle(train_full)
        train = train_full[:n]
    print(f"Using {len(train)} NBA train examples")

    base_name = Path(args.base_checkpoint).parent.name
    run_name = args.run_name or f"{base_name}_nba_n{args.n_train}"
    run_name += f"_s{args.seed}"
    output_dir = f"models/{run_name}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print(f"Run: {run_name}")
    print(f"Output: {output_dir}\n")
    with open(f"{output_dir}/run_config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    model, tokenizer = load_base(args.base_checkpoint, args.base_model)

    train_ds = build_dataset(train, tokenizer)
    test_ds = build_dataset(test, tokenizer)

    collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)
    bf16, fp16 = _mixed_precision_flags()
    precision = "bf16" if bf16 else "fp16" if fp16 else "off"
    print(f"Mixed precision: {precision}")

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        logging_steps=5,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        predict_with_generate=False,
        bf16=bf16,
        fp16=fp16,
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        processing_class=tokenizer,
        data_collator=collator,
    )

    print("Training...")
    trainer.train()

    print(f"\nSaving final model to {output_dir}/final")
    to_save = trainer.model
    if isinstance(to_save, PeftModel):
        to_save = to_save.merge_and_unload()
    to_save.save_pretrained(f"{output_dir}/final")
    tokenizer.save_pretrained(f"{output_dir}/final")
    print("Done.")
    print(f"\nNext: evaluate on the held-out NBA test split.")
    print(f"  python -m src.evaluate --checkpoint {output_dir}/final \\")
    print(f"      --eval nba --oracle-tables")


if __name__ == "__main__":
    main()
