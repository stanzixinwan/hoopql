"""
train.py — Fine-tune T5/CodeT5/Flan-T5 on Spider for text-to-SQL.

Supports three methods:
  - full:  full parameter fine-tuning
  - lora:  LoRA (parameter-efficient)
  - qlora: QLoRA (4-bit quantized + LoRA)

Usage:
    python -m src.train --method full --model t5-base --epochs 3
    python -m src.train --method full --model google/flan-t5-base --epochs 10
    python -m src.train --method lora --model google/flan-t5-base --rank 16 --epochs 5
    python -m src.train --method lora --model google/flan-t5-large --rank 16 --epochs 5 --batch-size 4

Outputs go to models/<run_name>/.
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

try:
    from src.data_utils import load_spider_splits
    from src.gpu_env import require_cuda
except ModuleNotFoundError:
    from data_utils import load_spider_splits
    from gpu_env import require_cuda


def load_tokenizer_with_fallback(model_name_or_path: str):
    """Load tokenizer with slow->fast fallback for CodeT5 variants."""
    try:
        return AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)
    except Exception as slow_exc:
        print(
            "Tokenizer load with use_fast=False failed; "
            f"retrying with use_fast=True for {model_name_or_path}.\n"
            f"  Reason: {slow_exc}"
        )
        return AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)


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


def _preferred_cuda_dtype():
    if _cuda_bf16_supported():
        return torch.bfloat16
    return torch.float16


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def tokenize_dataset(examples, tokenizer, max_input=1024, max_target=256):
    inputs = [ex["input_text"] for ex in examples]
    targets = [ex["target_text"] for ex in examples]
    model_inputs = tokenizer(inputs, max_length=max_input, truncation=True, padding=False)
    labels = tokenizer(text_target=targets, max_length=max_target,
                       truncation=True, padding=False)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def build_dataset(examples, tokenizer):
    ds = Dataset.from_list(examples)
    ds = ds.map(
        lambda batch: tokenize_dataset(
            [{"input_text": i, "target_text": t}
             for i, t in zip(batch["input_text"], batch["target_text"])],
            tokenizer,
        ),
        batched=True,
        remove_columns=ds.column_names,
    )
    return ds


def setup_model(model_name: str, method: str, lora_rank: int = 16):
    """Load base model and optionally wrap with PEFT.
    Uses AutoModelForSeq2SeqLM so it works for T5, Flan-T5, CodeT5 alike."""
    if method == "qlora":
        from transformers import BitsAndBytesConfig
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=_preferred_cuda_dtype(),
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name, quantization_config=bnb, device_map="auto"
        )
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    if method in ("lora", "qlora"):
        from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
        if method == "qlora":
            model = prepare_model_for_kbit_training(model)
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=lora_rank,
            lora_alpha=lora_rank * 2,
            lora_dropout=0.1,
            # T5/Flan-T5 attention + FFN projections
            target_modules=["q", "k", "v", "o", "wi", "wo"],
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="t5-base",
                        help="HF model id, e.g. t5-base, google/flan-t5-base, "
                             "google/flan-t5-large, Salesforce/codet5-base")
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="Optional tokenizer source if model tokenizer metadata is broken.",
    )
    parser.add_argument("--method", choices=["full", "lora", "qlora"], required=True)
    parser.add_argument("--rank", type=int, default=16, help="LoRA rank")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    require_cuda()
    _set_seed(args.seed)

    run_name = args.run_name or f"{args.method}_{args.model.split('/')[-1]}"
    if args.method == "lora":
        run_name += f"_r{args.rank}"
    run_name += f"_lr{args.lr:g}_s{args.seed}"
    output_dir = f"models/{run_name}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print(f"Run: {run_name}")
    print(f"Output: {output_dir}\n")
    with open(f"{output_dir}/run_config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    tokenizer_src = args.tokenizer or args.model
    print(f"Loading tokenizer: {tokenizer_src}")
    tokenizer = load_tokenizer_with_fallback(tokenizer_src)

    print("Loading Spider data...")
    train, dev = load_spider_splits(max_examples=args.max_train)
    print(f"  Train: {len(train)} | Dev: {len(dev)}")

    print("Tokenizing...")
    train_ds = build_dataset(train, tokenizer)
    dev_ds = build_dataset(dev[:500], tokenizer)

    print(f"Setting up {args.method} model...")
    model = setup_model(args.model, args.method, args.rank)

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
        warmup_steps=100,
        logging_steps=50,
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
        eval_dataset=dev_ds,
        processing_class=tokenizer,
        data_collator=collator,
    )

    print("Training...")
    trainer.train()

    print(f"\nSaving final model to {output_dir}/final")
    trainer.save_model(f"{output_dir}/final")
    tokenizer.save_pretrained(f"{output_dir}/final")
    print("Done.")


if __name__ == "__main__":
    main()
