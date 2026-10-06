# HoopQL

HoopQL turns natural-language basketball questions into executable SQL. This repository includes data processing, training, evaluation, and a demo.

## Project Snapshot

- **Task:** text-to-SQL generation (natural language -> SQL).
- **Primary training data:** Spider.
- **Domain evaluation data:** custom NBA SQLite dataset and NBA question/SQL pairs.
- **Modeling focus:** full fine-tuning vs LoRA vs QLoRA, plus schema retrieval (RAG).
- **Main evaluation metrics:** execution accuracy and exact match.

## Environment Setup

### 1) Create environment and install dependencies

`.venv` is per machine and gitignored. Recreate it after switching devices; do not copy it.

macOS (Apple Silicon):

```bash
./scripts/setup_venv.sh
source .venv/bin/activate
```

Windows with NVIDIA: `.\scripts\setup_gpu_venv.ps1` (see `GPU_SETUP.md`). That script also installs `requirements-gpu.txt` (`bitsandbytes`), which is CUDA-only and is not part of the Mac install.

### 2) GPU requirement

`src/train.py`, `src/train_nba.py`, and `src/evaluate.py` require CUDA by default.

- Recommended on Windows/NVIDIA: Python 3.12 or 3.13.
- For debug-only CPU runs, set:

```powershell
$env:HOOPQL_ALLOW_CPU=1
```

For additional Windows GPU setup details, see `GPU_SETUP.md`.

## Data Preparation

### Spider

Spider is downloaded automatically on first training/evaluation use via HuggingFace datasets.

### NBA SQLite database

Build `data/raw/nba.sqlite` from [nba-sql](https://github.com/mpope9/nba-sql):

```bash
git clone https://github.com/mpope9/nba-sql.git
cd nba-sql
python stats/nba_sql.py --database sqlite --sqlite-path ../data/raw/nba.sqlite
cd ..
```

Expected local files for NBA experiments:

- `data/raw/nba.sqlite`
- `data/nba/nba_questions.json`

## Train and Evaluate

### A) Prompt baseline (no fine-tuning)

```bash
python -m src.prompt_baseline --model t5-base --eval nba --split test --output eval/baseline_nba_zeroshot_t5-base_test.json
```

### B) Spider training (full / LoRA / QLoRA)

```bash
# Full fine-tuning
python -m src.train --method full --model t5-base --epochs 3 --seed 42

# LoRA
python -m src.train --method lora --model t5-base --rank 16 --epochs 3 --lr 1e-4 --seed 42

# QLoRA
python -m src.train --method qlora --model t5-base --rank 16 --epochs 3 --lr 1e-4 --seed 42
```

Training outputs are saved under `models/<run_name>/final`.

### C) NBA test evaluation (default held-out split)

```bash
python -m src.evaluate --checkpoint models/<run_name>/final --eval nba --split test
```

Evaluation outputs are saved under `eval/` (for example, `eval/<run_name>_nba_full_test.json`).

### D) RAG retrieval + NBA evaluation

```bash
# Build retrieval indices
python -m src.rag --build
python -m src.rag --build-bm25

# Evaluate with dense retrieval
python -m src.evaluate --checkpoint models/<run_name>/final --eval nba --use-rag --rag-backend dense --top-k 3 --split test

# Optional: bm25 or hybrid retrieval
python -m src.evaluate --checkpoint models/<run_name>/final --eval nba --use-rag --rag-backend bm25 --top-k 3 --split test
python -m src.evaluate --checkpoint models/<run_name>/final --eval nba --use-rag --rag-backend hybrid --top-k 3 --split test
```

### E) NBA adaptation (few-shot domain fine-tuning)

```bash
python -m src.train_nba --base-checkpoint models/<spider_run>/final --base-model t5-base --n-train 10 --epochs 10 --seed 42
python -m src.train_nba --base-checkpoint models/<spider_run>/final --base-model t5-base --n-train 20 --epochs 10 --seed 42
python -m src.train_nba --base-checkpoint models/<spider_run>/final --base-model t5-base --n-train 70 --epochs 10 --seed 42
python -m src.train_nba --base-checkpoint models/<spider_run>/final --base-model t5-base --n-train all --epochs 10 --seed 42
```

Then evaluate each adapted checkpoint on held-out NBA test:

```bash
python -m src.evaluate --checkpoint models/<adapted_run>/final --eval nba --oracle-tables --split test
```

## Reproduce Main Results

Use this sequence to regenerate core tables from `eval/*.json`:

1. Run baseline and trained-model evaluations on NBA test split (`--split test`).
2. Run RAG variants (`dense`, `bm25`, `hybrid`) if comparing retrieval.
3. Run adaptation checkpoints (`n=10/20/70/all`) and evaluate on NBA test split.
4. Build aggregate report:

```bash
python -m src.build_results_report --eval-dir eval --glob "*_test.json"
```

Primary aggregated outputs:

- `eval/results_summary.csv`
- `eval/results_summary.md`
- `eval/fewshot_curve.csv`
- `eval/rag_ablation.csv`
- `eval/spider_summary.csv`
- `eval/spider_summary.md`

## Demo

```bash
python -m src.demo --checkpoint models/<run_name>/final --base-model t5-base --nba-db data/raw/nba.sqlite
```

## Repository Map

```text
src/data_utils.py           data loading and schema serialization
src/train.py                Spider-stage training (full/LoRA/QLoRA)
src/train_nba.py            NBA adaptation training
src/evaluate.py             NBA/Spider evaluation
src/prompt_baseline.py      zero-shot/few-shot prompting baseline
src/rag.py                  dense/BM25/hybrid schema retrieval
src/build_results_report.py aggregate CSV/Markdown result tables
src/demo.py                 Gradio demo app
data/                       NBA inputs and SQLite database
models/                     checkpoints (generated locally)
eval/                       evaluation JSON and summary outputs
```

## Known Limitations

- SQL equivalence is approximated by execution match and normalized exact match; semantically equivalent but differently formatted SQL can still fail exact match.
- Spider execution accuracy is not computed in `src.evaluate.py` (exact match only) because per-database execution setup is not wired there.
- RAG quality depends on table-level retrieval; column-level retrieval and reranking are not included.
