# Research results

Frozen evaluation outputs from the original text-to-SQL experiments. Per-run dumps live in [`runs/`](runs/). The files in this directory are the summaries.

Regenerate the report tables with:

```bash
python -m research.build_results_report
```

That reads `runs/*.json` and rewrites the summary CSV and Markdown files here.

## Summary CSVs

- `results_summary.csv` — one row per NBA `*_test.json` run. Columns include model, method (`full`, `lora`, `qlora`, `baseline`), schema mode (`oracle`, `rag`, or other), adaptation size `n_train`, execution accuracy, exact match, and bootstrap confidence intervals. `results_summary.md` is the same table.
- `fewshot_curve.csv` — oracle-schema runs only, ordered by NBA adaptation size (`n_train` of 0, 10, 20, 70, or all). This is the adaptation curve: execution and exact-match accuracy versus how many in-domain examples the checkpoint saw.
- `rag_ablation.csv` — the RAG rows from `results_summary.csv` (execution and exact-match accuracy, with confidence intervals). It does not include retrieval recall; that is in the files below.
- `spider_summary.csv` — Spider dev runs (`*_spider.json` in `runs/`). Exact match is filled in. Execution accuracy is 0 because Spider execution was not wired in `research/evaluate.py`. `spider_summary.md` is the same table.
- `rag_retrieval_summary.csv` — retrieval only, on the 50-question NBA test split. One row per backend (`dense`, `bm25`, `hybrid`) and `k` in {1, 3, 5}, with mean table recall and the rate at which every gold table was retrieved.
- `rag_e2e_summary.csv` — the same backend × `k` grid, joined with downstream execution accuracy. `checkpoint_tag` separates the Spider-only CodeT5+ checkpoint from the one adapted on the full NBA train split. `mean_retrieval_recall` is measured on the examples in that run's JSON.
- `rag_recall_bins_summary.csv` — execution rate inside retrieval-recall bins, for each checkpoint, backend, and `k`. Used to separate "the retriever missed a table" from "the tables were retrieved and generation still failed."

## Other files

- `error_analysis_summary.json` — manual error classes (`correct`, `value`, `structural`, `schema_linking`) for the best oracle checkpoint and one RAG run, including a per-difficulty breakdown.
- `runs/` — raw per-run JSON. Each NBA test file is a list of example records (question, predicted SQL, execution match). Spider files are the same shape for the Spider split.
