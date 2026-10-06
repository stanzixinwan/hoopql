"""
demo.py — Gradio app for HoopQL inference.

Features:
  - Natural language input
  - SQL generation
  - SQL execution on SQLite
  - Optional RAG table retrieval display
"""

import argparse
import sqlite3

import gradio as gr

from src.evaluate import load_checkpoint
from src.data_utils import NBA_CORE_TABLES, get_sqlite_schema, serialize_schema
from src.prompt_baseline import generate_sql
from src.rag import SchemaRetriever


def _execute_sql(sql: str, db_path: str) -> tuple[str, list[list[str]]]:
    try:
        conn = sqlite3.connect(db_path, timeout=5.0)
        cur = conn.cursor()
        cur.execute(sql)
        columns = [d[0] for d in cur.description] if cur.description else []
        rows = cur.fetchall()
        conn.close()
        data = [list(columns)] + [list(map(str, row)) for row in rows[:100]]
        return "ok", data
    except Exception as exc:  # pragma: no cover
        return f"error: {exc}", []


def launch(args: argparse.Namespace) -> None:
    model, tokenizer = load_checkpoint(args.checkpoint, args.base_model)
    model.to(args.device).eval()
    full_schema = get_sqlite_schema(args.nba_db, tables=NBA_CORE_TABLES)

    retriever = None
    if args.use_rag:
        retriever = SchemaRetriever(model_name=args.rag_encoder)
        retriever.load()

    def infer(question: str):
        if not question.strip():
            return "", "Please enter a question.", []

        retrieved = []
        schema = full_schema
        if retriever is not None:
            retrieved = retriever.retrieve(question, top_k=args.top_k)
            schema = {t: full_schema[t] for t in retrieved if t in full_schema}

        schema_text = serialize_schema(schema, include_types=True)
        model_input = f"translate to SQL: {question} | {schema_text}"
        pred_sql = generate_sql(model, tokenizer, model_input, args.device)
        status, result = _execute_sql(pred_sql, args.nba_db)
        return pred_sql, status, result, ", ".join(retrieved)

    with gr.Blocks(title="HoopQL") as app:
        gr.Markdown("# HoopQL")
        gr.Markdown("输入英文问题，模型会生成 SQL 并在 NBA SQLite 上执行。")

        question = gr.Textbox(label="Question", placeholder="How many teams are in the NBA?")
        run = gr.Button("Run")

        sql_box = gr.Textbox(label="Generated SQL")
        status_box = gr.Textbox(label="Execution Status")
        table_view = gr.Dataframe(label="Query Results", wrap=True)
        rag_box = gr.Textbox(label="Retrieved Tables (RAG)", interactive=False)

        run.click(fn=infer, inputs=[question], outputs=[sql_box, status_box, table_view, rag_box])

    app.launch(server_name=args.host, server_port=args.port, share=args.share)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--base-model", default="t5-base")
    parser.add_argument("--nba-db", default="data/raw/nba.sqlite")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--use-rag", action="store_true")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--rag-encoder", default="all-MiniLM-L6-v2")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    launch(parse_args())
