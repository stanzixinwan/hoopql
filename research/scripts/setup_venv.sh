#!/usr/bin/env bash
# Research venv for the frozen training stack (macOS Apple Silicon and other non-CUDA machines).
# This is separate from the product environment created by `uv sync` at the repo root.
# Recreate it on each device; it is gitignored and not portable across OSes.
set -euo pipefail

Root="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$Root"
Venv="$Root/research/.venv"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required. Install it from https://docs.astral.sh/uv/ and re-run." >&2
  exit 1
fi

echo "Creating research/.venv with Python 3.12 ..."
uv venv "$Venv" --python 3.12

echo "Installing research requirements (CPU/MPS PyTorch from PyPI) ..."
uv pip install --python "$Venv/bin/python" -r research/requirements.txt

echo
"$Venv/bin/python" - <<'PY'
import torch
print("torch:", torch.__version__)
print("mps:", torch.backends.mps.is_available())
print("cuda:", torch.cuda.is_available())
mods = [
    "transformers",
    "peft",
    "datasets",
    "sentence_transformers",
    "faiss",
    "rank_bm25",
    "gradio",
    "sqlparse",
]
for name in mods:
    __import__(name)
    print("ok:", name)
PY

echo
echo "Done. Activate with: source research/.venv/bin/activate"
echo "Training still needs a CUDA machine, or: export HOOPQL_ALLOW_CPU=1"
