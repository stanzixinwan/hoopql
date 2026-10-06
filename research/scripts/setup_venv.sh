#!/usr/bin/env bash
# Local venv for macOS Apple Silicon (and any non-CUDA machine).
# Recreate .venv on each device; it is gitignored and not portable across OSes.
set -euo pipefail

Root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$Root"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required. Install it from https://docs.astral.sh/uv/ and re-run." >&2
  exit 1
fi

echo "Creating .venv with Python 3.12 ..."
uv venv .venv --python 3.12

echo "Installing requirements (CPU/MPS PyTorch from PyPI) ..."
uv pip install --python .venv/bin/python -r requirements.txt

echo
.venv/bin/python - <<'PY'
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
echo "Done. Activate with: source .venv/bin/activate"
echo "Training still needs the Windows CUDA machine, or: export HOOPQL_ALLOW_CPU=1"
