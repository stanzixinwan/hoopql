# GPU setup (Windows, NVIDIA)

## Why CPU was used before

- **`python` defaulted to 3.14**, which currently has **no official PyTorch wheels with CUDA** on Windows.
- **`pip install torch`** from PyPI often resolves to **`…+cpu`**.

Use **Python 3.12 or 3.13** for CUDA builds (your RTX 3080 works with cu124).

macOS Apple Silicon does not use this file. Run `./scripts/setup_venv.sh` there instead. `.venv` is not shared between machines.

## One-shot venv (recommended)

From the repo root:

```powershell
.\scripts\setup_gpu_venv.ps1
```

Then always activate before training or evaluation:

```powershell
.\.venv\Scripts\Activate.ps1
```

## Manual install (same as the script)

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## Verify

You should see `cuda: True` and your GPU name. Driver-side “CUDA Version” in `nvidia-smi` only needs to be **≥** the CUDA toolkit bundled with PyTorch (cu124 is fine with recent drivers).

## Optional CPU override (debug only)

Training and evaluation scripts refuse to run on CPU unless you set:

```powershell
$env:HOOPQL_ALLOW_CPU="1"
```

## Agent / automation note

Use the **activated `.venv`** interpreter or `py -3.13` so commands do not fall back to Python 3.14 + CPU PyTorch.
