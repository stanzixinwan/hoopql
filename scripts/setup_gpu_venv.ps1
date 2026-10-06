# Create .venv with Python 3.13 and PyTorch CUDA 12.4 (RTX / Windows).
# Python 3.14 has no official CUDA wheels yet; use 3.12 or 3.13 for GPU training.
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$Py = "3.13"
if (-not (py "-$Py" -c "import sys; print(sys.version)" 2>$null)) {
    Write-Error "Python $Py not found. Install from python.org or use 'py -0p' to list interpreters."
}

Write-Host "Creating venv at $Root\.venv with Python $Py ..."
py "-$Py" -m venv .venv

$Activate = Join-Path $Root ".venv\Scripts\Activate.ps1"
. $Activate

python -m pip install -U pip wheel
Write-Host "Installing PyTorch with CUDA 12.4 (large download) ..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

Write-Host "Installing project requirements ..."
pip install -r requirements.txt
pip install -r requirements-gpu.txt

Write-Host ""
python -c "import torch; print('torch:', torch.__version__); print('cuda:', torch.cuda.is_available()); print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'n/a')"
Write-Host ""
Write-Host "Done. Activate later with: .\.venv\Scripts\Activate.ps1"
