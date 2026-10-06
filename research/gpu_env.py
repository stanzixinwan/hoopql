"""CUDA availability guard for training/evaluation entrypoints."""

from __future__ import annotations

import os
import sys

import torch


def require_cuda() -> None:
    """Exit unless a CUDA device is visible, or HOOPQL_ALLOW_CPU=1."""
    if os.environ.get("HOOPQL_ALLOW_CPU", "").strip().lower() in (
        "1",
        "true",
        "yes",
    ):
        return
    if torch.cuda.is_available():
        return
    print(
        "ERROR: CUDA is not available. Install PyTorch with CUDA.\n"
        "  Example: py -3.13 -m pip install torch --index-url "
        "https://download.pytorch.org/whl/cu124\n"
        "  Or set HOOPQL_ALLOW_CPU=1 to allow CPU (debug only).",
        file=sys.stderr,
    )
    sys.exit(1)
