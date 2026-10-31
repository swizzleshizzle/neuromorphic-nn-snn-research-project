"""Choosing the torch device for judge training and search (EXP-084).

"cpu" changes nothing, so the CPU path stays byte-identical. Any other device must be available,
and gets full fp32: TF32 and reduced-precision matmul are switched off, so the GPU computes in the
same precision as the CPU and differs from it only by rounding order. ROCm exposes an AMD GPU as
"cuda".
"""

from __future__ import annotations

import torch


def setup_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if not torch.cuda.is_available():
        raise SystemExit(f"device {name!r} requested but torch.cuda.is_available() is False "
                         f"(torch {torch.__version__})")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    return torch.device(name)
