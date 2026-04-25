"""Seed all RNGs for reproducibility.

Important: call this *inside* your training function, not once at module level.
Seeding once globally lets run order leak into results — Config 3 sees a
different RNG state depending on whether it ran first or third. See
week-03-pytorch.md Session 2 for the full story.
"""

from __future__ import annotations

import os
import random


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch (CPU + CUDA) RNGs.

    Also sets `PYTHONHASHSEED` so dict ordering is deterministic across runs.

    Parameters
    ----------
    seed
        Integer seed value.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
