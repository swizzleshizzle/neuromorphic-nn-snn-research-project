"""Device detection — CPU / CUDA / (later) MPS."""

from __future__ import annotations


def get_device(prefer: str = "auto") -> "torch.device":
    """Return the best available torch.device.

    Parameters
    ----------
    prefer
        "auto" (default): CUDA if available, else CPU.
        "cpu": force CPU regardless of CUDA availability.
        "cuda": force CUDA, raise if unavailable.

    Returns
    -------
    torch.device
    """
    import torch

    if prefer == "cpu":
        return torch.device("cpu")
    if prefer == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")
    if prefer == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raise ValueError(f"Unknown device preference: {prefer!r}. Use 'auto', 'cpu', or 'cuda'.")
