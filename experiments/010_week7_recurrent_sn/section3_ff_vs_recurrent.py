"""Section 3 — Feedforward vs Recurrent: same input, different stories.

Drives a single plain ``snn.Leaky`` neuron (no memory) and a single
``snn.RLeaky`` neuron with V=0.9 (the memory regime from section 2) with the
*same* input pulse. Plots both membrane traces and spike rasters together so
the working-memory effect of the single recurrent connection is visible at a
glance.

The "memory amplification factor" printed at the end is just
``post_pulse_spikes(recurrent) / max(post_pulse_spikes(feedforward), 1)`` — a
crude one-number summary of how much longer the recurrent representation
outlives the input.

Usage:
    python section3_ff_vs_recurrent.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import snntorch as snn
import torch

from neuromorphic.viz import membrane_trace, spike_raster

HERE = Path(__file__).parent
OUTPUTS = HERE / "outputs"


# ---- experiment parameters (kept consistent with section 2) --------------

SEED = 42
BETA = 0.9
THRESHOLD = 1.0
NUM_STEPS = 100
PULSE_STEPS = 10
PULSE_AMPLITUDE = 1.5
V_RECURRENT = 0.9          # memory regime (from section 2 sweep)


# ---- network forward passes ----------------------------------------------

def run_feedforward(input_train: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    lif = snn.Leaky(beta=BETA, threshold=THRESHOLD)
    mem = lif.init_leaky()
    spk_rec, mem_rec = [], []
    for step in range(NUM_STEPS):
        spk, mem = lif(input_train[step], mem)
        spk_rec.append(spk)
        mem_rec.append(mem)
    return torch.stack(spk_rec).reshape(NUM_STEPS, 1), torch.stack(mem_rec).reshape(NUM_STEPS, 1)


def run_recurrent(input_train: torch.Tensor, V: float) -> tuple[torch.Tensor, torch.Tensor]:
    rlif = snn.RLeaky(
        beta=BETA,
        all_to_all=False,
        V=V,
        learn_recurrent=False,
        threshold=THRESHOLD,
    )
    spk, mem = rlif.init_rleaky()
    spk_rec, mem_rec = [], []
    for step in range(NUM_STEPS):
        spk, mem = rlif(input_train[step], spk, mem)
        spk_rec.append(spk)
        mem_rec.append(mem)
    return torch.stack(spk_rec).reshape(NUM_STEPS, 1), torch.stack(mem_rec).reshape(NUM_STEPS, 1)


# ---- main ----------------------------------------------------------------

def main() -> None:
    torch.manual_seed(SEED)
    OUTPUTS.mkdir(exist_ok=True)

    input_train = torch.zeros(NUM_STEPS, 1)
    input_train[:PULSE_STEPS] = PULSE_AMPLITUDE

    spk_a, mem_a = run_feedforward(input_train)
    spk_b, mem_b = run_recurrent(input_train, V_RECURRENT)

    # ---- diagnostic numbers
    pre_a = int(spk_a[:PULSE_STEPS].sum().item())
    post_a = int(spk_a[PULSE_STEPS:].sum().item())
    pre_b = int(spk_b[:PULSE_STEPS].sum().item())
    post_b = int(spk_b[PULSE_STEPS:].sum().item())
    amplification = post_b / max(post_a, 1)

    print("\nSpike counts")
    print(f"{'network':>14}  {'pre':>5}  {'post':>5}")
    print(f"{'feedforward':>14}  {pre_a:>5d}  {post_a:>5d}")
    print(f"{f'recurrent V={V_RECURRENT}':>14}  {pre_b:>5d}  {post_b:>5d}")
    print(f"\nMemory amplification factor (post-pulse): {amplification:.2f}x")

    # ---- side-by-side panels for membrane + raster
    fig, axes = plt.subplots(2, 2, figsize=(12, 6), sharex=True)

    membrane_trace(mem_a, ax=axes[0, 0], threshold=THRESHOLD)
    axes[0, 0].axvline(PULSE_STEPS, color="red", linestyle=":", label="input off")
    axes[0, 0].set_title("Feedforward (snn.Leaky)  —  membrane")
    axes[0, 0].legend(loc="upper right", fontsize=8)

    spike_raster(spk_a, ax=axes[0, 1])
    axes[0, 1].axvline(PULSE_STEPS, color="red", linestyle=":")
    axes[0, 1].set_title("Feedforward  —  spikes")

    membrane_trace(mem_b, ax=axes[1, 0], threshold=THRESHOLD)
    axes[1, 0].axvline(PULSE_STEPS, color="red", linestyle=":", label="input off")
    axes[1, 0].set_title(f"Recurrent (snn.RLeaky, V={V_RECURRENT})  —  membrane")
    axes[1, 0].legend(loc="upper right", fontsize=8)

    spike_raster(spk_b, ax=axes[1, 1])
    axes[1, 1].axvline(PULSE_STEPS, color="red", linestyle=":")
    axes[1, 1].set_title(f"Recurrent V={V_RECURRENT}  —  spikes")

    fig.suptitle(
        f"Feedforward vs Recurrent  (beta={BETA}, pulse={PULSE_STEPS} steps @ "
        f"{PULSE_AMPLITUDE}, amplification={amplification:.2f}x)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    out_path = OUTPUTS / "section3_ff_vs_recurrent.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"\nSaved: {out_path.relative_to(HERE.parent.parent)}")
    plt.close(fig)


if __name__ == "__main__":
    main()
