"""Section 2 — Single-neuron sustained-activity sweep over recurrent weight V.

Drives a single ``snn.RLeaky`` neuron (beta=0.9, all_to_all=False) with a brief
above-threshold input pulse, then turns the input off and records the membrane
trace plus spikes for ``num_steps``. Repeats for several values of V to map the
decay -> persistent -> runaway regime boundary.

Predict-before-execute: write your per-V predictions in the Obsidian week-7
note's "Section 2 — Mike's Observations" block *before* running this.

Usage:
    python section2_v_sweep.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import snntorch as snn
import torch

from neuromorphic.viz import membrane_trace, spike_raster

HERE = Path(__file__).parent
OUTPUTS = HERE / "outputs"


# ---- experiment parameters -----------------------------------------------

SEED = 42
BETA = 0.9
THRESHOLD = 1.0
NUM_STEPS = 100
PULSE_STEPS = 10           # input ON for steps 0..PULSE_STEPS-1
PULSE_AMPLITUDE = 1.5      # above threshold so spikes definitely happen
V_VALUES = [0.0, 0.3, 0.6, 0.9, 1.2]


# ---- single-V run --------------------------------------------------------

def run_one(V: float, input_train: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Run a single RLeaky neuron at the given recurrent weight.

    Returns ``(spk_rec, mem_rec)``, each shaped ``[T, 1]``.
    """
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

    # stack into [T, N=1]
    return torch.stack(spk_rec).reshape(NUM_STEPS, 1), torch.stack(mem_rec).reshape(NUM_STEPS, 1)


# ---- main ----------------------------------------------------------------

def main() -> None:
    torch.manual_seed(SEED)
    OUTPUTS.mkdir(exist_ok=True)

    # Input: constant above-threshold current during the pulse, zero after.
    # Shape [T, 1] so the neuron sees a 1-element input each step.
    input_train = torch.zeros(NUM_STEPS, 1)
    input_train[:PULSE_STEPS] = PULSE_AMPLITUDE

    results: dict[float, tuple[torch.Tensor, torch.Tensor]] = {}
    for V in V_VALUES:
        results[V] = run_one(V, input_train)

    # ---- post-pulse spike count table (the memory-horizon diagnostic)
    print("\nPost-pulse spike count (steps >= PULSE_STEPS)")
    print(f"{'V':>6}  {'total':>6}  {'pre':>6}  {'post':>6}")
    for V in V_VALUES:
        spk_rec, _ = results[V]
        total = int(spk_rec.sum().item())
        pre = int(spk_rec[:PULSE_STEPS].sum().item())
        post = int(spk_rec[PULSE_STEPS:].sum().item())
        print(f"{V:>6.2f}  {total:>6d}  {pre:>6d}  {post:>6d}")

    # ---- 5x2 panel: membrane trace + spike raster per V
    fig, axes = plt.subplots(len(V_VALUES), 2, figsize=(12, 2.0 * len(V_VALUES)),
                             sharex=True)
    for i, V in enumerate(V_VALUES):
        spk_rec, mem_rec = results[V]

        membrane_trace(mem_rec, ax=axes[i, 0], threshold=THRESHOLD)
        axes[i, 0].axvline(PULSE_STEPS, color="red", linestyle=":",
                           label="input off")
        axes[i, 0].set_title(f"V = {V:.2f}  —  membrane")
        axes[i, 0].legend(loc="upper right", fontsize=8)

        spike_raster(spk_rec, ax=axes[i, 1])
        axes[i, 1].axvline(PULSE_STEPS, color="red", linestyle=":")
        axes[i, 1].set_title(f"V = {V:.2f}  —  spikes")

    fig.suptitle(
        f"RLeaky single-neuron V sweep  (beta={BETA}, threshold={THRESHOLD}, "
        f"pulse={PULSE_STEPS} steps @ {PULSE_AMPLITUDE})",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    out_path = OUTPUTS / "section2_v_sweep.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"\nSaved: {out_path.relative_to(HERE.parent.parent)}")
    plt.close(fig)


if __name__ == "__main__":
    main()
