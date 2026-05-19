"""Section 2b — 10-neuron RLeaky population, V sweep.

Faithful-to-goal companion to section2_v_sweep.py. The week brief's daily goal
#3 calls for a "tiny single-layer RLeaky network (10 neurons, no learning)",
while the section-2 code spec uses a single neuron. This script bridges the
gap: same V sweep, same input pulse, but with N=10 neurons and small per-neuron
Gaussian noise on the input so the population raster is non-trivial.

With ``all_to_all=False`` the 10 neurons have no lateral coupling — they're 10
independent copies of the single-neuron experiment. The point of running them
together is to see the population-level statistics of the regime boundary:
near-critical V will fire some neurons indefinitely and let others decay,
depending on the noise realization. Above critical, runaway is robust.

Usage:
    python section2b_10neuron_population.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import snntorch as snn
import torch

from neuromorphic.viz import membrane_trace, population_rate, spike_raster

HERE = Path(__file__).parent
OUTPUTS = HERE / "outputs"


# ---- experiment parameters (mirrors section 2 except N and noise) --------

SEED = 42
BETA = 0.9
THRESHOLD = 1.0
NUM_STEPS = 100
PULSE_STEPS = 10
PULSE_AMPLITUDE = 1.5
NOISE_SIGMA = 0.15           # per-neuron Gaussian noise on the input
N_NEURONS = 10
V_VALUES = [0.0, 0.3, 0.6, 0.9, 1.2]


# ---- single-V run --------------------------------------------------------

def run_one(V: float, input_train: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Run a 10-neuron RLeaky layer at the given recurrent weight.

    Returns ``(spk_rec, mem_rec)``, each shaped ``[T, N]``.
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

    return torch.stack(spk_rec).reshape(NUM_STEPS, N_NEURONS), \
           torch.stack(mem_rec).reshape(NUM_STEPS, N_NEURONS)


# ---- main ----------------------------------------------------------------

def main() -> None:
    torch.manual_seed(SEED)
    OUTPUTS.mkdir(exist_ok=True)

    # Input: above-threshold pulse + per-neuron Gaussian noise. Shape [T, N].
    base = torch.zeros(NUM_STEPS, N_NEURONS)
    base[:PULSE_STEPS] = PULSE_AMPLITUDE
    noise = NOISE_SIGMA * torch.randn(NUM_STEPS, N_NEURONS)
    input_train = base + noise

    results: dict[float, tuple[torch.Tensor, torch.Tensor]] = {}
    for V in V_VALUES:
        results[V] = run_one(V, input_train)

    # ---- per-V post-pulse population statistics
    print("\nPost-pulse spikes per neuron (steps >= PULSE_STEPS) — mean ± std")
    print(f"{'V':>6}  {'mean':>6}  {'std':>6}  {'min':>4}  {'max':>4}")
    for V in V_VALUES:
        spk_rec, _ = results[V]
        post_counts = spk_rec[PULSE_STEPS:].sum(dim=0).float()
        print(f"{V:>6.2f}  {post_counts.mean():>6.2f}  "
              f"{post_counts.std():>6.2f}  "
              f"{int(post_counts.min()):>4d}  {int(post_counts.max()):>4d}")

    # ---- 5x2 grid: population raster + population rate per V
    fig, axes = plt.subplots(len(V_VALUES), 2, figsize=(12, 2.0 * len(V_VALUES)),
                             sharex=True)
    for i, V in enumerate(V_VALUES):
        spk_rec, _ = results[V]

        spike_raster(spk_rec, ax=axes[i, 0])
        axes[i, 0].axvline(PULSE_STEPS, color="red", linestyle=":",
                           label="input off")
        axes[i, 0].set_title(f"V = {V:.2f}  —  population raster (N={N_NEURONS})")
        axes[i, 0].legend(loc="upper right", fontsize=8)

        # population_rate wants [T, B, N] — wrap with a singleton batch dim.
        population_rate(spk_rec.reshape(NUM_STEPS, 1, N_NEURONS),
                        smoothing_window=5, ax=axes[i, 1])
        axes[i, 1].axvline(PULSE_STEPS, color="red", linestyle=":")
        axes[i, 1].set_title(f"V = {V:.2f}  —  population rate (5-step smoothed)")

    fig.suptitle(
        f"10-neuron RLeaky V sweep  (all_to_all=False, beta={BETA}, "
        f"noise sigma={NOISE_SIGMA}, pulse={PULSE_STEPS} steps @ {PULSE_AMPLITUDE})",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_path = OUTPUTS / "section2b_10neuron_population.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"\nSaved: {out_path.relative_to(HERE.parent.parent)}")
    plt.close(fig)

    # ---- bonus: per-neuron membrane traces at the memory-regime V
    spk_mem_v, mem_mem_v = results[0.9]
    fig2, (ax_mem, ax_ras) = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
    membrane_trace(mem_mem_v, ax=ax_mem, threshold=THRESHOLD)
    ax_mem.axvline(PULSE_STEPS, color="red", linestyle=":", label="input off")
    ax_mem.set_title("V=0.9 — per-neuron membrane traces (N=10)")
    ax_mem.legend(loc="upper right", fontsize=8)

    spike_raster(spk_mem_v, ax=ax_ras)
    ax_ras.axvline(PULSE_STEPS, color="red", linestyle=":")
    ax_ras.set_title("V=0.9 — population raster")

    fig2.tight_layout()
    out_path2 = OUTPUTS / "section2b_10neuron_memory_regime.png"
    fig2.savefig(out_path2, dpi=120, bbox_inches="tight")
    print(f"Saved: {out_path2.relative_to(HERE.parent.parent)}")
    plt.close(fig2)


if __name__ == "__main__":
    main()
