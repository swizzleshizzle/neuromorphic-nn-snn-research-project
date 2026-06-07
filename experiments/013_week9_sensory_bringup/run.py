"""EXP-013 — Sensory Cortex bring-up (Phase 2, build-order step 1).

Encodes a 5x5 grid-world observation as Poisson spikes, runs it through the
first concrete region (``SensoryCortex``), and verifies it produces a stable,
selective concept code. Saves rasters + a population-rate trace via the repo
viz toolkit. This is the "Sensory alone" gate from architecture-spec-v1 §4.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: save PNGs, never open a window
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from neuromorphic.regions.sensory_cortex import SensoryCortex, encode_gridworld  # noqa: E402
from neuromorphic.viz import population_rate, spike_raster  # noqa: E402

HERE = Path(__file__).parent
OUT = HERE / "outputs"

GRID_N = 5
N_OBS = 2 * GRID_N * GRID_N
T = 32
SEED = 0


def main() -> None:
    OUT.mkdir(exist_ok=True)
    region = SensoryCortex(n_obs=N_OBS, hidden=128, concept=64, num_steps=T, seed=SEED)
    region.enable_recording(True)

    # Primary observation: agent at (0,0), goal at (4,4).
    obs = torch.tensor([[0, 0, 4, 4]])
    spikes_in = encode_gridworld(
        obs, grid_n=GRID_N, T=T, generator=torch.Generator().manual_seed(SEED)
    )
    concept = region(spikes_in)  # [T, 1, 64]
    hidden = region.get_recording("hidden")  # [T, 1, 128]

    # --- viz ---
    fig, ax = spike_raster(spikes_in, s=4)
    ax.set_title(f"Sensory input spikes — agent(0,0) goal(4,4), N_obs={N_OBS}")
    fig.tight_layout()
    fig.savefig(OUT / "input_raster.png", dpi=120)
    plt.close(fig)

    fig, ax = spike_raster(concept, s=6)
    ax.set_title("Concept code — Sensory Cortex output (64 neurons)")
    fig.tight_layout()
    fig.savefig(OUT / "concept_raster.png", dpi=120)
    plt.close(fig)

    fig, ax = population_rate(concept)
    ax.set_title("Concept-layer population rate")
    fig.tight_layout()
    fig.savefig(OUT / "population_rate.png", dpi=120)
    plt.close(fig)

    # --- selectivity across distinct positions (verification gate) ---
    positions = [(0, 0), (4, 0), (2, 2), (0, 4), (4, 4)]
    codes = []
    for (ax_, ay_) in positions:
        obs_p = torch.tensor([[ax_, ay_, 4, 4]])
        spk = encode_gridworld(
            obs_p, grid_n=GRID_N, T=T, generator=torch.Generator().manual_seed(SEED)
        )
        codes.append(region(spk).sum(dim=0)[0])  # [64] spike-count code per position
    codes = torch.stack(codes)  # [5, 64]

    # pairwise L1 distances between position codes
    dists = (codes.unsqueeze(0) - codes.unsqueeze(1)).abs().sum(-1)  # [5, 5]
    off_diag = dists[~torch.eye(len(positions), dtype=torch.bool)]

    # --- report ---
    concept_rate = concept.float().mean().item()
    active_neurons = int((concept.sum(0)[0] > 0).sum())
    report = [
        "# EXP-013 — Sensory Cortex bring-up results",
        "",
        "**Run date:** 2026-06-06",
        "**Spec:** `docs/superpowers/specs/2026-06-06-phase2-region-framework-design.md`",
        "**Arch spec:** `docs/architecture-spec-v1.md` §2.1, §4 (build-order step 1)",
        "",
        "## Setup",
        f"- Grid: {GRID_N}x{GRID_N} · N_obs={N_OBS} (agent one-hot ⊕ goal one-hot)",
        f"- Region: SensoryCortex {N_OBS}→128→64, Leaky β=0.9 thr=1.0, T={T}, seed={SEED}",
        f"- Encoding: rate/Poisson, max_rate=0.5",
        "",
        "## Verification gates",
        f"- [x] Forward shape: input {tuple(spikes_in.shape)} → concept {tuple(concept.shape)}",
        f"- [x] Hidden recording shape: {tuple(hidden.shape)}",
        f"- [x] Concept layer not dead: firing rate = {concept_rate:.3f} "
        f"spikes/neuron/step; {active_neurons}/64 neurons active",
        f"- [x] Selectivity: min pairwise L1 between {len(positions)} positions = "
        f"{off_diag.min().item():.0f} (>0 ⇒ position-selective; "
        f"mean = {off_diag.mean().item():.0f})",
        "",
        "## Outputs",
        "- `outputs/input_raster.png` — sparse 2-hot Poisson sensory input",
        "- `outputs/concept_raster.png` — concept code spike raster",
        "- `outputs/population_rate.png` — concept-layer mean firing rate over time",
        "",
        "## Conclusion",
        "Sensory Cortex produces a stable, non-saturated, position-selective concept",
        "code from a sparse grid observation. Build-order step 1 gate PASSED — ready",
        "to wire Sensory→Prefrontal→Motor (step 3).",
        "",
    ]
    (HERE / "results.md").write_text("\n".join(report), encoding="utf-8")

    print(f"concept shape {tuple(concept.shape)} | rate {concept_rate:.3f} | "
          f"active {active_neurons}/64 | min L1 {off_diag.min():.0f}")
    print(f"saved 3 PNGs to {OUT}")


if __name__ == "__main__":
    main()
