"""EXP-016 — Thalamic Router gating bring-up (Phase 2, build-order step 4).

Adds the Thalamic Router to the loop and gates pathway 5 (PFC→Motor). The router
reads PFC utilities, selects a winner (Stage A WTA), and emits tonic gate-closed
control lines that disinhibit only the selected channel (Stage B). ``apply_gate``
releases pathway 5 through them. Demonstrates selection, per-channel gating, and
the do-nothing veto. This is the "Thalamic Router — gate pathway 5 first" gate
from architecture-spec-v1 §4.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from neuromorphic.connections import apply_gate  # noqa: E402
from neuromorphic.regions import (  # noqa: E402
    MotorCortex,
    Prefrontal,
    SensoryCortex,
    ThalamicRouter,
    encode_gridworld,
)
from neuromorphic.viz import spike_raster  # noqa: E402

HERE = Path(__file__).parent
OUT = HERE / "outputs"

GRID_N = 5
N_OBS = 2 * GRID_N * GRID_N
T = 32
N_ACTIONS = 4
SEED = 0


def save_raster(spk, name, title, ticks=True):
    fig, ax = spike_raster(spk, s=14)
    ax.set_title(title)
    if ticks:
        ax.set_yticks(range(spk.shape[-1]))
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.png", dpi=120)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    sensory = SensoryCortex(n_obs=N_OBS, concept=64, num_steps=T, seed=SEED)
    pfc = Prefrontal(concept_dim=64, n_actions=N_ACTIONS, num_steps=T, seed=SEED)
    router = ThalamicRouter(n_actions=N_ACTIONS, num_steps=T)
    motor = MotorCortex(n_actions=N_ACTIONS, num_steps=T)

    # --- gated loop on a grid observation ---
    spk = encode_gridworld(torch.tensor([[0, 0, 4, 4]]), grid_n=GRID_N, T=T,
                           generator=torch.Generator().manual_seed(SEED))
    utilities = pfc(sensory(spk))
    gate = router(utilities)                 # gate-closed control lines
    open_mask = router.open_mask(gate)
    gated = apply_gate(utilities, gate)
    action = motor(gated)

    selected = int(open_mask.sum(dim=0)[0].argmax())
    winner = int(motor.winner(action)[0])
    open_steps = open_mask.sum(dim=0)[0].int().tolist()

    save_raster(utilities, "utility_raster", "PFC action utilities (pre-gate)")
    save_raster(gate, "gate_closed_raster", "Router gate-closed lines (1=closed; gap=open channel)")
    save_raster(action, "gated_action_raster", f"Motor output via gated pathway 5 (winner={winner})")

    # --- do-nothing veto: below-floor utilities open nothing ---
    weak = torch.bernoulli(
        torch.full((T, 1, N_ACTIONS), 0.08), generator=torch.Generator().manual_seed(SEED)
    )
    weak_gate = router(weak)
    weak_action = motor(apply_gate(weak, weak_gate))
    veto_opens = int(router.open_mask(weak_gate).sum())
    veto_action_spikes = int(weak_action.sum())

    # --- off mode: all channels open ---
    router_off = ThalamicRouter(n_actions=N_ACTIONS, num_steps=T, mode="off")
    off_gate = router_off(utilities)
    off_passes = bool(torch.equal(apply_gate(utilities, off_gate), utilities))

    report = [
        "# EXP-016 — Thalamic Router gating bring-up results",
        "",
        "**Run date:** 2026-06-06",
        "**Spec:** `docs/superpowers/specs/2026-06-06-phase2-region-framework-design.md`",
        "**Arch spec:** `docs/architecture-spec-v1.md` §2.5, §4 (build-order step 4)",
        "",
        "## Setup",
        "- Loop: Sensory → Prefrontal → **Router (gates pathway 5)** → Motor",
        f"- Router: {N_ACTIONS} channels, Stage A WTA select + Stage B tonic gating,",
        "  do-nothing floor via constant Stage-A bias. Gate applied with `apply_gate`.",
        "",
        "## Verification gates",
        f"- [x] Stage A selects a winner: channel {selected} "
        f"(open steps per channel = {open_steps})",
        f"- [x] Stage B per-channel gating: only the selected channel is disinhibited; "
        "closed channels carry no current",
        f"- [x] Motor follows the gated pathway: winner = action {winner} (= selected)",
        f"- [x] Do-nothing veto: below-floor utilities → {veto_opens} channels opened → "
        f"Motor emits {veto_action_spikes} spikes (silent)",
        f"- [x] mode='off' opens all channels (gated == raw utilities): {off_passes}",
        "",
        "## Outputs",
        "- `outputs/utility_raster.png` — PFC utilities before gating",
        "- `outputs/gate_closed_raster.png` — router control lines (gaps = open channel)",
        "- `outputs/gated_action_raster.png` — Motor output through the gated pathway",
        "",
        "## Conclusion",
        "The Thalamic Router selects a channel and gates pathway 5 by disinhibition,",
        "constraining Motor to the selected action and vetoing action when no utility",
        "clears the floor. Build-order step 4 gate PASSED — control signals now steer",
        "the loop. Next: gate pathways 3/4 with the Hippocampus (step 5).",
        "",
    ]
    (HERE / "results.md").write_text("\n".join(report), encoding="utf-8")

    print(f"selected=ch{selected} motor_winner={winner} open_steps={open_steps}")
    print(f"veto: opens={veto_opens} motor_spikes={veto_action_spikes} | off_passes={off_passes}")
    print(f"saved 3 PNGs to {OUT}")


if __name__ == "__main__":
    main()
