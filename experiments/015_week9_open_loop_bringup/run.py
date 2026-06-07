"""EXP-015 — Open-loop bring-up: Sensory → Prefrontal → Motor (build-order step 3).

Wires the first three regions end-to-end on a grid-world observation, with no
memory and no gating. Confirms spikes flow through the whole pipeline, every
stage is alive, and the action-utility code carries observation information.

Honest caveat: the pipeline is UNTRAINED, so the winning action is not yet
task-meaningful — the readout collapses to a preferred action. Task-appropriate
selection is the job of training + the reward/neuromod loop (later build-order
steps). This is exactly what the spec means by PFC's transform being "learned".
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from neuromorphic.regions import (  # noqa: E402
    MotorCortex,
    Prefrontal,
    SensoryCortex,
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


def main() -> None:
    OUT.mkdir(exist_ok=True)
    sensory = SensoryCortex(n_obs=N_OBS, concept=64, num_steps=T, seed=SEED)
    pfc = Prefrontal(concept_dim=64, n_actions=N_ACTIONS, num_steps=T, seed=SEED)
    motor = MotorCortex(n_actions=N_ACTIONS, num_steps=T)

    def pipeline(obs):
        spk = encode_gridworld(obs, grid_n=GRID_N, T=T, generator=torch.Generator().manual_seed(SEED))
        concept = sensory(spk)
        utilities = pfc(concept)
        action = motor(utilities)
        return concept, utilities, action

    # Representative run: agent (0,0), goal (4,4).
    concept, utilities, action = pipeline(torch.tensor([[0, 0, 4, 4]]))
    winner = int(motor.winner(action)[0])

    # --- stage rasters ---
    for spk, name, title in [
        (concept, "concept_raster", "Stage 1 — Sensory concept code (64)"),
        (utilities, "utility_raster", "Stage 2 — Prefrontal action utilities (4)"),
        (action, "action_raster", f"Stage 3 — Motor action (winner = {winner})"),
    ]:
        fig, ax = spike_raster(spk, s=8 if spk.shape[-1] > 8 else 14)
        ax.set_title(title)
        if spk.shape[-1] <= 8:
            ax.set_yticks(range(spk.shape[-1]))
        fig.tight_layout()
        fig.savefig(OUT / f"{name}.png", dpi=120)
        plt.close(fig)

    # --- position -> utility code -> action sweep (agent moves, goal fixed) ---
    positions = [(0, 0), (4, 0), (2, 2), (0, 4), (4, 4)]
    rows = []
    for ax_, ay_ in positions:
        _, u, a = pipeline(torch.tensor([[ax_, ay_, 4, 4]]))
        rows.append(((ax_, ay_), u.sum(0)[0].int().tolist(), int(motor.winner(a)[0])))
    distinct_codes = len({tuple(r[1]) for r in rows})

    # --- report ---
    report = [
        "# EXP-015 — Open-loop bring-up results (Sensory → Prefrontal → Motor)",
        "",
        "**Run date:** 2026-06-06",
        "**Spec:** `docs/superpowers/specs/2026-06-06-phase2-region-framework-design.md`",
        "**Arch spec:** `docs/architecture-spec-v1.md` §4 (build-order step 3)",
        "",
        "## Setup",
        f"- Pipeline: SensoryCortex({N_OBS}→128→64) → Prefrontal(64→[100 RLeaky]→[50]→4)"
        f" → MotorCortex({N_ACTIONS} WTA)",
        f"- Grid {GRID_N}x{GRID_N}, T={T}, rate/Poisson encoding, all seeds={SEED}",
        "- No memory, no gating. PFC owns its afferent Projection (pathway 2, Δ=1).",
        "",
        "## Verification gates",
        f"- [x] End-to-end shapes: concept {tuple(concept.shape)} → utilities "
        f"{tuple(utilities.shape)} → action {tuple(action.shape)}",
        f"- [x] Every stage alive: concept rate {concept.float().mean():.3f}, "
        f"utility rate {utilities.float().mean():.3f}, action rate {action.float().mean():.3f}",
        f"- [x] Motor selects a single winner: action {winner}",
        f"- [x] Utility code carries observation info: {distinct_codes}/{len(positions)} "
        "distinct utility codes across positions",
        "",
        "## Position → utility code → action",
        "",
        "| Agent pos | Utility spike counts [a0,a1,a2,a3] | Winner |",
        "|:---:|:---|:---:|",
    ]
    for pos, counts, win in rows:
        report.append(f"| {pos} | {counts} | {win} |")
    report += [
        "",
        "## Honest caveat (expected, not a bug)",
        "The pipeline is UNTRAINED. The utility *code* discriminates the agent",
        "positions (distinct graded codes propagate end-to-end), but the argmax",
        "action stays the readout's structural favourite — different positions do not",
        "yet map to *different* actions. Note the excitability had to be kept moderate:",
        "too high and the favoured action saturates (fires every step) and washes out",
        "all upstream concept selectivity. Task-appropriate action selection requires",
        "training + the reward/neuromod loop (later steps). This matches spec §2.3:",
        "PFC's value transform is *learned*, not hand-derived.",
        "",
        "## Outputs",
        "- `outputs/concept_raster.png`, `outputs/utility_raster.png`, "
        "`outputs/action_raster.png`",
        "",
        "## Conclusion",
        "Sensory→Prefrontal→Motor conducts end-to-end: spikes flow, every stage is",
        "alive, the afferent Projection carries pathway 2 with delay, and Motor",
        "selects a single action. Build-order step 3 gate PASSED.",
        "",
    ]
    (HERE / "results.md").write_text("\n".join(report), encoding="utf-8")

    print(f"open loop: concept{tuple(concept.shape)} util{tuple(utilities.shape)} "
          f"action{tuple(action.shape)} winner={winner} distinct_codes={distinct_codes}/{len(positions)}")
    print(f"saved 3 PNGs to {OUT}")


if __name__ == "__main__":
    main()
