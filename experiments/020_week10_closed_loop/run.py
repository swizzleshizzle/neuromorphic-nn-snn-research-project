"""EXP-020 — Week-10 closed loop: Sensory → PFC + gated Hippocampus recall.

Wires the Week-10 region upgrades end-to-end and verifies the payoff:

- **pathway 2** Sensory concept → PFC (driver, sensory afferent),
- **pathway 3** Sensory concept → Hippocampus **store** content, router-gated — the
  stored snapshot is the *sensory* concept, delivered directly (spec §2.1/§2.2; the
  store content originates from Sensory, *not* PFC),
- **pathway 4** Hippocampus recall → PFC **second afferent**, router-gated (spec §2.3).

The load-bearing comparison is **recall ON vs OFF**: with the recall gate open the
hippocampal memory reaches PFC's memory afferent and shifts the utility code; with it
closed (recall=None → zeros) PFC reproduces the EXP-015 sensory-only output. That the
two differ is the integration payoff of Task 1 (PFC multi-source) + Task 3
(sensory-snapshot store). The loop is untrained, so the winning action is not yet
task-meaningful — the gate is spike flow, content-specific memory, and a recall-driven
shift, with the router resolving a single Motor winner.

Run (Windows, repo root, venv active):
    python experiments/020_week10_closed_loop/run.py
"""

from __future__ import annotations

import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from neuromorphic.connections import apply_gate  # noqa: E402
from neuromorphic.regions import (  # noqa: E402
    Hippocampus,
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
CONTENT = 64
N_HIPPO = 150
N_ACTIONS = 4
T = 32
SEED = 0
OBS = torch.tensor([[0, 0, 4, 4]])   # agent at (0,0), goal at (4,4)
STORE_WINDOW = (0, 6)                 # router store/recall command: gate open here
HELD_PASS = 0.90                      # stored-pattern late-window recovery threshold
LEAK_PASS = 0.10                      # max tolerated off-pattern activation


def gate_closed_window(lo, hi, n):
    """Router store/recall command as a [T,1,n] gate: open (0) inside [lo,hi)."""
    g = torch.ones(T, 1, n)
    g[lo:hi] = 0.0
    return g


def held_leak(population, pat):
    """Late-window (post-cue) mean rate on / off the stored pattern."""
    late = population[T // 2:].float().mean(dim=0)[0]
    return late[pat].mean().item(), late[~pat].mean().item()


def main() -> None:
    OUT.mkdir(exist_ok=True)
    gen = torch.Generator().manual_seed(SEED)

    sensory = SensoryCortex(n_obs=N_OBS, concept=CONTENT, num_steps=T, seed=SEED)
    pfc = Prefrontal(concept_dim=CONTENT, recall_dim=CONTENT, n_actions=N_ACTIONS, num_steps=T, seed=SEED)
    hippo = Hippocampus(content_dim=CONTENT, n_neurons=N_HIPPO, num_steps=T, seed=SEED)
    router = ThalamicRouter(n_actions=N_ACTIONS, num_steps=T)
    motor = MotorCortex(n_actions=N_ACTIONS, num_steps=T)

    # --- pathway 2: Sensory concept code ---
    sensory.enable_recording(True)
    concept = sensory(encode_gridworld(OBS, grid_n=GRID_N, T=T, generator=gen))
    sensory.enable_recording(False)

    # --- pathway 3: store the (gated) Sensory snapshot into the attractor ---
    store_gate = gate_closed_window(*STORE_WINDOW, CONTENT)
    snapshot = apply_gate(concept, store_gate).mean(dim=0)   # [B, CONTENT] sensory snapshot
    pat = hippo.store(snapshot).bool()                       # one-shot Hebbian imprint
    cue = apply_gate(concept, store_gate)                    # gated sensory cue drives the attractor
    hippo.enable_recording(True)
    recall = hippo(cue)                                      # [T,1,CONTENT] recall code
    population = hippo.get_recording("population")
    hippo.enable_recording(False)

    # --- pathway 4: gated recall → PFC second afferent. ON vs OFF. ---
    util_on = pfc(concept, recall_spikes=recall)
    util_off = pfc(concept, recall_spikes=None)              # gate closed → sensory only
    counts_on = util_on.sum(dim=0)[0]
    counts_off = util_off.sum(dim=0)[0]

    # --- pathway 5: router gates PFC→Motor; Motor resolves a winner ---
    gate5 = router(util_on)
    action = motor(apply_gate(util_on, gate5))
    motor_counts = action.sum(dim=0)[0]
    winner = int(motor_counts.argmax())

    # --- metrics ---
    held, leak = held_leak(population, pat)
    util_l1 = (counts_on - counts_off).abs().sum().item()
    n_changed = int((counts_on != counts_off).sum())

    # --- verification gates ---
    g_alive = bool(
        concept.sum() > 0 and population.sum() > 0 and recall.sum() > 0
        and util_on.sum() > 0 and action.sum() > 0
    )
    g_store = held >= HELD_PASS and leak <= LEAK_PASS
    g_shift = util_l1 > 0
    g_winner = bool(
        motor_counts[winner] > 0
        and motor_counts[winner] > motor_counts.sum() - motor_counts[winner]
    )
    all_pass = g_alive and g_store and g_shift and g_winner

    # --- viz: concept, attractor, PFC utility (ON vs OFF), motor winner ---
    fig, ax = spike_raster(concept, s=4)
    ax.set_title("Sensory concept code — pathway 2 (obs agent(0,0) goal(4,4))")
    fig.tight_layout(); fig.savefig(OUT / "sensory_concept_raster.png", dpi=120); plt.close(fig)

    fig, ax = spike_raster(population, s=6)
    ax.set_title(f"Hippocampus attractor — {int(pat.sum())}/{N_HIPPO} from the sensory snapshot")
    fig.tight_layout(); fig.savefig(OUT / "hippo_attractor_raster.png", dpi=120); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
    for a, util, lbl in ((axes[0], util_on, "recall ON"), (axes[1], util_off, "recall OFF")):
        spike_raster(util, ax=a, s=20)
        a.set_title(f"PFC utility — {lbl}")
    fig.tight_layout(); fig.savefig(OUT / "pfc_utility_raster.png", dpi=120); plt.close(fig)

    fig, ax = spike_raster(action, s=20)
    ax.set_title(f"Motor winner — action {winner} (router-gated pathway 5)")
    fig.tight_layout(); fig.savefig(OUT / "motor_winner_raster.png", dpi=120); plt.close(fig)

    # --- results.md (EXP-019 house style) ---
    status = "PASSED" if all_pass else "FAILED — review"
    report = [
        "# EXP-020 — Week-10 closed loop results",
        "",
        f"**Run date:** {datetime.date.today().isoformat()}",
        "**Arch spec:** `docs/architecture-spec-v2.md` §2.3 (PFC multi-source), "
        "§2.1/§2.2 (Sensory→Hippo store), §3 pathways 2/3/4/5.",
        "**Goal:** Week 10 build day — *wire the upgrades; recall must shift PFC utilities.*",
        "",
        "## Setup",
        f"- Regions: Sensory({N_OBS}→{CONTENT}), Hippocampus({N_HIPPO}, one-shot Hebbian), "
        f"PFC(two summed afferents, recall_dim={CONTENT}), Router, Motor — all seed={SEED}.",
        f"- Observation: agent {tuple(OBS[0, :2].tolist())}, goal {tuple(OBS[0, 2:].tolist())} "
        f"on a {GRID_N}×{GRID_N} grid; window T={T}.",
        f"- **Store content = the Sensory concept snapshot** (not PFC output), gated into the "
        f"attractor during the router store window t={STORE_WINDOW} (pathway 3).",
        "- **Recall ON vs OFF**: ON feeds the gated hippocampal recall into PFC's memory afferent "
        "(pathway 4); OFF closes the gate (`recall=None` → zeros, the EXP-015 sensory-only path).",
        "",
        "## Results",
        "",
        "| stage | measure | value |",
        "|---|---|---|",
        f"| Sensory | concept spikes | {int(concept.sum())} |",
        f"| Hippocampus | stored pattern | {int(pat.sum())}/{N_HIPPO} |",
        f"| Hippocampus | held / leak (late window) | {held:.2f} / {leak:.2f} |",
        f"| PFC | utility counts — recall ON | {[int(c) for c in counts_on.tolist()]} |",
        f"| PFC | utility counts — recall OFF | {[int(c) for c in counts_off.tolist()]} |",
        f"| PFC | recall-driven shift (L1, actions changed) | {util_l1:.0f} over {n_changed} |",
        f"| Motor | winner (counts) | action {winner} ({[int(c) for c in motor_counts.tolist()]}) |",
        "",
        "## Verification gates",
        f"- [{'x' if g_alive else ' '}] Every stage emits spikes (Sensory, Hippo population, "
        "recall, PFC utility, Motor).",
        f"- [{'x' if g_store else ' '}] Store from Sensory is content-specific: held={held:.2f} "
        f"(≥{HELD_PASS}), leak={leak:.2f} (≤{LEAK_PASS}) — the sensory snapshot imprints a clean "
        "attractor (reuses the EXP-017/019 recall metric).",
        f"- [{'x' if g_shift else ' '}] Recall shifts PFC utilities: ON vs OFF differ by L1={util_l1:.0f} "
        f"across {n_changed} action(s) — integrating the gated memory measurably moves the output "
        "(payoff of Task 1 + Task 3).",
        f"- [{'x' if g_winner else ' '}] Motor still selects a single winner through the router-gated "
        f"pathway 5: action {winner} dominates.",
        "",
        "## Outputs",
        "- `outputs/sensory_concept_raster.png` — the sensory concept code entering (pathway 2).",
        "- `outputs/hippo_attractor_raster.png` — the attractor imprinted from the sensory snapshot.",
        "- `outputs/pfc_utility_raster.png` — PFC utility, recall ON vs OFF (the shift).",
        "- `outputs/motor_winner_raster.png` — the single Motor winner after router gating.",
        "",
        "## Conclusion",
        f"Closed loop {status}. The hippocampus stores the **Sensory** snapshot under router gating "
        "and holds it as a clean attractor; opening the recall gate feeds that memory into PFC's "
        "second afferent and measurably shifts the utility code versus the sensory-only case, while "
        "the router still resolves a single Motor winner. This closes pathway 4 wiring and the "
        "pathway 3 content-source rewire, end-to-end, on grid-world. (Untrained — selection is not "
        "yet task-meaningful; reward-modulated plasticity is the next step, EXP-021/§6.)",
        "",
    ]
    (HERE / "results.md").write_text("\n".join(report), encoding="utf-8")

    print(f"concept={int(concept.sum())} pattern={int(pat.sum())}/{N_HIPPO} "
          f"held={held:.2f} leak={leak:.2f}")
    print(f"util ON={[int(c) for c in counts_on.tolist()]} "
          f"OFF={[int(c) for c in counts_off.tolist()]} L1={util_l1:.0f} changed={n_changed}")
    print(f"motor winner=action {winner} counts={[int(c) for c in motor_counts.tolist()]}")
    print(f"gates: alive={g_alive} store={g_store} shift={g_shift} winner={g_winner} => {status}")
    print(f"saved 4 PNGs to {OUT}")


if __name__ == "__main__":
    main()
