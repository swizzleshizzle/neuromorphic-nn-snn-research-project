"""EXP-021 — R-STDP first taste: reward-modulated plasticity on the PFC readout.

The smallest possible taste of the learning engine designed in L11 (spec §6):
**reward-modulated STDP / three-factor** plasticity on **one** trainable projection —
the PFC utility read-out (`fc_utility`), the synapse closest to reward and the
smallest in the network. Everything upstream (Sensory→PFC state→transform) is frozen,
so the read-out is the only thing that learns; this keeps it a one-off and does *not*
bake plasticity into the shared `Projection`.

The rule (Izhikevich 2007; Frémaux & Gerstner 2016): each synapse carries an
**eligibility trace** `e_ij` (STDP-fed: pre transform activity × post utility
depolarization, slow decay `τ_e`). The dopamine scalar broadcast on the `NeuromodBus`
**is the third factor** `(R − b)` (reward − running baseline). At each trial we apply
`Δw_ij = β · dopamine · e_ij`, with the eligibility credited to the **action taken**
that trial. Eligibility is normalised and weights clipped — bounded synapses, the
biological reality that keeps the positive-feedback Hebbian loop stable.

Toy task: teach PFC to select a **target** action that is *not* its untrained
structural favourite — the EXP-015 limit ("the argmax action is a fixed structural
favourite"). We ε-greedily explore actions and reward only the target; over trials the
target's read-out weights grow, its utility (spike count) rises from zero, the
favourite is depressed (negative dopamine on unrewarded exploration), and greedy
selection flips to the target — selection moving in the reward direction.

Run (Windows, repo root, venv active):
    python experiments/021_week10_rstdp_taste/run.py
"""

from __future__ import annotations

import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import snntorch as snn  # noqa: E402
import torch  # noqa: E402

from neuromorphic.neuromod import NeuromodBus  # noqa: E402
from neuromorphic.regions import Prefrontal, SensoryCortex, encode_gridworld  # noqa: E402

HERE = Path(__file__).parent
OUT = HERE / "outputs"

GRID_N = 5
N_OBS = 2 * GRID_N * GRID_N
CONTENT = 64
N_ACTIONS = 4
T = 32
SEED = 0
OBS = torch.tensor([[0, 0, 4, 4]])

TARGET = 1          # the action we teach: a non-favourite with nonzero eligibility
                    # under the sensory concept (the favourite is action 2)
N_TRIALS = 150
BETA = 0.05         # learning rate β
EPS = 0.5           # ε-greedy exploration (so the silent target gets tried)
TAU_E = 0.9         # eligibility-trace decay
BASELINE_EMA = 0.15  # reward-baseline update rate (b ← (1−α)b + αR)
W_MAX = 1.5         # synaptic bound (clip) — keeps the Hebbian loop stable
THRESHOLD = 1.0
LIF_BETA = 0.9


def readout_trial(weight, bias, pre):
    """Run the PFC utility read-out for one window with the current weights.

    Args:
        weight: ``[N_actions, n_transform]`` read-out weights.
        bias: ``[N_actions]`` read-out bias.
        pre: ``[T, n_transform]`` frozen pre-synaptic transform spikes.

    Returns:
        ``(eligibility [N_actions, n_transform], spike_counts [N_actions])`` — the
        eligibility trace at window end and the per-action spike count.
    """
    lif = snn.Leaky(beta=LIF_BETA, threshold=THRESHOLD, reset_mechanism="subtract")
    mem = lif.init_leaky()
    elig = torch.zeros_like(weight)
    spikes = torch.zeros(weight.shape[0])
    for t in range(T):
        spk, mem = lif(pre[t] @ weight.t() + bias, mem)
        # STDP-fed eligibility: pre (transform spikes) × post (utility depolarisation),
        # graded so a sub-threshold target still accrues credit; slow decay τ_e.
        elig = TAU_E * elig + torch.outer(torch.relu(mem), pre[t])
        spikes += spk
    return elig, spikes


def main() -> None:
    OUT.mkdir(exist_ok=True)
    gen = torch.Generator().manual_seed(SEED)

    # --- frozen feature extractor: Sensory → PFC state/transform → pre spikes ---
    sensory = SensoryCortex(n_obs=N_OBS, concept=CONTENT, num_steps=T, seed=SEED)
    pfc = Prefrontal(concept_dim=CONTENT, n_actions=N_ACTIONS, num_steps=T, seed=SEED)
    concept = sensory(encode_gridworld(OBS, grid_n=GRID_N, T=T, generator=gen))
    pfc.enable_recording(True)
    pfc(concept)
    pre = pfc.get_recording("transform")[:, 0, :]  # [T, n_transform], constant (upstream frozen)
    pfc.enable_recording(False)

    weight = pfc.fc_utility.weight.detach().clone()
    bias = pfc.fc_utility.bias.detach().clone()
    bus = NeuromodBus()

    _, spikes0 = readout_trial(weight, bias, pre)
    favourite = int(spikes0.argmax())          # the untrained structural favourite
    w_norm0 = weight[TARGET].norm().item()

    # --- R-STDP loop ---
    baseline = 0.0
    tgt_spikes, fav_spikes, tgt_wnorm, dopamine_hist = [], [], [], []
    for _ in range(N_TRIALS):
        elig, spikes = readout_trial(weight, bias, pre)

        # ε-greedy action selection (explore so the silent target gets sampled).
        if torch.rand(1, generator=gen).item() < EPS:
            action = int(torch.randint(0, N_ACTIONS, (1,), generator=gen))
        else:
            action = int(spikes.argmax())

        reward = 1.0 if action == TARGET else 0.0
        dopamine = reward - baseline                       # third factor (R − b)
        baseline = (1 - BASELINE_EMA) * baseline + BASELINE_EMA * reward
        bus.set(dopamine=dopamine)

        # Δw = β · dopamine · e, credited to the action taken; bounded synapses.
        elig_n = elig / (elig.abs().max() + 1e-6)
        weight = weight.clone()
        weight[action] = (weight[action] + BETA * bus.dopamine * elig_n[action]).clamp(-W_MAX, W_MAX)

        tgt_spikes.append(int(spikes[TARGET]))
        fav_spikes.append(int(spikes[favourite]))
        tgt_wnorm.append(weight[TARGET].norm().item())
        dopamine_hist.append(dopamine)

    _, spikes_f = readout_trial(weight, bias, pre)
    selected_final = int(spikes_f.argmax())
    w_norm_f = weight[TARGET].norm().item()

    # --- verification gates ---
    g_weights_grow = w_norm_f > w_norm0
    g_utility_rises = int(spikes_f[TARGET]) > int(spikes0[TARGET]) and spikes_f[TARGET] > 0
    # Selection genuinely moved (not a degenerate all-zero argmax): the target must be
    # the strict greedy winner over the old favourite.
    g_selection_moves = (
        favourite != TARGET
        and selected_final == TARGET
        and int(spikes_f[TARGET]) > int(spikes_f[favourite])
    )
    g_favourite_depressed = int(spikes_f[favourite]) < int(spikes0[favourite])
    all_pass = g_weights_grow and g_utility_rises and g_selection_moves

    # --- viz: utility (spike count) learning curve + target weight-norm curve ---
    trials = list(range(1, N_TRIALS + 1))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(trials, tgt_spikes, "-", label=f"target (action {TARGET}) utility")
    ax.plot(trials, fav_spikes, "--", label=f"favourite (action {favourite}) utility")
    ax.set_xlabel("rewarded trial"); ax.set_ylabel("utility spike count")
    ax.set_title("R-STDP: rewarded action's utility rises, favourite depressed")
    ax.legend(); fig.tight_layout()
    fig.savefig(OUT / "utility_learning_curve.png", dpi=120); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(trials, tgt_wnorm, "-", color="tab:green")
    ax.axhline(w_norm0, color="grey", lw=0.8, ls=":", label="start norm")
    ax.set_xlabel("rewarded trial"); ax.set_ylabel(f"target read-out ‖w‖ (action {TARGET})")
    ax.set_title("R-STDP: target read-out weights grow under reward")
    ax.legend(); fig.tight_layout()
    fig.savefig(OUT / "target_weight_norm.png", dpi=120); plt.close(fig)

    # --- results.md (EXP-019 house style) ---
    status = "PASSED" if all_pass else "FAILED — review"
    report = [
        "# EXP-021 — R-STDP first taste results",
        "",
        f"**Run date:** {datetime.date.today().isoformat()}",
        "**Arch spec:** `docs/architecture-spec-v2.md` §6 (R-STDP three-factor), §2.3 (PFC limit).",
        "**Goal:** Week 10 stretch — *one trainable synapse learns from reward, moving "
        "selection past the fixed structural favourite.*",
        "",
        "## Setup",
        "- Trainable synapse: the PFC utility read-out `fc_utility` only; everything upstream "
        "(Sensory→state→transform) frozen — the read-out is the lone learner (one-off; the shared "
        "`Projection` is untouched).",
        f"- Rule: `Δw = β·dopamine·e`, β={BETA}, eligibility `e` STDP-fed (pre transform × post "
        f"depolarisation, decay τ_e={TAU_E}), normalised; weights clipped to ±{W_MAX}.",
        f"- Dopamine = `R − b` on the `NeuromodBus` (third factor); baseline `b` EMA α={BASELINE_EMA}.",
        f"- Toy task: teach action {TARGET} (a non-favourite); ε-greedy exploration ε={EPS}; "
        f"{N_TRIALS} trials; seed={SEED}. Untrained favourite = action {favourite}.",
        "",
        "## Results",
        "",
        "| measure | start | end |",
        "|---|---|---|",
        f"| target (action {TARGET}) utility — spikes/{T} | {int(spikes0[TARGET])} | {int(spikes_f[TARGET])} |",
        f"| favourite (action {favourite}) utility — spikes/{T} | {int(spikes0[favourite])} | {int(spikes_f[favourite])} |",
        f"| target read-out ‖w‖ | {w_norm0:.2f} | {w_norm_f:.2f} |",
        f"| greedy selection (argmax) | action {favourite} | action {selected_final} |",
        "",
        "## Verification gates",
        f"- [{'x' if g_weights_grow else ' '}] Target read-out weights grow: ‖w‖ "
        f"{w_norm0:.2f} → {w_norm_f:.2f}.",
        f"- [{'x' if g_utility_rises else ' '}] Target utility rises: spike count "
        f"{int(spikes0[TARGET])} → {int(spikes_f[TARGET])} (> 0).",
        f"- [{'x' if g_selection_moves else ' '}] Selection moves in the reward direction: greedy "
        f"argmax flips from the structural favourite (action {favourite}) to the rewarded target "
        f"(action {selected_final}) — past the EXP-015 fixed-favourite limit.",
        f"- [{'x' if g_favourite_depressed else ' '}] Reward-sign tracking: the unrewarded favourite "
        f"is depressed (utility {int(spikes0[favourite])} → {int(spikes_f[favourite])}) by negative "
        "dopamine on unrewarded exploration.",
        "",
        "## Outputs",
        "- `outputs/utility_learning_curve.png` — target vs favourite utility (spike count) over trials.",
        "- `outputs/target_weight_norm.png` — target read-out weight norm growing under reward.",
        "",
        "## Conclusion",
        f"R-STDP taste {status}. A single reward-modulated read-out synapse, fed a normalised "
        "eligibility trace and gated by the dopamine third factor, learns to select a target action "
        "the untrained network never favoured: the target's weights grow, its utility rises, "
        "the old favourite is depressed, and greedy selection flips to the target. This is the "
        "first evidence the network can move selection in the reward direction — the headline next "
        "step out of the untrained regime (§6, §7). Kept a one-off; baking the eligibility trace into "
        "the shared `Projection` is the next build.",
        "",
    ]
    (HERE / "results.md").write_text("\n".join(report), encoding="utf-8")

    print(f"favourite=action {favourite} target=action {TARGET}")
    print(f"target utility {int(spikes0[TARGET])} -> {int(spikes_f[TARGET])}; "
          f"favourite {int(spikes0[favourite])} -> {int(spikes_f[favourite])}")
    print(f"target w-norm {w_norm0:.2f} -> {w_norm_f:.2f}; greedy {favourite} -> {selected_final}")
    print(f"gates: weights={g_weights_grow} utility={g_utility_rises} "
          f"selection={g_selection_moves} depressed={g_favourite_depressed} => {status}")
    print(f"saved 2 PNGs to {OUT}")


if __name__ == "__main__":
    main()
