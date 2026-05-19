"""Section 4 — RSynaptic API smoke check.

snn.RSynaptic adds an explicit synaptic current state on top of RLeaky:

    I_syn[t+1] = alpha * I_syn[t] + V * S_out[t] + I_in[t+1]
    U[t+1]     = beta  * U[t]     + I_syn[t+1] - R(.)

Two time constants (alpha, beta) instead of one. This script just confirms the
three-state forward pass (spk, syn, mem) runs end-to-end, then plots an
RLeaky vs RSynaptic comparison on the same input pulse so the qualitative
effect of the extra time constant is visible.

No full sweep here — RLeaky already taught the recurrence story in section 2;
this is the "skim and move on" check the week brief asked for.

Usage:
    python section4_rsynaptic_check.py
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
ALPHA = 0.9            # synaptic-current decay (RSynaptic only)
THRESHOLD = 1.0
NUM_STEPS = 100
PULSE_STEPS = 10
PULSE_AMPLITUDE = 1.5
V = 0.7                # below RLeaky's memory regime — see the effect of I_syn


# ---- runners -------------------------------------------------------------

def run_rleaky(input_train: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    rlif = snn.RLeaky(beta=BETA, all_to_all=False, V=V,
                      learn_recurrent=False, threshold=THRESHOLD)
    spk, mem = rlif.init_rleaky()
    spk_rec, mem_rec = [], []
    for step in range(NUM_STEPS):
        spk, mem = rlif(input_train[step], spk, mem)
        spk_rec.append(spk); mem_rec.append(mem)
    return torch.stack(spk_rec).reshape(NUM_STEPS, 1), torch.stack(mem_rec).reshape(NUM_STEPS, 1)


def run_rsynaptic(input_train: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rsyn = snn.RSynaptic(alpha=ALPHA, beta=BETA, all_to_all=False, V=V,
                         learn_recurrent=False, threshold=THRESHOLD)
    spk, syn, mem = rsyn.init_rsynaptic()
    spk_rec, syn_rec, mem_rec = [], [], []
    for step in range(NUM_STEPS):
        spk, syn, mem = rsyn(input_train[step], spk, syn, mem)
        spk_rec.append(spk); syn_rec.append(syn); mem_rec.append(mem)
    return (
        torch.stack(spk_rec).reshape(NUM_STEPS, 1),
        torch.stack(syn_rec).reshape(NUM_STEPS, 1),
        torch.stack(mem_rec).reshape(NUM_STEPS, 1),
    )


# ---- main ----------------------------------------------------------------

def main() -> None:
    torch.manual_seed(SEED)
    OUTPUTS.mkdir(exist_ok=True)

    input_train = torch.zeros(NUM_STEPS, 1)
    input_train[:PULSE_STEPS] = PULSE_AMPLITUDE

    spk_r, mem_r = run_rleaky(input_train)
    spk_s, syn_s, mem_s = run_rsynaptic(input_train)

    # ---- API smoke assertions
    expected_shape = (NUM_STEPS, 1)
    assert tuple(spk_s.shape) == expected_shape, f"spk shape {spk_s.shape}"
    assert tuple(syn_s.shape) == expected_shape, f"syn shape {syn_s.shape}"
    assert tuple(mem_s.shape) == expected_shape, f"mem shape {mem_s.shape}"
    print(f"RSynaptic API ok — shapes spk={tuple(spk_s.shape)}, "
          f"syn={tuple(syn_s.shape)}, mem={tuple(mem_s.shape)}")

    pre_r = int(spk_r[:PULSE_STEPS].sum().item())
    post_r = int(spk_r[PULSE_STEPS:].sum().item())
    pre_s = int(spk_s[:PULSE_STEPS].sum().item())
    post_s = int(spk_s[PULSE_STEPS:].sum().item())
    print("\nSpike counts (V=%.2f, alpha=%.2f, beta=%.2f)" % (V, ALPHA, BETA))
    print(f"{'neuron':>10}  {'pre':>5}  {'post':>5}")
    print(f"{'RLeaky':>10}  {pre_r:>5d}  {post_r:>5d}")
    print(f"{'RSynaptic':>10}  {pre_s:>5d}  {post_s:>5d}")

    # ---- comparison plot: membrane, raster, and (for RSynaptic) I_syn
    fig, axes = plt.subplots(3, 2, figsize=(12, 8), sharex=True)

    membrane_trace(mem_r, ax=axes[0, 0], threshold=THRESHOLD)
    axes[0, 0].axvline(PULSE_STEPS, color="red", linestyle=":", label="input off")
    axes[0, 0].set_title(f"RLeaky V={V}  —  membrane")
    axes[0, 0].legend(loc="upper right", fontsize=8)

    spike_raster(spk_r, ax=axes[0, 1])
    axes[0, 1].axvline(PULSE_STEPS, color="red", linestyle=":")
    axes[0, 1].set_title(f"RLeaky V={V}  —  spikes")

    membrane_trace(mem_s, ax=axes[1, 0], threshold=THRESHOLD)
    axes[1, 0].axvline(PULSE_STEPS, color="red", linestyle=":", label="input off")
    axes[1, 0].set_title(f"RSynaptic V={V}, alpha={ALPHA}  —  membrane")
    axes[1, 0].legend(loc="upper right", fontsize=8)

    spike_raster(spk_s, ax=axes[1, 1])
    axes[1, 1].axvline(PULSE_STEPS, color="red", linestyle=":")
    axes[1, 1].set_title(f"RSynaptic V={V}, alpha={ALPHA}  —  spikes")

    # I_syn — the extra state that RSynaptic adds
    axes[2, 0].plot(syn_s.reshape(-1).numpy(), linewidth=1.2, color="C2")
    axes[2, 0].axvline(PULSE_STEPS, color="red", linestyle=":", label="input off")
    axes[2, 0].set_xlabel("Time step")
    axes[2, 0].set_ylabel(r"Synaptic current $I_{syn}[t]$")
    axes[2, 0].set_title(f"RSynaptic V={V}, alpha={ALPHA}  —  I_syn")
    axes[2, 0].grid(alpha=0.3)
    axes[2, 0].legend(loc="upper right", fontsize=8)

    axes[2, 1].axis("off")
    axes[2, 1].text(
        0.05, 0.5,
        "RSynaptic adds the synaptic-current\n"
        "low-pass between input/feedback and\n"
        "membrane. With alpha=beta=0.9 here\n"
        "you should see the post-pulse decay\n"
        "smoother than RLeaky's.",
        fontsize=10, va="center",
    )

    fig.suptitle(
        f"RLeaky vs RSynaptic  (beta={BETA}, alpha={ALPHA}, V={V}, "
        f"pulse={PULSE_STEPS} steps @ {PULSE_AMPLITUDE})",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    out_path = OUTPUTS / "section4_rsynaptic_check.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"\nSaved: {out_path.relative_to(HERE.parent.parent)}")
    plt.close(fig)


if __name__ == "__main__":
    main()
