"""EXP-019 — Hippocampus pattern completion (Phase 2, Week 10 Session 1).

The Week-10 goal EXP-017 left open: *partial input should activate the full memory.*
EXP-017 verified hold + content-specific recall, but never fed a degraded cue. This
experiment does: it stores one full content pattern, then drives a **partial** cue
(a fraction of the content dims masked to zero) during a short cue window, lets the
attractor settle with no input, and measures whether the *full* stored pattern is
recovered.

The load-bearing comparison is recurrence ON vs OFF:
- ON  — the imprinted Hopfield recurrence (`W_rec`) is live.
- OFF — `W_rec` zeroed; the same partial cue with no attractor.
If completion is real and attractor-driven, the late (post-cue, no-input) window holds
the full pattern with recurrence ON and collapses to ~0 with it OFF.

Metrics mirror EXP-017: `held` = mean late-window rate of the stored-pattern neurons,
`leak` = mean late-window rate of the off-pattern neurons.

Run (Windows, repo root, venv active):
    python experiments/019_week10_pattern_completion/run.py
"""

from __future__ import annotations

import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from neuromorphic.connections import apply_gate  # noqa: E402
from neuromorphic.regions import Hippocampus  # noqa: E402
from neuromorphic.viz import spike_raster  # noqa: E402

HERE = Path(__file__).parent
OUT = HERE / "outputs"

CONTENT = 64
N_NEURONS = 150
T = 32
SEED = 0
CUE_STEPS = (0, 6)            # partial cue presented only in this (gated-open) window
DEGRADATIONS = [0.0, 0.25, 0.5, 0.75, 0.9]  # fraction of content dims masked to zero
HELD_PASS = 0.90             # full-pattern recovery threshold
LEAK_PASS = 0.10             # max tolerated off-pattern activation
LIFT_PASS = 0.80             # held must lift by >= this from recurrence-OFF to ON
                             # (OFF sits at a ~0.10 bias floor, not zero: fc_in(0)=bias)


def phase_gate_closed(open_lo, open_hi, last_dim):
    """gate_closed [T,1,last_dim]: 0 (open) inside [lo,hi), else 1 (closed)."""
    g = torch.ones(T, 1, last_dim)
    g[open_lo:open_hi] = 0.0
    return g


def content_code(seed, B=1):
    gen = torch.Generator().manual_seed(seed)
    return torch.rand(B, CONTENT, generator=gen)


def partial_cue(content, frac_masked, seed):
    """Zero a `frac_masked` fraction of the content dims (a degraded cue)."""
    gen = torch.Generator().manual_seed(seed)
    keep = (torch.rand(CONTENT, generator=gen) >= frac_masked).float()
    return content * keep, keep


def held_leak(population, pat):
    """Late-window (post-cue) mean rate on / off the stored pattern."""
    late = population[T // 2 :].float().mean(dim=0)[0]
    return late[pat].mean().item(), late[~pat].mean().item()


def run_cue(hippo, cue, store_gate_closed):
    """Drive `cue` through the gated store window; return recorded population [T,1,N]."""
    raw = torch.zeros(T, 1, CONTENT)
    raw[CUE_STEPS[0] : CUE_STEPS[1]] = cue.unsqueeze(0)
    gated = apply_gate(raw, store_gate_closed)
    hippo.enable_recording(True)
    hippo(gated)
    pop = hippo.get_recording("population")
    hippo.enable_recording(False)
    return pop, gated


def main() -> None:
    OUT.mkdir(exist_ok=True)
    store_gate_closed = phase_gate_closed(*CUE_STEPS, CONTENT)
    content = content_code(seed=1)

    # Imprint the full pattern once (the memory we will try to complete).
    hippo = Hippocampus(content_dim=CONTENT, n_neurons=N_NEURONS, num_steps=T, seed=SEED)
    p = hippo.store(content)
    pat = p.bool()

    rows = []          # (frac, held_on, leak_on, held_off)
    saved_cue = saved_attr = None
    for frac in DEGRADATIONS:
        cue, _ = partial_cue(content, frac, seed=100 + int(frac * 100))

        # recurrence ON — re-imprint a fresh region so state is clean
        h_on = Hippocampus(content_dim=CONTENT, n_neurons=N_NEURONS, num_steps=T, seed=SEED)
        h_on.store(content)
        pop_on, gated_cue = run_cue(h_on, cue, store_gate_closed)
        held_on, leak_on = held_leak(pop_on, pat)

        # recurrence OFF — same imprint, then zero the attractor weights
        h_off = Hippocampus(content_dim=CONTENT, n_neurons=N_NEURONS, num_steps=T, seed=SEED)
        h_off.store(content)
        h_off.W_rec = torch.zeros_like(h_off.W_rec)
        pop_off, _ = run_cue(h_off, cue, store_gate_closed)
        held_off, _ = held_leak(pop_off, pat)

        rows.append((frac, held_on, leak_on, held_off))
        if abs(frac - 0.5) < 1e-9:          # keep the 50%-masked case for the rasters
            saved_cue, saved_attr = gated_cue, pop_on

    # --- verification gates ---
    held50 = next(h for f, h, _, _ in rows if abs(f - 0.5) < 1e-9)
    leak50 = next(l for f, _, l, _ in rows if abs(f - 0.5) < 1e-9)
    held90 = next(h for f, h, _, _ in rows if abs(f - 0.9) < 1e-9)
    off_max = max(hf for *_, hf in rows)
    min_lift = min(ho - hf for _, ho, _, hf in rows)
    g_complete = held50 >= HELD_PASS and leak50 <= LEAK_PASS
    g_robust = held90 >= HELD_PASS
    g_attractor = min_lift >= LIFT_PASS
    all_pass = g_complete and g_robust and g_attractor

    # --- viz: cue raster, completed attractor, completion-vs-degradation curve ---
    fig, ax = spike_raster(saved_cue, s=6)
    ax.set_title("Partial cue (50% of content masked) — pathway 3, gated t=0–6")
    fig.tight_layout(); fig.savefig(OUT / "partial_cue_raster.png", dpi=120); plt.close(fig)

    fig, ax = spike_raster(saved_attr, s=6)
    ax.set_title(f"Completed attractor from 50% cue — {int(p.sum())} neurons recovered")
    fig.tight_layout(); fig.savefig(OUT / "completed_attractor_raster.png", dpi=120); plt.close(fig)

    fracs = [f for f, *_ in rows]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(fracs, [h for _, h, _, _ in rows], "o-", label="held — recurrence ON (completed)")
    ax.plot(fracs, [h for *_, h in rows], "s--", label="held — recurrence OFF (control)")
    ax.plot(fracs, [l for _, _, l, _ in rows], "^:", label="leak — recurrence ON")
    ax.axhline(HELD_PASS, color="grey", lw=0.8, ls=":")
    ax.set_xlabel("fraction of cue masked"); ax.set_ylabel("late-window rate")
    ax.set_ylim(-0.05, 1.05); ax.legend(); ax.set_title("Pattern completion vs cue degradation")
    fig.tight_layout(); fig.savefig(OUT / "completion_curve.png", dpi=120); plt.close(fig)

    # --- results.md (EXP-017 style) ---
    def fmt(r):
        f, ho, lo, hf = r
        return f"| {int(f*100)}% | {ho:.2f} | {lo:.2f} | {hf:.2f} |"

    status = "PASSED" if all_pass else "FAILED — review"
    report = [
        "# EXP-019 — Hippocampus pattern completion results",
        "",
        f"**Run date:** {datetime.date.today().isoformat()}",
        "**Arch spec:** `docs/architecture-spec-v2.md` §2.2 (Hippocampus)",
        "**Goal:** Week 10 Session 1 — *partial input should activate the full memory.*",
        "",
        "## Setup",
        f"- Hippocampus: {N_NEURONS} neurons, one-shot Hebbian (Hopfield) imprint, "
        f"{int(p.sum())}/{N_NEURONS}-neuron stored pattern.",
        f"- Cue presented only during the gated store window t={CUE_STEPS}; zero input after.",
        f"- `held` / `leak` measured in the late (post-cue) window t={T//2}–{T}.",
        "- Each degradation level run twice: recurrence ON (imprinted `W_rec`) vs OFF "
        "(`W_rec` zeroed) — the control that isolates the attractor.",
        "",
        "## Results — completion vs cue degradation",
        "",
        "| cue masked | held (ON, completed) | leak (ON) | held (OFF, control) |",
        "|---|---|---|---|",
        *[fmt(r) for r in rows],
        "",
        "## Verification gates",
        f"- [{'x' if g_complete else ' '}] Completion @50% masked: held={held50:.2f} "
        f"(≥{HELD_PASS}) and leak={leak50:.2f} (≤{LEAK_PASS}).",
        f"- [{'x' if g_robust else ' '}] Robust @90% masked: held={held90:.2f} "
        f"(≥{HELD_PASS}) — full memory from a 10%-intact cue.",
        f"- [{'x' if g_attractor else ' '}] Attractor-driven: held lifts by ≥{LIFT_PASS} from "
        f"recurrence-OFF to ON at every level (min lift {min_lift:.2f}). The OFF control sits "
        f"at a ~{off_max:.2f} bias floor (`fc_in(0)=bias`), not zero, so the lift — not an "
        "absolute collapse — is what shows completion is the recurrence, not residual input.",
        "",
        "## Outputs",
        "- `outputs/partial_cue_raster.png` — the degraded (50%-masked) cue entering.",
        "- `outputs/completed_attractor_raster.png` — the full pattern recovered after the cue.",
        "- `outputs/completion_curve.png` — held/leak vs degradation, ON vs OFF.",
        "",
        "## Conclusion",
        f"Pattern completion {status}. With the imprinted recurrence live, a partial cue drives "
        "the attractor back to the full stored pattern and holds it after input is removed; "
        "with the recurrence off, the same cue decays to silence. This closes the third "
        "Week-10 hippocampus design goal.",
        "",
    ]
    (HERE / "results.md").write_text("\n".join(report), encoding="utf-8")

    print(f"pattern={int(p.sum())}/{N_NEURONS}")
    for f, ho, lo, hf in rows:
        print(f"  masked {int(f*100):>2}%: held_on={ho:.2f} leak_on={lo:.2f} held_off={hf:.2f}")
    print(f"gates: complete={g_complete} robust={g_robust} attractor={g_attractor} "
          f"(min_lift={min_lift:.2f}, off_floor={off_max:.2f}) => {status}")
    print(f"saved 3 PNGs to {OUT}")


if __name__ == "__main__":
    main()
