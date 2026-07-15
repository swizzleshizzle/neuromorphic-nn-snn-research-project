# EXP-027 Results — Encoder Characterization (committed record)

> **Why this file exists:** the raw run artifacts under `outputs/` are gitignored and were
> generated on a laptop over SSH; a Phase-2 checkpoint audit (2026-07-13) correctly flagged that
> the causal-distributedness result existed only in a vault note and could not be verified from the
> repo. This file commits the numbers into the repository so the record is reproducible from
> mainline. Provenance: 12 seeds (0-11), grid 5, run on the laptop 2026-07-09 (Component A) and
> 2026-07-13 (Component B); 12 checkpoints minted. Regenerate with the commands at the bottom.

## Component A — cross-region decodability + geometry (Aim 1: specialization)

Per-region held-out decodability, mean across 12 seeds (chance = each probe's shuffle-null band):

| region | displacement R2 | null | per-action acc | null |
|---|---|---|---|---|
| **sensory concept[64]** | **0.86** | -0.04 | **0.89** | 0.59 |
| **sensory hidden[128]** | **0.90** | -0.22 | 0.89 | 0.57 |
| prefrontal **state[100]** | **0.76** | -0.13 | 0.85 | 0.58 |
| prefrontal utility[4] | 0.03 | 0.01 | 0.60 | 0.59 |
| router gate[4] | 0.01 | 0.00 | 0.59 | 0.59 |
| motor action[4] | -0.00 | 0.00 | 0.59 | 0.59 |
| hippocampus pop[150] | -0.01 | -0.01 | 0.59 | 0.59 (zero-filled, bypassed) |

Cross-region paired win-fraction (sensory concept vs each region, `027_matrix.md`): the sensory
concept beats **every non-sensory spectator** (prefrontal/state/router/motor/hippocampus) on both
targets in **100% of 12 seeds**. It does not beat `sensory_hidden` (0% / 50%) — expected, since the
128-d hidden layer is the richer part of the same sensory hierarchy (R2 0.90 > 0.86). Controls:
shuffle-null band (n=10 label permutations) and PCA-matched k=4,8 to neutralize the width confound.

**Geometry (Aim 2A):** concept participation ratio **~23.6 / 64** effective dims (tight 22.6-25.0
across seeds); **~50%** of the 64 units are needed to reach 90% of full-population decode R2.
Distributed code, not a few load-bearing units.

**Honest gradient (not a strawman):** PFC internal state[100] carries a real lossy image (disp R2
0.76) that attenuates hard at each bottleneck (PFC output utility[4] 0.03 -> router/motor ~0);
hippocampus is a true zero because it is bypassed (`recall=False`), reported as zero-filled.

## Component B — causal dropout-on-navigation (Aim 2B: causal distributedness)

Held-out navigation success under concept-unit masking, mean across 12 seeds (`027_dropout.md`).
Full-population (k=0) baseline held-out success = **34.7%** (per-seed 0-83%):

| k dropped (of 64) | random | top | bottom |
|---|---|---|---|
| **0 (baseline)** | **34.7%** | 34.7% | 34.7% |
| 2 | 33% | 32% | 39% |
| 4 | 33% | 25% | 33% |
| 8 | 32% | 22% | 38% |
| 16 | 30% | 25% | 39% |
| 32 | 24% | 22% | 22% |

**Read:** distributed and causally robust. Random-k degrades gracefully (dropping *half* the 64
units costs only ~11 pts). Importance ordering is real but non-catastrophic — `bottom >= random >=
top` holds at k=2/4/8/16, yet even dropping the 16 most-important units only pulls top-k to 25% (no
small-k cliff). Geometry (2A) and behavior (2B) agree -> the strong form of the distributedness
claim. **Caveat:** absolute success is modest (~35%) with high per-seed variance; this
characterizes *how* the code is represented, it does not raise the navigation cap (that was EXP-026).

## Regenerate

```powershell
# Component A only (~15-25 min):
.venv\Scripts\python.exe experiments\027_encoder_characterization\run.py --seeds 0 1 2 3 4 5 6 7 8 9 10 11
# + Component B causal dropout (mints 12 checkpoints, ~1h):
.venv\Scripts\python.exe experiments\027_encoder_characterization\run.py --seeds 0 1 2 3 4 5 6 7 8 9 10 11 --dropout
```
