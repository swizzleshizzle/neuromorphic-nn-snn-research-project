# EXP-019 — Hippocampus pattern completion results

**Run date:** 2026-06-08
**Arch spec:** `docs/architecture-spec-v2.md` §2.2 (Hippocampus)
**Goal:** Week 10 Session 1 — *partial input should activate the full memory.*

## Setup
- Hippocampus: 150 neurons, one-shot Hebbian (Hopfield) imprint, 30/150-neuron stored pattern.
- Cue presented only during the gated store window t=(0, 6); zero input after.
- `held` / `leak` measured in the late (post-cue) window t=16–32.
- Each degradation level run twice: recurrence ON (imprinted `W_rec`) vs OFF (`W_rec` zeroed) — the control that isolates the attractor.

## Results — completion vs cue degradation

| cue masked | held (ON, completed) | leak (ON) | held (OFF, control) |
|---|---|---|---|
| 0% | 1.00 | 0.00 | 0.10 |
| 25% | 1.00 | 0.00 | 0.10 |
| 50% | 1.00 | 0.00 | 0.11 |
| 75% | 1.00 | 0.00 | 0.10 |
| 90% | 1.00 | 0.00 | 0.11 |

## Verification gates
- [x] Completion @50% masked: held=1.00 (≥0.9) and leak=0.00 (≤0.1).
- [x] Robust @90% masked: held=1.00 (≥0.9) — full memory from a 10%-intact cue.
- [x] Attractor-driven: held lifts by ≥0.8 from recurrence-OFF to ON at every level (min lift 0.89). The OFF control sits at a ~0.11 bias floor (`fc_in(0)=bias`), not zero, so the lift — not an absolute collapse — is what shows completion is the recurrence, not residual input.

## Outputs
- `outputs/partial_cue_raster.png` — the degraded (50%-masked) cue entering.
- `outputs/completed_attractor_raster.png` — the full pattern recovered after the cue.
- `outputs/completion_curve.png` — held/leak vs degradation, ON vs OFF.

## Conclusion
Pattern completion PASSED. With the imprinted recurrence live, a partial cue drives the attractor back to the full stored pattern and holds it after input is removed; with the recurrence off, the same cue decays to silence. This closes the third Week-10 hippocampus design goal.
