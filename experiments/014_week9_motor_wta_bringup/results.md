# EXP-014 — Motor Cortex + WTA bring-up results

**Run date:** 2026-06-06
**Spec:** `docs/superpowers/specs/2026-06-06-phase2-region-framework-design.md`
**Arch spec:** `docs/architecture-spec-v1.md` §2.4, §4 (build-order step 2)

## Setup
- Region: MotorCortex, 4 actions, Leaky β=0.9 thr=1.0, T=32
- WTA: action-aligned input drive + lateral inhibition (default strength 3.0)
- Decompression stage deferred (WTA-focused bring-up; needs training)

## Verification gates
- [x] Forward shape: input (32, 1, 4) → action (32, 1, 4)
- [x] Single winner: action 2 = argmax utility ([0.2, 0.2, 0.9, 0.2]); spike counts [0, 0, 31, 1]
- [x] Near one-hot: winner share = 0.97 of all output spikes

## Inhibition sweep (close competitors, rates=[0.6, 0.9, 0.55, 0.5])

| Inhibition | Output spike counts | Winner share |
|---:|:---|---:|
| 0.0 | [28, 32, 29, 28] | 0.27 |
| 1.0 | [6, 25, 10, 3] | 0.57 |
| 3.0 | [1, 19, 6, 1] | 0.70 |
| 5.0 | [1, 23, 2, 1] | 0.85 |

Winner share rises monotonically with inhibition — lateral inhibition
sharpens the competition as designed (ACh-gain hook scales this further).

## Outputs
- `outputs/input_raster.png` — candidate action utilities
- `outputs/action_raster.png` — near one-hot WTA output
- `outputs/inhibition_matrix.png` — lateral inhibition weight matrix

## Conclusion
Motor Cortex selects a single winning action from candidate utilities via
lateral inhibition. Build-order step 2 gate PASSED — ready to wire
Sensory→Prefrontal→Motor open loop (step 3).
