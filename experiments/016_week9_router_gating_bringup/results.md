# EXP-016 — Thalamic Router gating bring-up results

**Run date:** 2026-06-06
**Spec:** `docs/superpowers/specs/2026-06-06-phase2-region-framework-design.md`
**Arch spec:** `docs/architecture-spec-v1.md` §2.5, §4 (build-order step 4)

## Setup
- Loop: Sensory → Prefrontal → **Router (gates pathway 5)** → Motor
- Router: 4 channels, Stage A WTA select + Stage B tonic gating,
  do-nothing floor via constant Stage-A bias. Gate applied with `apply_gate`.

## Verification gates
- [x] Stage A selects a winner: channel 2 (open steps per channel = [0, 0, 5, 0])
- [x] Stage B per-channel gating: only the selected channel is disinhibited; closed channels carry no current
- [x] Motor follows the gated pathway: winner = action 2 (= selected)
- [x] Do-nothing veto: below-floor utilities → 0 channels opened → Motor emits 0 spikes (silent)
- [x] mode='off' opens all channels (gated == raw utilities): True

## Outputs
- `outputs/utility_raster.png` — PFC utilities before gating
- `outputs/gate_closed_raster.png` — router control lines (gaps = open channel)
- `outputs/gated_action_raster.png` — Motor output through the gated pathway

## Conclusion
The Thalamic Router selects a channel and gates pathway 5 by disinhibition,
constraining Motor to the selected action and vetoing action when no utility
clears the floor. Build-order step 4 gate PASSED — control signals now steer
the loop. Next: gate pathways 3/4 with the Hippocampus (step 5).
