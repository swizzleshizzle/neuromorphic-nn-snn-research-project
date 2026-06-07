# EXP-015 — Open-loop bring-up results (Sensory → Prefrontal → Motor)

**Run date:** 2026-06-06
**Spec:** `docs/superpowers/specs/2026-06-06-phase2-region-framework-design.md`
**Arch spec:** `docs/architecture-spec-v1.md` §4 (build-order step 3)

## Setup
- Pipeline: SensoryCortex(50→128→64) → Prefrontal(64→[100 RLeaky]→[50]→4) → MotorCortex(4 WTA)
- Grid 5x5, T=32, rate/Poisson encoding, all seeds=0
- No memory, no gating. PFC owns its afferent Projection (pathway 2, Δ=1).

## Verification gates
- [x] End-to-end shapes: concept (32, 1, 64) → utilities (32, 1, 4) → action (32, 1, 4)
- [x] Every stage alive: concept rate 0.413, utility rate 0.133, action rate 0.180
- [x] Motor selects a single winner: action 2
- [x] Utility code carries observation info: 5/5 distinct utility codes across positions

## Position → utility code → action

| Agent pos | Utility spike counts [a0,a1,a2,a3] | Winner |
|:---:|:---|:---:|
| (0, 0) | [0, 1, 16, 0] | 2 |
| (4, 0) | [0, 0, 19, 0] | 2 |
| (2, 2) | [0, 0, 10, 1] | 2 |
| (0, 4) | [0, 1, 18, 0] | 2 |
| (4, 4) | [0, 0, 30, 0] | 2 |

## Honest caveat (expected, not a bug)
The pipeline is UNTRAINED. The utility *code* discriminates the agent
positions (distinct graded codes propagate end-to-end), but the argmax
action stays the readout's structural favourite — different positions do not
yet map to *different* actions. Note the excitability had to be kept moderate:
too high and the favoured action saturates (fires every step) and washes out
all upstream concept selectivity. Task-appropriate action selection requires
training + the reward/neuromod loop (later steps). This matches spec §2.3:
PFC's value transform is *learned*, not hand-derived.

## Outputs
- `outputs/concept_raster.png`, `outputs/utility_raster.png`, `outputs/action_raster.png`

## Conclusion
Sensory→Prefrontal→Motor conducts end-to-end: spikes flow, every stage is
alive, the afferent Projection carries pathway 2 with delay, and Motor
selects a single action. Build-order step 3 gate PASSED.
