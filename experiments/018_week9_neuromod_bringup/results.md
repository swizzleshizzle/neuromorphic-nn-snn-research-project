# EXP-018 — Neuromodulatory bus bring-up results

**Run date:** 2026-06-06
**Spec:** `docs/superpowers/specs/2026-06-06-phase2-region-framework-design.md`
**Arch spec:** `docs/architecture-spec-v1.md` §1, §4 (build-order step 6)

## Setup
- Loop: Sensory → Prefrontal → Router → gate → **Motor (reads ACh from bus)**
- `NeuromodBus`: dopamine (reward/learning-enable) + ACh (gain/precision),
  broadcast one-to-all.

## Verification gates
- [x] Full gated loop conducts with the bus attached: Motor winner = action 2
- [x] ACh modulates precision (Motor facing competing proposals) —

| ACh | Motor winner share |
|---:|---:|
| 0.3 | 0.55 |
| 1.0 | 0.70 |
| 2.0 | 0.85 |
| 3.0 | 0.81 |

  (winner share rises 0.55 → 0.85 then saturates: higher ACh ⇒ sharper WTA)
- [x] Dopamine broadcast / learning-enable: dopamine 0.0 → learning_enabled=False; dopamine 0.8 → learning_enabled=True

## Outputs
- `outputs/ach_sweep.png` — winner share vs ACh
- `outputs/action_low_ach.png`, `outputs/action_high_ach.png` — Motor rasters

## Honest caveats
- Dopamine has no plasticity to gate yet (no learning in Phase 2) — it is broadcast
  as a global signal with `learning_enabled` exposed as the hook for future STDP /
  reward-modulated learning. ACh's effect is live and measurable today.
- ACh only matters where channels compete. In the fully-gated loop the router has
  already selected one channel, so ACh is moot there — the sweep is therefore run
  on Motor facing competing proposals (the pre-selection / router-open regime).

## Conclusion
The neuromodulatory bus broadcasts dopamine + ACh to the loop; ACh sharpens the
Motor WTA on demand. Build-order step 6 gate PASSED — Phase 2's five regions,
connection primitives, gating, and neuromod bus are all in place.
