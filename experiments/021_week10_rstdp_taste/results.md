# EXP-021 — R-STDP first taste results

**Run date:** 2026-06-14
**Arch spec:** `docs/architecture-spec-v2.md` §6 (R-STDP three-factor), §2.3 (PFC limit).
**Goal:** Week 10 stretch — *one trainable synapse learns from reward, moving selection past the fixed structural favourite.*

## Setup
- Trainable synapse: the PFC utility read-out `fc_utility` only; everything upstream (Sensory→state→transform) frozen — the read-out is the lone learner (one-off; the shared `Projection` is untouched).
- Rule: `Δw = β·dopamine·e`, β=0.05, eligibility `e` STDP-fed (pre transform × post depolarisation, decay τ_e=0.9), normalised; weights clipped to ±1.5.
- Dopamine = `R − b` on the `NeuromodBus` (third factor); baseline `b` EMA α=0.15.
- Toy task: teach action 1 (a non-favourite); ε-greedy exploration ε=0.5; 150 trials; seed=0. Untrained favourite = action 2.

## Results

| measure | start | end |
|---|---|---|
| target (action 1) utility — spikes/32 | 1 | 31 |
| favourite (action 2) utility — spikes/32 | 16 | 0 |
| target read-out ‖w‖ | 1.94 | 5.06 |
| greedy selection (argmax) | action 2 | action 1 |

## Verification gates
- [x] Target read-out weights grow: ‖w‖ 1.94 → 5.06.
- [x] Target utility rises: spike count 1 → 31 (> 0).
- [x] Selection moves in the reward direction: greedy argmax flips from the structural favourite (action 2) to the rewarded target (action 1) — past the EXP-015 fixed-favourite limit.
- [x] Reward-sign tracking: the unrewarded favourite is depressed (utility 16 → 0) by negative dopamine on unrewarded exploration.

## Outputs
- `outputs/utility_learning_curve.png` — target vs favourite utility (spike count) over trials.
- `outputs/target_weight_norm.png` — target read-out weight norm growing under reward.

## Conclusion
R-STDP taste PASSED. A single reward-modulated read-out synapse, fed a normalised eligibility trace and gated by the dopamine third factor, learns to select a target action the untrained network never favoured: the target's weights grow, its utility rises, the old favourite is depressed, and greedy selection flips to the target. This is the first evidence the network can move selection in the reward direction — the headline next step out of the untrained regime (§6, §7). Kept a one-off; baking the eligibility trace into the shared `Projection` is the next build.
