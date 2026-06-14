# EXP-020 — Week-10 closed loop results

**Run date:** 2026-06-14
**Arch spec:** `docs/architecture-spec-v2.md` §2.3 (PFC multi-source), §2.1/§2.2 (Sensory→Hippo store), §3 pathways 2/3/4/5.
**Goal:** Week 10 build day — *wire the upgrades; recall must shift PFC utilities.*

## Setup
- Regions: Sensory(50→64), Hippocampus(150, one-shot Hebbian), PFC(two summed afferents, recall_dim=64), Router, Motor — all seed=0.
- Observation: agent (0, 0), goal (4, 4) on a 5×5 grid; window T=32.
- **Store content = the Sensory concept snapshot** (not PFC output), gated into the attractor during the router store window t=(0, 6) (pathway 3).
- **Recall ON vs OFF**: ON feeds the gated hippocampal recall into PFC's memory afferent (pathway 4); OFF closes the gate (`recall=None` → zeros, the EXP-015 sensory-only path).

## Results

| stage | measure | value |
|---|---|---|
| Sensory | concept spikes | 845 |
| Hippocampus | stored pattern | 30/150 |
| Hippocampus | held / leak (late window) | 1.00 / 0.00 |
| PFC | utility counts — recall ON | [0, 0, 18, 0] |
| PFC | utility counts — recall OFF | [0, 1, 16, 0] |
| PFC | recall-driven shift (L1, actions changed) | 3 over 2 |
| Motor | winner (counts) | action 2 ([0, 0, 12, 0]) |

## Verification gates
- [x] Every stage emits spikes (Sensory, Hippo population, recall, PFC utility, Motor).
- [x] Store from Sensory is content-specific: held=1.00 (≥0.9), leak=0.00 (≤0.1) — the sensory snapshot imprints a clean attractor (reuses the EXP-017/019 recall metric).
- [x] Recall shifts PFC utilities: ON vs OFF differ by L1=3 across 2 action(s) — integrating the gated memory measurably moves the output (payoff of Task 1 + Task 3).
- [x] Motor still selects a single winner through the router-gated pathway 5: action 2 dominates.

## Outputs
- `outputs/sensory_concept_raster.png` — the sensory concept code entering (pathway 2).
- `outputs/hippo_attractor_raster.png` — the attractor imprinted from the sensory snapshot.
- `outputs/pfc_utility_raster.png` — PFC utility, recall ON vs OFF (the shift).
- `outputs/motor_winner_raster.png` — the single Motor winner after router gating.

## Conclusion
Closed loop PASSED. The hippocampus stores the **Sensory** snapshot under router gating and holds it as a clean attractor; opening the recall gate feeds that memory into PFC's second afferent and measurably shifts the utility code versus the sensory-only case, while the router still resolves a single Motor winner. This closes pathway 4 wiring and the pathway 3 content-source rewire, end-to-end, on grid-world. (Untrained — selection is not yet task-meaningful; reward-modulated plasticity is the next step, EXP-021/§6.)
