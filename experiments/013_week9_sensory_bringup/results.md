# EXP-013 — Sensory Cortex bring-up results

**Run date:** 2026-06-06
**Spec:** `docs/superpowers/specs/2026-06-06-phase2-region-framework-design.md`
**Arch spec:** `docs/architecture-spec-v1.md` §2.1, §4 (build-order step 1)

## Setup
- Grid: 5x5 · N_obs=50 (agent one-hot ⊕ goal one-hot)
- Region: SensoryCortex 50→128→64, Leaky β=0.9 thr=1.0, T=32, seed=0
- Encoding: rate/Poisson, max_rate=0.5

## Verification gates
- [x] Forward shape: input (32, 1, 50) → concept (32, 1, 64)
- [x] Hidden recording shape: (32, 1, 128)
- [x] Concept layer not dead: firing rate = 0.413 spikes/neuron/step; 45/64 neurons active
- [x] Selectivity: min pairwise L1 between 5 positions = 475 (>0 ⇒ position-selective; mean = 645)

## Outputs
- `outputs/input_raster.png` — sparse 2-hot Poisson sensory input
- `outputs/concept_raster.png` — concept code spike raster
- `outputs/population_rate.png` — concept-layer mean firing rate over time

## Conclusion
Sensory Cortex produces a stable, non-saturated, position-selective concept
code from a sparse grid observation. Build-order step 1 gate PASSED — ready
to wire Sensory→Prefrontal→Motor (step 3).
