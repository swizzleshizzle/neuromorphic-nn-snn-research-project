# EXP-017 — Hippocampus store/recall bring-up results

**Run date:** 2026-06-06
**Spec:** `docs/superpowers/specs/2026-06-06-phase2-region-framework-design.md`
**Arch spec:** `docs/architecture-spec-v1.md` §2.2, §4 (build-order step 5)

## Setup
- Hippocampus: 150 neurons, recurrent attractor, one-shot Hebbian imprint
- Stored pattern sparsity: 30/150 neurons
- Schedule: store gate open t=(0, 6), delay (both closed), recall gate open t=(20, 32)
- Pathways 3 (store) and 4 (recall) gated with `apply_gate` (router-driven in the closed loop).

## Verification gates
- [x] Store imprints attractor: W_rec non-zero, pattern = 30 neurons
- [x] Attractor-persistence: stored neurons held at rate 1.00 through the deep delay, leak 0.00 (clean fixed point)
- [x] Recall is content-specific: 27/64 read-out units differ between two distinct stored patterns
- [x] Gated pathways: store input present only t=(0, 6), recall released only t=(20, 32)

## Outputs
- `outputs/store_input_raster.png` — gated content entering (pathway 3)
- `outputs/attractor_raster.png` — the held attractor pattern across the delay
- `outputs/recall_raster.png` — gated recall read-out (pathway 4)

## Conclusion
The Hippocampus stores a content pattern via a one-shot Hebbian imprint, holds
it as a stable attractor across a no-input delay, and recalls it on demand —
all under router-style gating of pathways 3/4. Build-order step 5 gate PASSED.
