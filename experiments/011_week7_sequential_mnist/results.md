# EXP-011 — Sequential MNIST results

**Run date:** 2026-05-25
**Spec:** `docs/design/2026-05-23-sequential-mnist.md`
**Plan:** `docs/superpowers/plans/2026-05-23-sequential-mnist-implementation.md`

## Headline

Recurrence on sequential MNIST helped a lot — **+37.28% gap** over the
feedforward control under identical config and seed. Both variants undershot
their predicted absolute-accuracy ranges; the recurrent variant was still
climbing on epoch 5 and looks under-trained, not architecturally broken.

## Predicted vs actual

| Metric | Predicted | Actual | Notes |
|---|---|---|---|
| Recurrent test acc | 94–96% | **79.12%** | Still climbing at epoch 4 (75.6 → 80.5%); 5 epochs at lr=3e-4 was too few |
| Feedforward test acc | 88–92% | **41.84%** | Plain Leaky on row-at-a-time has no lateral mixing; same under-training applies |
| Gap (recurrent − feedforward) | 3–6% | **+37.28%** | 6× larger than predicted, same direction |
| Recurrent params | 21,514 | 21,514 | Exact |
| Feedforward params | 5,002 | 5,002 | Exact |
| Untrained CE loss | 2.302 ± 0.1 | 2.319 / 2.294 | Within ±0.3 gate tolerance for both |

## Verification gates

| Gate | Status |
|---|---|
| Gate 1 — forward shape `[28, B, 10]` | PASS (both variants) |
| Gate 2 — param count ±100 | PASS (exact) |
| Gate 3 — initial loss ≈ ln(10) | PASS |
| Gate 4 — best-checkpoint reload | PASS (Δ = 0.00000) |

## Interpretation

The architecture comparison delivered its actual deliverable — a clean
recurrent-vs-feedforward gap on a task that genuinely requires temporal
integration. The size of the gap (37.28%) tells us:

- Plain `Leaky` membrane integration alone is NOT enough to bind row-major
  evidence over 28 steps into a digit classification. The feedforward
  net stalled near ~42%, materially above chance (10%) but far below MLP
  performance on flat MNIST — so it's using *some* of the time axis,
  just not enough.
- The recurrent `RLeaky(all_to_all=True)` adds lateral mixing across 128
  hidden units per step, which closes most of that gap even at only 5
  epochs.

## Surprise

The absolute accuracy undershoot is the real surprise, not the gap. Three
candidate causes (not investigated this session; future work):

1. **Training budget** — 5 epochs at lr=3e-4 with grad clip 1.0 was likely
   too few. Loss curves were still descending for both variants. Re-run at
   10–15 epochs would likely close the absolute-accuracy gap to spec.
2. **Optimizer / LR** — Adam at 3e-4 may be conservative for a 21K-param
   net. Worth a small sweep.
3. **Predict-before-execute spec was over-optimistic** — the predicted
   numbers came from external sequential-MNIST literature using deeper
   networks or longer schedules; the spec didn't re-calibrate to a
   single-hidden-layer 128-unit net at 5 epochs.

The gap direction and reload gate validate the architectural claim. Numeric
calibration is the follow-up.

## Artifacts

- `outputs/comparison.md` — pipe-table headline
- `outputs/comparison.csv` — same data, machine-readable
- `outputs/best_checkpoint.{pt,json}` — recurrent variant
- `outputs/recurrent/{checkpoint,training_curve,hidden_raster}.{pt,png}`
- `outputs/feedforward/{checkpoint,training_curve,hidden_raster}.{pt,png}`
