# MNIST SNN Optimization — Session Design Spec

**Date:** 2026-05-16 (Saturday afternoon, Week 6)
**Status:** Approved by Mike, ready for implementation.
**Companion morning spec:** [`2026-05-16-viz-toolkit-implementation.md`](2026-05-16-viz-toolkit-implementation.md)
**Implementing agent context:** This is a single-session experiment, not a library build. The deliverables are a comparison table, a best checkpoint, and three trained models with training curves. No pytest layer.

---

## 1. Goal

Push the week-5 MNIST SNN test accuracy from the **93.5% baseline** to **≥95%** by introducing convolutional spiking layers and tuning the leak constant. Three variants, each fully trained, all compared in one table.

**Success:** at least one variant reaches ≥95.0% on the full MNIST test set AND `comparison.md` + `comparison.csv` + `best_checkpoint.pt` are committed.
**Stretch:** all three variants documented and `training_curve.png` rendered for each (dogfoods the morning's viz toolkit).
**Time budget:** 90 minutes.
**Hardware:** RTX 2080, CUDA verified available.

---

## 2. Variants

| Variant | Arch | num_steps | beta | Epochs | Hypothesis |
|---|---|---|---|---|---|
| `v1_mlp_tuned` | Feedforward SNN 784→1000→1000→10 (week-5 baseline) | 25 | 0.95 | **5** (was 3) | ~94–95% — answers "does the MLP just need more epochs?" |
| `v2_cnn_baseline` | Spiking CNN (see §3) | 25 | 0.95 | 3 | ~96–98% — answers "does conv help?" |
| `v3_cnn_tuned` | Same as V2 | 25 | **0.90** | 3 | Marginal gain over V2 — answers "does a faster leak help?" |

All variants share: `batch_size=128`, `optimizer=adam`, `lr=5e-4`, `gain=1.0`, `seed=42`, rate encoding.

---

## 3. SpikingCNN architecture (V2, V3)

```
Input: [num_steps, batch, 1, 28, 28]   (rate-coded MNIST images, image shape preserved)
  │
  ├─ time loop, t = 0..num_steps-1:
  │
  │     Conv2d(1, 16, kernel=5, padding=2)   → [B, 16, 28, 28]
  │     MaxPool2d(2)                         → [B, 16, 14, 14]
  │     snn.Leaky(beta, threshold=1.0)       → spk1, mem1
  │
  │     Conv2d(16, 32, kernel=5, padding=2)  → [B, 32, 14, 14]
  │     MaxPool2d(2)                         → [B, 32,  7,  7]
  │     snn.Leaky(beta, threshold=1.0)       → spk2, mem2
  │
  │     Flatten                              → [B, 32*7*7 = 1568]
  │     Linear(1568, 10)                     → [B, 10]
  │     snn.Leaky(beta, threshold=1.0)       → spk3, mem3
  │
  │     append spk3, mem3
  │
  └─ return torch.stack(spk_rec), torch.stack(mem_rec)
       both shape [num_steps, batch, 10]
```

- **Parameter count:** ~29K (vs. ~1.80M for the MLP — verified: 416 + 12,832 + 15,690 = 28,938 conv+linear params; `snn.Leaky` adds 0 by default). Smaller, locality-aware.
- **Padding `=2`** with kernel `=5` preserves spatial dims so Pool→14→7 is clean.
- **Reset:** `init_leaky()` on all three LIFs at start of every `forward()`.
- **Output contract:** same `[T, B, 10]` shape as `FeedforwardSNN`, so the existing loss + accuracy logic doesn't change.

**Encoding generalization:** the week-5 `encode_rate` uses `probs.unsqueeze(0).expand(num_steps, -1, -1)` which only works on 2D input. Replace with:

```python
probs_expanded = probs.unsqueeze(0).expand(num_steps, *([-1] * probs.ndim))
```

…so the same encoder works for both `[B, 784]` and `[B, 1, 28, 28]` inputs.

---

## 4. Config and module changes

### 4.1 `src/neuromorphic/config.py`

One-line update to the `arch` field comment:

```python
arch: str = "baseline_mlp"  # baseline_mlp | tiny_mlp | simple_cnn | feedforward_snn | spiking_cnn
```

No new fields. CNN topology (channels, kernel size, pool size) is hardcoded inside `SpikingCNN.__init__` — promoting to config is YAGNI today; revisit when a future week sweeps CNN topology.

### 4.2 `experiments/009_week6_snn_mnist_optimization/models.py`

Contains:
- `encode_rate(data, num_steps, gain)` — lifted from week 5 with the generic `.expand` fix above.
- `FeedforwardSNN` — verbatim from `experiments/008_week5_snn_mnist_baseline/run2.py` (so this experiment doesn't import from another experiment's folder).
- `SpikingCNN` — new, per §3.

---

## 5. File layout

```
experiments/009_week6_snn_mnist_optimization/
├── v1_mlp_tuned.yaml
├── v2_cnn_baseline.yaml
├── v3_cnn_tuned.yaml
├── models.py              # encode_rate + FeedforwardSNN + SpikingCNN
├── run.py                 # def train(config) -> metrics_dict; CLI for one-variant runs
├── run_all.py             # iterates 3 YAMLs, writes comparison.{md,csv}, copies best ckpt
└── outputs/
    ├── v1_mlp_tuned/
    │   ├── checkpoint.pt
    │   └── training_curve.png
    ├── v2_cnn_baseline/
    │   ├── checkpoint.pt
    │   └── training_curve.png
    ├── v3_cnn_tuned/
    │   ├── checkpoint.pt
    │   └── training_curve.png
    ├── best_checkpoint.pt       # copy of highest-acc variant
    ├── best_checkpoint.json     # {variant_name, final_test_acc, config_summary}
    ├── comparison.md
    └── comparison.csv
```

`run.py` is importable (`from run import train`) AND has a CLI (`python run.py --config v1_mlp_tuned.yaml`). The CLI is for ad-hoc single-variant runs and debugging; `run_all.py` uses the importable form so it can collect return values.

---

## 6. `run.py` contract

```python
def train(config: ExperimentConfig) -> dict:
    """Train one variant. Returns metrics dict + writes per-variant artifacts.

    Returns
    -------
    dict with keys:
      variant         : str   — config.run_name
      arch            : str   — config.arch
      num_params      : int
      num_steps       : int
      beta            : float
      epochs          : int
      train_seconds   : float
      final_test_acc  : float
      checkpoint_path : str   — absolute path to outputs/{variant}/checkpoint.pt
    """
```

Side effects, in order:
1. Build model from `config.arch` ("feedforward_snn" → `FeedforwardSNN`; "spiking_cnn" → `SpikingCNN`). Raise `ValueError` on other values.
2. Start `ExperimentTracker(config)` (W&B + TB, as week 5).
3. Train for `config.epochs`, timing the loop with `time.perf_counter()`.
4. Full-test-set eval at the end (same `evaluate` helper structure as week 5).
5. Save `checkpoint.pt` to `experiments/009_.../outputs/{config.run_name}/`. The checkpoint contains: `model_state`, `config.to_dict()`, `loss_hist`, `test_loss_hist`, `test_acc_hist`, `final_accuracy`, `train_seconds`.
6. Render `training_curve.png` using `neuromorphic.viz.training_curve(history, log_interval=config.log_interval)` — saves to the same per-variant dir.
7. Close tracker, return the metrics dict.

**Arch dispatch in the training loop:** for `feedforward_snn`, flatten to `[B, 784]` before encoding; for `spiking_cnn`, keep `[B, 1, 28, 28]`. Single `if`-branch at the data-prep step is enough.

---

## 7. `run_all.py` behavior

```python
def main():
    variant_yamls = ["v1_mlp_tuned.yaml", "v2_cnn_baseline.yaml", "v3_cnn_tuned.yaml"]
    results = []
    for y in variant_yamls:
        config = load_config(here / y)
        metrics = train(config)            # imports from run.py
        results.append(metrics)

    write_comparison_md(results, here / "outputs" / "comparison.md")
    write_comparison_csv(results, here / "outputs" / "comparison.csv")
    best = max(results, key=lambda r: r["final_test_acc"])
    shutil.copy(best["checkpoint_path"], here / "outputs" / "best_checkpoint.pt")
    json.dump(
        {"variant": best["variant"], "final_test_acc": best["final_test_acc"],
         "config_summary": {k: best[k] for k in
                            ["arch", "num_steps", "beta", "epochs", "num_params"]}},
        open(here / "outputs" / "best_checkpoint.json", "w"),
        indent=2,
    )
    print_summary(results, best)
```

Two writers, both deterministic given `results`:
- `write_comparison_md(results, path)` — pipe-table format, columns in this order: `Variant | Arch | Params | num_steps | beta | Epochs | Train time | Test acc`. Train time rendered as `Hm Ss`. Accuracy as `XX.X%`.
- `write_comparison_csv(results, path)` — `csv.DictWriter`, same columns, raw numeric values (no formatting).

---

## 8. Dependency on the morning viz toolkit

This experiment **imports `neuromorphic.viz.training_curve`** in step 6 of `run.py`. If the morning viz-toolkit build did not complete (or `training_curve` is broken), `run.py` will `ImportError` on first call.

**Sequencing rule:** the morning build (`docs/superpowers/plans/2026-05-16-viz-toolkit-implementation.md`) MUST be merged and verified (`pytest tests/viz/ -v` reports 24 passed) before this experiment runs. If something blocks the toolkit, defer this experiment.

---

## 9. Verification (manual, no pytest)

Three correctness gates checked as we build:

1. **V1 sanity:** V1 final accuracy must be within ±0.5% of the week-5 baseline 93.5% (or higher — the 2 extra epochs may help). Substantially lower means the refactor broke something; substantially higher than ~95% is suspicious — investigate before trusting.
2. **CNN forward shape:** before V2 training, run one batch through `SpikingCNN.forward` and assert `spk_out.shape == (num_steps, batch_size, 10)`. Catches indexing bugs in the time loop or kernel/pool sizing.
3. **Best-checkpoint reload:** after `run_all.py`, load `best_checkpoint.pt` into a fresh model instance, re-evaluate on the full test set, assert reproduced accuracy matches `best_checkpoint.json["final_test_acc"]` within ±0.1%. Catches save/load mismatches.

---

## 10. Build order (90 min budget)

```
0:00  Create 009/ folder. Lift FeedforwardSNN + encode_rate into models.py,
      apply the generic .expand shape fix.                              ~10 min
0:10  Write SpikingCNN class. Standalone CNN forward-shape check (§9 #2). ~15 min
0:25  Write run.py: train(config) returning metrics dict. Arch branch.   ~10 min
0:35  v1_mlp_tuned.yaml + V1 full training (5 epochs MLP).               ~7 min
0:42  v2_cnn_baseline.yaml + V2 full training (3 epochs CNN).           ~15 min
0:57  v3_cnn_tuned.yaml + V3 full training (CNN, beta=0.90).            ~15 min
1:12  run_all.py: comparison.md/csv writers, best-ckpt copy, reload check.~10 min
1:22  Commit. Eyeball comparison.md and three training_curve.png files.  ~8 min
1:30  done.
```

**Cut order if running long:**
1. Drop V3 first (smallest insight per minute).
2. Drop the best-checkpoint reload check second.
3. Never drop V2 — the CNN is the highest-leverage variant for the accuracy goal.

---

## 11. Non-goals for today

- No spiking-CNN topology sweep (channels, kernel size). The V2/V3 architecture is fixed.
- No alternative encodings (latency, direct). Rate encoding throughout.
- No alternative optimizers. Adam @ 5e-4 throughout.
- No learning-rate scheduling.
- No reset-mechanism comparisons ("subtract" throughout).
- No batch-size search.
- No pytest tests for this experiment (it's an experiment, not library code — `models.py` could grow tests later, but today is for results).

---

## 12. Definition of done

The afternoon is complete when:

1. `experiments/009_week6_snn_mnist_optimization/outputs/comparison.md` exists with 3 rows.
2. `experiments/009_week6_snn_mnist_optimization/outputs/comparison.csv` exists with the same data, machine-readable.
3. `experiments/009_week6_snn_mnist_optimization/outputs/best_checkpoint.pt` exists, with `best_checkpoint.json` sidecar.
4. At least one variant in `comparison.md` shows ≥95.0% test accuracy.
5. Verification gate §9 #3 (reload check) passes.
6. All artifacts under `outputs/` are gitignored (per the existing `outputs/*` rule added in the morning); the source files (YAMLs, models.py, run.py, run_all.py) ARE committed.

If item 4 fails but everything else passes, commit the table anyway — a documented near-miss is better than no record. Note the gap in the commit message.

---

## Revision history

| Date | Change |
|---|---|
| 2026-05-16 | Initial draft. Three-variant architecture progression (MLP-tuned → CNN-baseline → CNN-beta-tuned), shared training harness, comparison artifacts. |
