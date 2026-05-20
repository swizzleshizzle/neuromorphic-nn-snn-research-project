# Sequential MNIST (Recurrent vs Feedforward SNN) — Session Design Spec

**Spec date:** 2026-05-20 (Wednesday, Week 7 design session)
**Implementation date:** 2026-05-23 (Saturday, Week 7 build session)
**Phase/Step:** Phase 1, Step 1.3 / Module L7
**Experiment ID:** EXP-011 → folder `experiments/011_week7_sequential_mnist/`
**Status:** Draft pending Mike's review.
**Companion handoff:** [`docs/handoffs/week7_11_handoff`](../handoffs/week7_11_handoff)
**Implementing agent context:** This is a single-session two-variant experiment, not a library build. The deliverable is a comparison table proving (or disproving) that a recurrent hidden layer beats a feedforward one on a task that requires temporal integration. No pytest layer except for the row-encoder helper.

---

## 1. Goal

Quantify the accuracy gap between a **recurrent** SNN and an **architecturally identical feedforward** SNN on **sequential MNIST** — where each image is presented one pixel row per timestep (T=28). Both networks see the same data, train under the same budget, and differ only in whether the hidden layer is `snn.RLeaky(all_to_all=True)` or `snn.Leaky`.

**Success (operational):** both variants train to completion under identical seed/config; `comparison.{md,csv}` and `best_checkpoint.pt` are committed; reload-and-verify gate passes within ±0.1%.
**Success (scientific):** the predicted recurrent-vs-feedforward gap (§9) is confirmed within its predicted range, OR the actual gap is documented honestly even if it falls outside the prediction.
**Time budget:** 90 minutes.
**Hardware:** RTX 2080, CUDA verified available. 28-step BPTT on two small layers is well within budget.

This experiment exists to answer one specific question: **what does lateral mixing (W_rec) buy on top of single-cell membrane integration?** Plain `snn.Leaky` already has memory across timesteps via its membrane potential — the feedforward baseline is NOT memory-less. So the comparison isolates *lateral coupling* as the architectural primitive being tested.

---

## 2. Variants

| Variant | Hidden layer | num_steps | beta | Epochs | Predicted acc |
|---|---|---|---|---|---|
| `recurrent` | `snn.RLeaky(all_to_all=True, linear_features=128)` | 28 | 0.9 | 5 | 94–96% |
| `feedforward` | `snn.Leaky()` | 28 | 0.9 | 5 | 88–92% |

**Shared:** `batch_size=128`, `optimizer=adam`, `lr=3e-4` (one notch below week-6's 5e-4 to be conservative with 28-step BPTT), `seed=42`, `hidden_size=128`, `readout_window=4`, `threshold=1.0`, `reset_mechanism="subtract"`, `sequential=True`, **direct-current input (NO rate coding)**.

**Predicted gap: 3–6%.** Predict-before-execute commitment. The "feedforward wins by less than expected, but recurrent still wins" outcome is the most likely; the "feedforward effectively ties" outcome is the most informative if it happens.

---

## 3. Architecture (both variants)

```
Input: MNIST batch [B, 1, 28, 28]
  │
  ├─ squeeze channel + row-encode: [B, 1, 28, 28] → [28, B, 28]   (row t = pixels row t)
  │
  ├─ time loop, t = 0..27:
  │     x_t = sequence[t]                        # [B, 28]
  │     cur1 = fc1(x_t)                          # Linear(28, 128) → [B, 128]
  │
  │     # ---- HIDDEN LAYER ---- (the only variant-dependent branch)
  │     # recurrent:    spk1, mem1 = rlif1(cur1, spk1, mem1)   # RLeaky, all_to_all=True
  │     # feedforward:  spk1, mem1 = lif1(cur1, mem1)          # plain Leaky
  │
  │     cur2 = fc2(spk1)                         # Linear(128, 10) → [B, 10]
  │     spk2, mem2 = lif2(cur2, mem2)            # output Leaky
  │
  │     append spk2 → spk_rec
  │
  └─ return torch.stack(spk_rec)                 # [28, B, 10]

Loss   = CE( spk_rec[24:28].sum(dim=0), labels )       # readout window = last 4 steps
Accuracy = argmax( spk_rec[24:28].sum(dim=0), dim=1 ) == labels
```

**Why direct-current input (no rate coding):** the time axis must mean exactly "row index". Rate-coding the rows would mean "the network sees noisy samples of row $t$" instead of "the network sees row $t$ once at time $t$". For a task whose entire point is temporal-order integration, the encoding must not interfere with the temporal axis. (This intentionally differs from weeks 5–6, which DID rate-code.)

**Why `all_to_all=True` for recurrent:** lateral coupling is the architectural primitive being tested. The per-neuron `V` self-feedback from EXP-010 was a diagnostic for single-neuron sustained activity; this experiment tests *population-level* memory.

**Why readout window = last 4 steps:** by step 24, all 28 rows have been presented and the recurrent network has had ~4 steps to settle. Pure last-step is brittle (one bad timestep dominates the loss); full 28-step sum dilutes the integration requirement (early-row spikes contribute as much as integrated-state spikes).

**State init contract:** every `forward()` call calls `init_rleaky()` / `init_leaky()` on every spiking layer before the time loop. State does NOT carry across samples within a batch (snnTorch broadcasts the init to the batch dimension on first forward).

---

## 4. Predicted parameter counts

| Variant | Linear(28,128) | W_rec (128,128) | Linear(128,10) | Total |
|---|---|---|---|---|
| recurrent  | 3,712 | 16,512 | 1,290 | **21,514** |
| feedforward | 3,712 | — | 1,290 | **5,002** |

Bias terms included (each `Linear(in, out)` contributes `in*out + out`).

**Confound flagged honestly:** the recurrent variant has ~4× more parameters than the feedforward, because the W_rec matrix is the architectural primitive being added. We intentionally do NOT widen the feedforward to match params — that would change a second architectural variable (width) on top of the one we're isolating (recurrence). The spec records this so the comparison-table writeup acknowledges it.

If a future experiment wants to disentangle "lateral coupling per se" from "more params helps", a third variant `feedforward_wide` at H=285 (~21K params) is the right way to add it. Out of scope for EXP-011.

---

## 5. Config and module changes

### 5.1 `src/neuromorphic/config.py`

Add four fields to `ExperimentConfig` (alphabetical placement within their group):

```python
# --- Model architecture ---
arch: str = "baseline_mlp"  # ... | spiking_cnn | sequential_snn
hidden_size: int = 128         # NEW — used when arch == "sequential_snn"
recurrent: bool = False        # NEW — RLeaky (True) vs Leaky (False) for the hidden layer
readout_window: int = 4        # NEW — number of trailing steps summed for loss

# --- Temporal simulation (SNN) ---
num_steps: int = 25
sequential: bool = False       # NEW — feed input row-at-a-time (forces num_steps=28)
```

Add a `__post_init__` assertion: if `sequential` is True, `num_steps` must equal 28 and `encoding` must equal `"direct"` (newly defined value meaning "no encoder transformation — the row of pixel intensities is fed straight into `fc1`"). Surfaces config typos at load time, not at epoch 47.

**Pre-flight git check before editing config.py:** `git status && git diff src/neuromorphic/config.py` to confirm HEAD-clean. Week-6 lesson.

### 5.2 `experiments/011_week7_sequential_mnist/models.py`

Contains:
- `row_encode(images: torch.Tensor) -> torch.Tensor` — takes `[B, 1, 28, 28]` (or `[B, 28, 28]`) and returns `[28, B, 28]`. Pure function, pytest-testable.
- `SequentialSNN` — single class with a `recurrent: bool` switch. Constructor builds either RLeaky or Leaky for the hidden layer based on the flag. Forward pass branches on `self.recurrent` exactly once (RLeaky's forward takes `prev_spk`; Leaky's does not).

No `arch` dispatch in this experiment's `run.py` — the single class handles both variants. `arch="sequential_snn"` exists just so the config string makes sense in W&B logging.

---

## 6. File layout

```
experiments/011_week7_sequential_mnist/
├── recurrent.yaml                  # recurrent=True
├── feedforward.yaml                # recurrent=False
├── models.py                       # row_encode + SequentialSNN
├── run.py                          # def train(config) -> dict; CLI for single-variant runs
├── run_all.py                      # both variants → comparison.{md,csv} → best ckpt → reload check
└── outputs/                        # gitignored (existing rule)
    ├── recurrent/
    │   ├── checkpoint.pt
    │   ├── training_curve.png
    │   └── hidden_raster.png       # hidden-layer spikes on one test image
    ├── feedforward/
    │   ├── checkpoint.pt
    │   ├── training_curve.png
    │   └── hidden_raster.png
    ├── best_checkpoint.pt
    ├── best_checkpoint.json
    ├── comparison.md
    └── comparison.csv

tests/
└── test_row_encode.py              # pytest — shape + content of the row encoder
```

`run.py` is importable (`from run import train`) AND has a CLI. The CLI is for ad-hoc single-variant runs and debugging; `run_all.py` uses the importable form so it can collect return values.

---

## 7. `run.py` contract

```python
def train(config: ExperimentConfig) -> dict:
    """Train one variant (recurrent or feedforward sequential SNN).

    Returns
    -------
    dict with keys:
      variant         : str   — config.run_name
      recurrent       : bool  — config.recurrent
      num_params      : int
      num_steps       : int   — 28
      hidden_size     : int   — 128
      beta            : float
      epochs          : int
      train_seconds   : float
      final_test_acc  : float
      checkpoint_path : str   — absolute path to outputs/{variant}/checkpoint.pt
    """
```

Side effects, in order:
1. Build model: `SequentialSNN(num_inputs=28, hidden_size=128, num_outputs=10, beta=0.9, recurrent=config.recurrent, num_steps=28, readout_window=4)`.
2. Assert `config.sequential == True and config.num_steps == 28` (defense against silent config drift).
3. Run the three §8 verification gates on one batch BEFORE training (forward-shape, param-count, initial-loss).
4. Start `ExperimentTracker(config)` (W&B + TB, as week 5).
5. Train for `config.epochs`, timing the loop with `time.perf_counter()`. Use gradient clipping `max_norm=1.0` (cheap insurance for 28-step BPTT).
6. Full-test-set eval at the end (same `evaluate` helper structure as week 5).
7. Save `checkpoint.pt` containing: `model_state`, `config.to_dict()`, `loss_hist`, `test_loss_hist`, `test_acc_hist`, `final_accuracy`, `train_seconds`, `num_params`.
8. Render `training_curve.png` using `neuromorphic.viz.training_curve(history, log_interval=config.log_interval)`.
9. Render `hidden_raster.png`: pick test image index 0, run forward, extract hidden-layer spike record `[28, 1, 128]`, render with `neuromorphic.viz.spike_raster`. Save next to the curve.
10. Close tracker, return the metrics dict.

---

## 8. Verification gates (manual, in `run.py` before training)

Three correctness gates checked on a single batch before the training loop starts:

1. **Forward-shape gate.** Run one batch through `SequentialSNN.forward`. Assert `spk_out.shape == (28, batch_size, 10)`. Catches indexing bugs in the time loop and any shape drift in the row encoder.
2. **Param-count gate.** `sum(p.numel() for p in model.parameters())` must equal **21,514 ± 100** (recurrent) or **5,002 ± 100** (feedforward). Catches silent regressions in layer wiring (forgotten bias, wrong `linear_features`, etc.).
3. **Initial-loss gate.** Compute CE loss on one untrained-model batch. Must be `ln(10) ≈ 2.302 ± 0.1`. Loss far below = bug (model already discriminates somehow); loss far above = bug (e.g. summed counts are 0 across all classes, log of 0).

Plus a fourth gate after training, in `run_all.py`:

4. **Reload-and-verify gate.** Load `best_checkpoint.pt` into a fresh `SequentialSNN` instance, re-evaluate on the full test set, assert reproduced accuracy matches `best_checkpoint.json["final_test_acc"]` within ±0.1%. Catches save/load mismatches and stale state.

---

## 9. `run_all.py` behavior

```python
def main():
    variant_yamls = ["recurrent.yaml", "feedforward.yaml"]
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
         "gap_vs_other": best["final_test_acc"] - min(r["final_test_acc"] for r in results),
         "config_summary": {k: best[k] for k in
                            ["recurrent", "num_steps", "hidden_size", "beta",
                             "epochs", "num_params"]}},
        open(here / "outputs" / "best_checkpoint.json", "w"),
        indent=2,
    )
    run_reload_gate(best)              # gate #4
    print_summary(results, best)
```

Comparison table columns (both `.md` and `.csv`):

`Variant | Hidden layer | Params | Epochs | Train time | Test acc | Gap vs feedforward`

Train time rendered as `Hm Ss` in `.md`, raw seconds in `.csv`. Accuracy as `XX.XX%` in `.md`, float in `.csv`. Gap as `+X.XX%` in `.md`, signed float in `.csv`. Feedforward row's gap is 0.

---

## 10. Build order (90 min budget)

```
0:00  Pre-flight: git status clean. Read this spec end-to-end one more time.  ~3 min
0:03  Add 4 config fields + __post_init__ assertion to src/neuromorphic/
      config.py. Run any existing tests to confirm green.                     ~10 min
0:13  Scaffold experiments/011_week7_sequential_mnist/. Write models.py:
      row_encode + SequentialSNN with recurrent switch.                       ~15 min
0:28  Write tests/test_row_encode.py. Confirm green.                          ~5 min
0:33  Write run.py with the three pre-training gates (§8) + training loop.    ~12 min
0:45  recurrent.yaml + first training run. Confirm gates pass, eyeball
      training_curve.png, capture final_test_acc.                             ~12 min
0:57  feedforward.yaml + second training run. Same checks.                    ~12 min
1:09  Write run_all.py: comparison writers + best-ckpt copy + reload gate.    ~10 min
1:19  Run run_all.py end-to-end. Eyeball comparison.md.                       ~5 min
1:24  Commit in scoped increments (matches week-5/6 commit style).            ~6 min
1:30  done.
```

**Cut order if running long:**
1. Drop the `hidden_raster.png` step (item 9 in `run.py`) first — it's nice-to-have, not deliverable-blocking.
2. Drop the per-variant `training_curve.png` second.
3. NEVER drop the feedforward variant — without it, the experiment has no result.
4. NEVER drop the param-count gate or reload gate — both have caught real bugs in past experiments.

---

## 11. Non-goals for today

- No CNN front-end (Conv2d on each row). Linear(28,128) is sufficient to test the recurrence question.
- No alternative encodings (rate, latency). Direct current throughout.
- No beta sweep. β=0.9 for both layers, locked.
- No batch-size or LR sweep.
- No multi-seed runs. Single seed=42 for both variants. (Stretch goal if budget allows: re-run both variants on seeds 43, 44 and report mean ± std. Treat as cut-first if budget is tight.)
- No RSynaptic variant. Maybe in week-8 if the recurrent result is interesting.
- No reset-mechanism comparisons.
- No learning-rate scheduling.

---

## 12. Definition of done

The Saturday session is complete when:

1. `experiments/011_week7_sequential_mnist/outputs/comparison.md` exists with 2 rows.
2. `experiments/011_week7_sequential_mnist/outputs/comparison.csv` exists with the same data, machine-readable.
3. `experiments/011_week7_sequential_mnist/outputs/best_checkpoint.pt` exists, with `best_checkpoint.json` sidecar.
4. The recurrent-vs-feedforward gap is reported in `comparison.md` regardless of sign or magnitude.
5. Reload gate (§8 #4) passes.
6. Source files (YAMLs, `models.py`, `run.py`, `run_all.py`, `tests/test_row_encode.py`, config-py edits) are committed in scoped increments. Outputs are gitignored.
7. The EXP-011 row in `experiment-log.md` is filled in (variant table, gap, training time, any surprises).

If the predicted accuracy range is missed (in either direction), commit the table anyway — a documented surprise is the most valuable kind of result. Note the surprise in the commit message and queue a follow-up for week 8.

---

## 13. Open questions to revisit during build (not blockers)

- Does the RLeaky's default `W_rec` init (PyTorch Linear uniform) work, or do we need a smaller-scale init to avoid runaway at step 0? Mitigation: the §8 initial-loss gate will catch runaway. If it fires, scale `rlif.recurrent.weight.data *= 0.1` before training.
- Is gradient clipping at `max_norm=1.0` too aggressive or too loose for 28-step BPTT? Check the gradient-norm log on the first few iterations; tighten or loosen if needed.
- Should we capture the hidden-layer mean firing rate as an additional comparison metric? Cheap to add; defer to a follow-up if the basic comparison is interesting.

---

## Revision history

| Date | Change |
|---|---|
| 2026-05-20 | Initial draft. Two-variant comparison (recurrent vs feedforward sequential SNN), 90-min budget, predicted gap 3–6%, predicted params 21.5K vs 5.0K, four verification gates. |
