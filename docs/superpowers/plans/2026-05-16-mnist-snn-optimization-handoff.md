# MNIST SNN Optimization — Build-Agent Handoff

Point a fresh Claude Code session at this file. Everything the build agent needs is here or linked from here.

---

You're running an MNIST SNN optimization experiment for a neuromorphic AI research project. A full design spec and a task-by-task implementation plan already exist. Your job is to execute the plan exactly as written. Deliverables are a comparison table (markdown + CSV), three trained models, and a best-checkpoint at ≥95% test accuracy.

## Hard dependency — check first

This experiment imports `neuromorphic.viz.training_curve` from a sibling project. Before doing anything else:

```powershell
.venv\Scripts\python.exe -m pytest tests/viz/ -q
.venv\Scripts\python.exe -c "from neuromorphic.viz import training_curve; print('viz ok')"
```

Both must succeed (pytest: `24 passed`, second command: `viz ok`). If either fails, STOP — the morning viz-toolkit plan hasn't been completed yet. Do not proceed.

## Read first, in this order

1. `docs/superpowers/plans/2026-05-16-mnist-snn-optimization.md` — the plan. Tasks 0–9 with bite-sized steps. Follow it verbatim; do not improvise.
2. `docs/design/2026-05-16-mnist-snn-optimization.md` — the session spec the plan implements. Read for the variant rationale (§2), CNN topology (§3), the `train()` contract (§6), and the three verification gates (§9).
3. `experiments/008_week5_snn_mnist_baseline/run2.py` — week-5 reference for `FeedforwardSNN` and the training-loop structure. `models.py` is partially lifted from here.

## Use the executing-plans skill

Invoke `superpowers:executing-plans` before starting Task 0. That skill governs how you work through the checkboxes.

## Hard rules (don't violate)

- **No pytest tests for this experiment.** Verification is manual via inline smoke commands at the end of each task. The spec explicitly forbids a pytest layer here — this is an experiment, not library code.
- **Predict before each component.** Each task has a "Predict before executing" block stating expected shapes, accuracies, or runtimes. State them out loud, run, compare. Divergence is a stop-signal.
- **Three correctness gates are non-negotiable:**
  1. **CNN forward shape** (Task 2 step 2) — must produce `(25, 8, 10)` on a `(25, 8, 1, 28, 28)` input and exactly 28,938 params. Catch indexing/padding bugs BEFORE training V2.
  2. **V1 sanity** (Task 4 step 3) — V1 final accuracy must land within ±0.5% of the week-5 baseline 93.5% (or higher). If V1 ≪ 93%, the refactor broke something. Diagnose before continuing.
  3. **Best-checkpoint reload** (Task 8) — reloaded accuracy must match `best_checkpoint.json["final_test_acc"]` within ±0.1%. Catch save/load bugs.
- **Outputs are gitignored. Source files are committed.** Each task that creates source files (YAMLs, `models.py`, `run.py`, `run_all.py`, `verify_best.py`) commits them. The `outputs/` directory is gitignored by the morning task's `.gitignore` rule.
- Plain commit messages as written in the plan — no Co-Authored-By trailer needed.
- No new dependencies. Use only what's installed (torch, snntorch 0.9.4, torchvision, matplotlib, pyyaml).

## Long-running steps

Tasks 4, 5, 6 are full training runs (~7 min, ~12 min, ~12 min on the RTX 2080). Don't try to subdivide them. Just run them, watch the logs, and verify the result. Total wall-clock for the plan including writes is roughly 90 minutes.

## Environment quirks (Windows)

- Shell is PowerShell. Use `\` in paths, `.venv\Scripts\python.exe` (not `python` — may not be on PATH). The Bash tool is also available if you prefer POSIX-style; if you use it, write `.venv/Scripts/python.exe`.
- The active venv is at `.venv\`. Already has everything: torch (CUDA), snntorch 0.9.4, matplotlib 3.10, pyyaml, pytest, the `neuromorphic` package in editable mode.
- CUDA is available on an RTX 2080. `get_device()` returns `cuda` automatically.
- MNIST is already downloaded under `./data/`. No internet required.

## Permission expectations

You'll be writing files under `experiments/009_week6_snn_mnist_optimization/` and `src/neuromorphic/config.py` (one comment-line edit). You'll run `python` for both training runs and inline smoke checks. You'll create git commits. All in-scope.

## Done criteria (from the plan's tail)

1. `experiments/009_week6_snn_mnist_optimization/outputs/comparison.md` exists with 3 data rows.
2. `experiments/009_week6_snn_mnist_optimization/outputs/comparison.csv` exists with matching data, machine-readable.
3. `experiments/009_week6_snn_mnist_optimization/outputs/best_checkpoint.pt` and `best_checkpoint.json` exist.
4. `python verify_best.py` prints `VERIFY OK`.
5. At least one variant in the table shows ≥95.0% test accuracy. (If missed, the experiment is still "complete" — just document the gap in the final commit.)
6. Source files (`models.py`, `run.py`, `run_all.py`, `verify_best.py`, 3 YAMLs, `__init__.py`, the `config.py` update) are all committed; `outputs/` artifacts are not.

Report back when done with: number of tasks completed, the final `comparison.md` table contents, the best variant + accuracy, and any places where reality diverged from the predictions.
