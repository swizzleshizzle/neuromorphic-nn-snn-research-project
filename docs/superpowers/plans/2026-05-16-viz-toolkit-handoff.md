# Stage 1 Viz Toolkit — Build-Agent Handoff

Point a fresh Claude Code session at this file. Everything the build agent needs is here or linked from here.

---

You're implementing the Stage 1 visualization toolkit for a neuromorphic SNN research project. A full design spec and a task-by-task implementation plan already exist. Your job is to execute the plan exactly as written.

## Read first, in this order

1. `docs/superpowers/plans/2026-05-16-viz-toolkit-implementation.md` — the plan. Tasks 0–9, each with bite-sized TDD steps (test → run → impl → run → eyeball → commit). Follow it verbatim; do not improvise.
2. `docs/design/2026-05-16-viz-toolkit-implementation.md` — the session spec that the plan implements. Read for the contracts (§4) and the "predict-before-execute" discipline (§9).
3. `docs/design/stage-1-visualization-toolkit.md` — the parent design doc. Skim the "Agent Hand-Off Context" section at the top; it explains the project's expectations for build discipline.

## Use the executing-plans skill

Invoke `superpowers:executing-plans` before starting Task 0. That skill governs how you work through the checkboxes.

## Hard rules (don't violate)

- Implement ONE function, run its pytest smoke, eyeball ONE render against synthetic data, THEN move to the next function. Do not batch all 7 implementations and test at the end. The parent spec calls this non-negotiable.
- Before each function task, write out a 3-line prediction: expected output shape of any intermediate tensor, what the plot should look like, what would indicate it's broken. Then run. Compare prediction to result.
- Every public function must return `(Figure, Axes)`. Never call `plt.show()` or `fig.savefig()` inside a viz function. Surface contract violations loudly — do not paper over them with try/except.
- Canonical tensor shape is `[T, B, N]`. Already verified: `splt.raster` accepts an `ax`, `splt.traces` does NOT (uses GridSpec) — that's why `membrane_trace` is implemented from scratch, not wrapped.
- Commit after each task per the plan's commit step. Plain commit messages as written in the plan — no Co-Authored-By trailer needed.
- No new dependencies beyond pytest (which Task 0 adds). No seaborn, no sklearn, no plotly.

## Environment quirks (Windows)

- Shell is PowerShell. Use `\` in paths, `.venv\Scripts\python.exe` (not `python` — it may not be on PATH). Bash tool is also available if needed for POSIX-style commands.
- The active venv is at `.venv\`. Already has torch, snntorch 0.9.4, matplotlib 3.10. pytest is NOT installed — Task 0 installs it.

## Permission expectations

You'll be writing files under `src/neuromorphic/viz/`, `tests/viz/`, `experiments/008_week5_snn_mnist_baseline/`, and `outputs/`. You'll run `pytest` and small `python -c "..."` smoke commands. You'll create git commits. All in-scope.

## Done criteria (from the plan's tail)

1. `pytest tests/viz/ -v` reports `24 passed`.
2. `python experiments/008_week5_snn_mnist_baseline/viz_smoke.py` runs without error and saves 8 PNGs to `experiments/008_week5_snn_mnist_baseline/outputs/snn_mnist_baseline/viz_smoke/`.
3. All 8 PNGs visually match the per-task predictions.
4. `from neuromorphic.viz import spike_raster, membrane_trace, weight_histogram, weight_heatmap, training_curve, population_rate, psth` succeeds.

Report back when done with: number of tasks completed, full pytest output of the final run, and any places where reality diverged from the predictions.
