# Multi-Region Dashboard — Batch Viz (Stage 2, MVP)

**Date:** 2026-06-24
**Module:** Phase 2, L17 — Build Dashboard v1 (Implementation)
**Status:** Approved (design), implementation in progress
**Artifact:** `src/dashboard/multi_region_viz.py`

## Purpose

Batch (post-experiment) dashboard: read a saved NEURO·SCOPE JSONL trace and
render a single multi-panel matplotlib figure summarizing one episode of the
five-region brain on grid-world. This is the Stage-2 dashboard MVP from the
project plan — static PNG output, no live/streaming, no React.

This is a stepping stone toward the full NEURO·SCOPE platform (WebGL hero +
React panels + live transport), which remains separate, later work. Tonight's
scope is "aesthetic north star, MVP execution" (option A): honor the locked
region/pathway names, signal-flow order, and NEURO·SCOPE color language where
cheap, but ship the static figure.

## Data contract (already locked)

Source of truth: `docs/superpowers/specs/2026-06-17-stage2-dashboard-design.md`.
Two-part JSONL: line 0 = header, lines 1+ = one frame per env step.

Verified against `outputs/week11_dashboard_trace.jsonl` and
`outputs/week11_trained_trace.jsonl` (both: 1 header + 100 frames, single
episode 0, 100 steps, agent never reaches goal → return walks to −100).

Fields this script consumes:

- **Header:** `brain.{id,config_hash,seed,T}`, `task.{grid_n,action_labels}`,
  `regions[].{id,label}`, `pathways[].{id,src,dst,gated}`.
- **Frame:** `episode`, `step`, `task.{agent,goal,action_label,reward,return}`,
  `field[region].spikes` → `[T][N]` binary matrix, `pathways[id].intensity`.
- Region field shapes (verified): sensory `32×64`, hippocampus `32×150`,
  prefrontal `32×4`, router `32×4`, motor `32×4`.
- Pathways (exactly four, real edges): `sens_hippo`, `sens_pfc`, `hippo_pfc`,
  `pfc_motor`.

## Architecture (small, testable units)

- `load_trace(path) -> (header, frames)` — parse two-part JSONL. Pure I/O.
- `summarize_reward(frames) -> (x_label, xs, ys, playhead_x)` — **auto-detect**:
  if >1 distinct `episode`, plot per-episode final `return` (x = episode);
  else within-episode cumulative `return` (x = step). `playhead_x` marks the
  rendered step on the curve.
- `render_dashboard(header, frames, step) -> Figure` — pure layout/draw, no I/O.
  Composed of small per-panel helpers (`_raster`, `_comm_heatmap`, `_grid`,
  `_reward`). Per-frame panels render the single chosen `step`.
- `main(argv)` — argparse CLI; wires the above and saves PNG.

## Layout (8 panels, GridSpec)

```
┌─ title: five-region · seed 0 · cfg b65fb41f · step N/99 · ep return R ─┐
│  sensory  raster [32×64]   │   inter-region comm heatmap (5×5)         │
│  hippo    raster [32×150]  │                                           │
│  pfc      raster [32×4]    │──────────────────────────────────────────│
│  router   raster [32×4]    │   grid-world state (agent ▶ goal ★)       │
│  motor    raster [32×4]    │──────────────────────────────────────────│
│  (shared time axis, T=32)  │   reward / return curve (auto-detected)   │
└────────────────────────────┴───────────────────────────────────────────┘
```

- **Left column:** 5 spike rasters stacked in signal-flow order
  (sensory→hippocampus→prefrontal→router→motor), shared x-axis (time, T=32),
  each tinted with its region accent color. Reuse
  `src/neuromorphic/viz/spikes.py::spike_raster` where it fits; fall back to a
  local `imshow` raster if the helper's API/coloring doesn't compose cleanly.
- **Right column:** comm heatmap (top), grid-world state (middle), reward curve
  (bottom).
- **Comm heatmap:** 5×5 region×region matrix (rows=src, cols=dst), cells lit
  only for the four real pathways, value = `pathways[id].intensity`; gated edges
  annotated. Non-edges shown as background.
- **Grid-world panel:** `grid_n × grid_n` cells, agent and goal marked, current
  `action_label` shown as a readout/arrow.
- **Reward curve:** auto-detected series with a playhead marker at the rendered
  step.

## Aesthetic (NEURO·SCOPE option A)

Dark "scope" theme. Region accent palette (from design file):

| region | hex |
|---|---|
| sensory | `#3fd2ff` |
| hippocampus | `#ad8bff` |
| prefrontal | `#ffd24a` |
| router | `#ff5a8a` |
| motor | `#46f0a0` |

Background `#0a0a0c`, panel `#11161f`, text `#e9edf6`, muted `#9aa3b6`,
grid/edges `#646d80`. Reward: positive `#46f0a0`, negative `#ffb37a`.
Centralized in a `PALETTE` dict / small theme block so it stays the single
source of styling truth.

## CLI

```
python -m dashboard.multi_region_viz <trace.jsonl> [--step N] [--out PATH] [--show]
```

- `trace_path` (positional) — JSONL trace.
- `--step` — env step index for per-frame panels; default = **last step**.
  Negative indices and out-of-range are clamped with a warning.
- `--out` — output PNG; default `outputs/<trace-stem>_dashboard.png`.
- `--show` — also `plt.show()` (off by default for headless runs).

## Testing

TDD. Unit tests against a tiny synthetic in-memory trace (2 regions / few steps)
plus a smoke test on the real `outputs/week11_dashboard_trace.jsonl`:

- `load_trace` splits header vs frames correctly; bad/empty lines handled.
- `summarize_reward` picks per-episode branch for multi-episode input and
  within-episode branch for single-episode input; playhead lands on chosen step.
- `render_dashboard` returns a `Figure` with the expected number of axes and
  does not raise on real and synthetic data (use `Agg` backend).
- `--step` clamping behavior.

## Out of scope (tonight)

- Live/streaming/WebSocket, React frontend, WebGL hero.
- Per-neuron membrane traces (deferred — flash/spike only).
- Animation across frames (`render_dashboard` stays single-step; an
  `animate_episode` may be added later if time permits / for option C).
- Regenerating a goal-reaching trace from the trained brain (optional follow-up;
  the viz works on existing traces regardless).
```
