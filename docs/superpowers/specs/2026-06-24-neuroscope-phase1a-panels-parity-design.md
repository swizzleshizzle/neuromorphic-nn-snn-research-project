# NEURO·SCOPE Phase 1a — Panels Parity

**Date:** 2026-06-24
**Status:** Approved (design), ready for implementation plan
**Builds on:** Phase 0 (`docs/superpowers/plans/2026-06-18-neuroscope-phase0-foundation.md`, merged),
the platform roadmap (`docs/superpowers/specs/2026-06-18-neuroscope-platform-design.md`, §8 Phase 1),
the data contract (`docs/superpowers/specs/2026-06-17-stage2-dashboard-design.md`),
and the design comps (`docs/handoffs/claude_design/`, primary: `NEURO-SCOPE Dashboard.dc.html`).

## Context — Phase 1 is decomposed

Phase 1 ("Parity — match the design spec") is too large for one spec. It is split into three
sequenced, independently-shippable slices, each with its own spec → plan → build cycle:

- **1a — Panels parity** (this spec): complete Panels 02/04/05, upgrade 01/03 to full spec, on a
  CSS theme-token foundation. Reactive React/SVG; low risk; extends the Phase-0 panel pattern.
- **1b — Hero parity** (later): Flow Map 2D treatment + full 3D Cloud (edges, travelling pulses,
  spike flash, agent receptive-field, sensory-grid overlay, Cloud/Flow toggle). Canvas/WebGL.
- **1c — Chrome & UX** (later): Focus mode, Observatory/Clinical theme toggle, full scrubber,
  encoding key.

This document specifies **1a only**.

## Goal

Bring the dashboard's panel layer up to the design-spec standard: all five panels rendering the
real trace, data-driven from the header, on a theme-token foundation that makes the 1c theme swap
additive. Done when all five panels render correctly against `outputs/week11_dashboard_trace.jsonl`
with unit tests green and the Playwright smoke asserting the new panels.

## Approach

**Token foundation + a shared `Panel` chrome primitive + reactive per-panel components.** The comp
gives every panel identical card chrome (kicker `PANEL 0X · …`, title, accent dot, same
border/radius/padding). Extracting one `Panel` wrapper keeps the five panels DRY and visually
consistent; each panel stays a small, independently-testable component consuming the Zustand store
reactively (the Phase-0 pattern). Rejected: self-styling every panel (repetition, drift) and
merging panels into rail containers (untestable, low cohesion).

## 1. Theme token foundation (the 1c seam)

New `dashboard/src/theme/tokens.css`, imported once at app entry. Defines the **Observatory** token
set as CSS custom properties on `:root` (one theme, no toggle — 1c adds the Clinical set + toggle,
purely additively). Existing Phase-0 panels (01, 03), `TopBar`, `Scrubber`, and `Shell` are
refactored off inline hex onto these tokens.

Surfaces / text / blur (Observatory):

| token | value | token | value |
|---|---|---|---|
| `--bg` | `#05060a` | `--text` | `#e9edf6` |
| `--bg2` | `#080a12` | `--text-dim` | `#9aa3b6` |
| `--bar` | `rgba(8,10,16,.86)` | `--text-faint` | `#5b6378` |
| `--panel` | `rgba(13,16,24,.66)` | `--edge` | `rgba(255,255,255,.075)` |
| `--panel2` | `rgba(22,26,37,.55)` | `--edge2` | `rgba(255,255,255,.045)` |
| `--blur` | `blur(9px)` | | |

Region hues: `--c-sensory #3fd2ff`, `--c-hippocampus #ad8bff`, `--c-prefrontal #ffd24a`,
`--c-router #ff5a8a`, `--c-motor #46f0a0`.
Semantic: `--gate-open #46f0a0`, `--gate-closed #ff5a8a`, `--reward-pos #46f0a0`,
`--reward-neg #ffb37a`, `--return-neg #ff8aa6`.

A small TS helper maps a region/pathway id → its hue token (e.g. `regionHue("sensory")` →
`var(--c-sensory)`), keyed off `header` so it stays data-driven, not hardcoded to five regions.

## 2. Shared `Panel` primitive

`dashboard/src/panels/Panel.tsx` — props `{ kicker: string, title: string, accent?: string,
children }`. Renders the card chrome (background `var(--panel)`, `1px solid var(--edge)`,
`border-radius: 11px`, `backdrop-filter: var(--blur)`, header with kicker + title + optional accent
dot). Every panel renders through it.

## 3. The five panels

Placement matches the comp: **left rail (316px): Panel 01 + Panel 04**; **center: hero (unchanged
in 1a)**; **right rail (336px): Panel 02 + Panel 03 + Panel 05**. `Shell` is updated to mount all
five.

All panels read `frames[envStep]` and `header` from the store reactively. `winTi` drives the
Panel 05 playhead. No new contract fields — every value already exists in the trace.

### Panel 01 — Region Activity (upgrade)
One row per `header.regions[]`. Adds, over the Phase-0 bars: a **rate sparkline** from
`regions[r].rate_t` (`[T]`), the numeric `regions[r].rate`, `active_frac`, spike count
(`regions[r].spikes`), and a status dot tinted by the region hue. Row label = `region.label`.

### Panel 04 — Thalamic Router State (new)
Header row `ACTION · UTILITY · GATE`; one row per `header.task.action_labels[a]`:
- label (uppercased),
- **utility bar** width = `router.utilities[a] / max(utilities)`, tinted region/neutral,
- **gate pill**: `router.gate_open[a] > 0.5` → text `OPEN`, bg `var(--gate-open)`, dark text; else
  `CLOSED`, bg `var(--gate-closed)`. Pill shows state text, not the number (reconciliation rule 2).
- Selected action (`task.action === a`) → brighter row background.
Accent dot uses `--c-router`. Footer: `selected action ▸ {action_label}`.

### Panel 02 — Inter-Region Communication Flow (new)
- **SVG network diagram** (viewBox ~`0 0 300 132`): a node per `header.regions[]` placed
  left→right in signal-flow order; a bézier edge per `header.pathways[]`, stroke colored by source
  region hue, `stroke-width`/`stroke-opacity` tracking `frame.pathways[p].intensity`,
  **`stroke-dasharray` when the pathway is gated and closed** (`gate_open <= 0.5`).
- **Pathway list** below: one row per `header.pathways[]` → `label` · gate tag · intensity.
  Gate tag: `gate_open > 0.5` → `OPEN` (`--gate-open`); `<= 0.5` → `CLOSED` (`--gate-closed`);
  the quiescent-by-design `sens_hippo` (`gate_open === 0`) → `STORE` (neutral `--edge`).
  For `pfc_motor` (per-action `gate_open` array) the tag aggregates: `max(gate_open) > 0.5` → OPEN.
Reads `header.pathways[]` + `frame.pathways[p].{intensity, gate_open}`.

### Panel 03 — Task State (upgrade)
Keep the `grid_n × grid_n` grid + `[data-cell]` markers. Add: larger cells, agent (filled, sensory
hue) and goal (ring, motor hue) markers, an action-direction arrow from `task.action_label`, and
**reward/return colored by sign** (`>= 0` → `--reward-pos`; reward `< 0` → `--reward-neg`,
return `< 0` → `--return-neg`).

### Panel 05 — Spike Raster (new)
Title `Spike Raster`, kicker `PANEL 05 · field`, tag `field · flash-only`. One **SVG strip per
neuron** for the **prefrontal** region's 4 neurons (hardcoded in 1a — see Deferred): vertical
spike marks where `field.prefrontal.spikes[ti][n] === 1` across `ti ∈ [0, T)`, plus a **playhead**
vertical line at the store's `winTi`. Footer: `t₀ · inference window · T={T} · t{T-1}`. Spike color
`--c-prefrontal`.

## 4. Testing

- **Unit (testing-library)**, one per panel, rendering against a loaded store with a synthetic
  frame and asserting the load-bearing output: Panel 01 sparkline + rate present; Panel 02 gate
  tags (OPEN/CLOSED/STORE) and a node/edge per topology entry; Panel 04 utility values + pill
  state + selected-row highlight; Panel 03 the `grid_n²` cells + sign-colored reward; Panel 05 one
  strip per prefrontal neuron + a playhead element. Follow the Phase-0 panel test style; mock
  nothing but the store.
- **e2e**: extend `e2e/smoke.spec.ts` to assert Panels 02/04/05 render on the real trace (e.g. a
  gate tag, an action label row, and the spike-raster panel are visible).
- `npx tsc -b` clean and `npm run build` succeeds (the standing React-task verification).

## 5. Deferred — noted, not forgotten

- **Panel 05 region selection** — make the inspected region selectable (click a Panel 01 row →
  drive Panel 05 via a `selectedRegion` store field, default `prefrontal`). Deferred from 1a; the
  panel hardcodes prefrontal for now. The "Selected Neuron" name anticipates this.
- **Membrane voltage traces** — Panel 05's forward-looking purpose. Needs a region-level
  `_record("membrane")` hook in the Python monitor first; reconciliation rule 3 keeps v1
  flash-only.
- Richer Panel 02 interactions (edge hover / inspect).

## 6. Out of scope (other slices)

- The hero (Flow Map + 3D Cloud treatments, overlays, agent receptive-field) — **slice 1b**.
- Focus mode, the Observatory/Clinical theme **toggle** + Clinical token set, full scrubber
  (slider/speed/labels), encoding key — **slice 1c**. (1a ships the Observatory tokens the toggle
  will later switch.)
- Live/WebSocket, trace registry, membrane — later phases per the platform roadmap.
