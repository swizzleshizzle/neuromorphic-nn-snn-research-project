# NEURO·SCOPE — Honest Pulse System (design)

**Date:** 2026-07-20 · **Slice:** Phase-1 "1c" follow-up (Mike-prioritized) · **Scope:** `dashboard/src/hero/Pathways.tsx` + `dashboard/src/hero/edges.ts`

## Problem

The hero's pathway pulses lie. In `Pathways.tsx`, pulse **position** is a wall-clock loop:

```ts
const t = clock.elapsedTime;
let pp = (t * 0.18 + k / PULSES + ei * 0.21) % 1;   // motion is decorative, not data
```

Only pulse opacity/size is data-driven (`0.06 + inten * 0.3`). Consequences:
- Pulses keep travelling when playback is **paused**.
- Pulses keep travelling on every non-gated edge even when **nothing is flowing** (low intensity).

A viewer cannot read pulse motion as genuine inter-region communication, and quiet pathways do not look
quiet. This is the same class of defect the honest-dashboard work fixed elsewhere: the UI animating
activity that isn't real.

## Goal

Pulse motion should depict **real activity propagating src→dst within the inference window**, so that:
- pulses **freeze when playback is paused**, and move at the data tempo when playing;
- a **quiet pathway is visibly still**, not just dim.

## Data available (grounded in the real trace)

- Store playback state: `winTi` (0..T-1, the current timestep in the T-window; advances during play,
  frozen when paused), `T`, `envStep`, `playing`. `winTi` is the honest "playhead."
- Per-pathway per-frame: `PathwayState.intensity` (real range 0–0.40, mean ~0.20) + `gate_open`.
- (`field[region].spikes` is `[T][N]` per-timestep, but this design does **not** use it — see Non-goals.)

## Design

### 1. Phase from the playhead, not the wall clock

Derive pulse phase from `winTi`:

```
base = T > 0 ? winTi / T : 0     // sweeps 0→1 as the window plays out
```

A pulse thus travels src→dst once per inference window, at the playback tempo. Because `winTi` is frozen
while paused, pulses freeze while paused — with no explicit `playing` check. `clock` is removed from
`useFrame`; wall-clock no longer influences motion.

The honest semantic: a pulse crossing (e.g.) sensory→PFC over the sweep *is* activity propagating through
the T-window.

### 2. Intensity threshold → quiet edges are still

An edge draws pulses only when `!quiescent && intensity >= THRESH` (`THRESH = 0.05`). Below threshold the
edge shows only its dim static line (zero pulses). Above threshold:

- **pulse count scales with intensity**: 1 pulse near threshold, up to `PULSES` (3) at high intensity;
- **pulse size stays ∝ intensity**: the existing `0.02 + inten * 0.05` (additive blending → bigger reads
  as brighter).

So: quiet = still + dim line; active = flowing + denser + bigger.

### 3. Pure, testable helpers (the `<Canvas>`-not-jsdom-testable constraint)

R3F `<Canvas>` cannot render in jsdom, so the math lives in pure functions in `edges.ts`, unit-tested;
`Pathways.tsx` calls them inside `useFrame`. Two new exports:

```ts
// 0 below threshold; otherwise 1..maxPulses, monotonically increasing in inten.
export function pulseCount(inten: number, thresh: number, maxPulses: number): number;

// Position in [0,1) along the edge for pulse k of `count`, given the playhead.
// winTi=0 → base 0; identical winTi → identical result (freeze); always wrapped into [0,1).
export function pulsePhase(winTi: number, T: number, k: number, count: number, edgeOffset: number): number;
```

`edgeOffset` is a **static** per-edge phase offset (derived from the edge index) used **only** for visual
spacing so active edges don't pulse in perfect unison. It is cosmetic and encodes no data (we have no
per-edge propagation-delay signal); this is documented at the call site.

### 4. `Pathways.tsx` changes

- Read `winTi` and `T` from the store (alongside the existing `frames`, `envStep`).
- Drop `clock` / `t`.
- Per edge: compute `count = pulseCount(st.inten, THRESH, PULSES)`; if `count === 0` (or `st.quiescent`),
  draw no pulses; else place `count` pulses at `pulsePhase(winTi, T, k, count, edgeOffset)` via the
  existing `quadPoint`, size `0.02 + st.inten * 0.05`.
- Keep the existing unused-instance parking (scale 0 at origin) for all instances not filled this frame.

### Unchanged

Line rendering (color via `hueFor`, dashed-when-gated via `gapSize`, opacity), morph/bow, src→dst
direction, `SEGMENTS`, the `PULSES = 3` instance capacity, and everything outside these two files.

## Testing

- **Unit (vitest, `edges.test.ts`):**
  - `pulseCount`: `inten < thresh → 0`; `inten === thresh → 1`; high `inten → maxPulses`; monotonic
    non-decreasing across increasing `inten`.
  - `pulsePhase`: `winTi = 0 → 0` (for `k = 0`); identical `winTi` → identical output (freeze proof);
    result always in `[0, 1)` including for `k = count-1` and wrap cases.
- **e2e (Playwright):** existing hero smoke stays (renders without crash). It cannot assert motion; no new
  e2e is added rather than over-invest in brittle animation assertions.
- **Gates:** `npx tsc -b` clean (ES2020 target — no ES2022 APIs), `npx vitest run` green, `npm run build`
  succeeds.

## Non-goals / deferred

- **Event-driven emission from `field[src].spikes[winTi]`** (per-timestep real spikes). Considered; the
  per-frame `intensity` proxy is sufficient for the honesty goal and far less wiring. Left as a possible
  future upgrade.
- **NeuronField `(1-frac)` flash-decay** — a separate, smaller hero-honesty polish; its own follow-up.
- No changes to the contract, store shape, Python monitor, or any panel.

## Success criteria

1. Pausing playback freezes all pulses; resuming moves them at the data tempo.
2. A pathway with `intensity < THRESH` shows no pulses (only its static line); higher-intensity pathways
   show more/bigger pulses.
3. `tsc -b` + vitest + build all green; the two helpers are unit-tested.
