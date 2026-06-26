# NEURO·SCOPE Phase 1b: Hero Parity (real-3D) Design

**Date:** 2026-06-25
**Status:** Approved (pending implementation plan)
**Slice:** Phase 1 / 1b (hero). Follows 1a (panels parity, shipped). Precedes 1c (chrome).
**Design source:** `docs/handoffs/claude_design/NEURO-SCOPE Dashboard.dc.html` (the hero `_layout` / `_drawCloud` / `_drawFlow` / `_edgeState` / `_neuronGlow` code) and the platform spec `docs/superpowers/specs/2026-06-18-neuroscope-platform-design.md` (§6).

---

## 1. Goal

Bring the NEURO·SCOPE web hero from the minimal Phase-0 placeholder (an instanced-sphere cloud with spike flash) to a full, "go big" centerpiece: a real three.js 3D neuron cloud with glowing additive neurons, pathway edges and travelling pulses, region and motor labels, a sensory-input overlay, and a smooth camera-morph transition to a flat Flow Map, switchable by an on-stage Cloud/Flow toggle.

The hero must stay **data-driven**: it renders whatever the trace header declares (regions, neuron counts, pathways), never hardcoding today's five regions.

### Why real 3D (not the comp's 2D canvas)

The comp produces its 3D look with a hand-rolled 2D-canvas renderer (manual projection, painter's-algorithm depth sort, additive `lighter` compositing, depth-alpha fog). We are instead building on real R3F/three.js because the hero is a long-term centerpiece we intend to grow (grab-to-spin interactivity, future post-processing, richer scenes). Real 3D gives us a real depth buffer, real fog, and free user-orbit; it drops the comp's manual `_project` math entirely. The cost is reproducing the comp's additive-glow aesthetic with three.js primitives, which §4 addresses with per-neuron additive sprites.

## 2. Scope

**In 1b:**
- 3D Cloud treatment (neurons, glow, spike flash, fog, auto-rotate + drag-to-spin camera).
- Pathway edges with travelling pulses; dashed/dimmed when gated closed.
- Region labels and motor action labels.
- Flow Map treatment (flat left-to-right layout, arced bezier edges).
- Camera-morph transition between Cloud and Flow, driven by an on-stage Cloud/Flow toggle.
- Sensory-input grid overlay and hero caption (DOM, over the canvas).

**Explicitly NOT in 1b (stays 1c):**
- Encoding-key overlay (lives visually in the hero, but deferred to 1c with the rest of the chrome).
- Theme toggle (Observatory / Clinical), focus mode, full header chrome.
- Knob UI for `glowStrength` / `showTrails` / `propagationSpeed` (defaults are hardcoded in 1b).

## 3. Architecture

### Files

```
dashboard/src/hero/
  Hero.tsx          # R3F component tree (replaces the Phase-0 minimal hero)
  layout.ts         # buildCloudLayout, buildFlowLayout, clusterCentroids (+ existing isSpiking kept/renamed)
  edges.ts          # buildEdges, edgeState, pulse position helpers
  sensory.ts        # aggregateSensoryGrid
  glow.ts           # neuronGlow, morph/lerp helpers (or fold into layout.ts)
  overlays/         # DOM overlays: SensoryGrid.tsx, HeroCaption.tsx, CloudFlowToggle.tsx
```

Exact file split is a plan-stage detail; the contract is: **all geometry and frame-state logic lives in pure, unit-tested functions; the R3F component only wires them to three.js objects.**

### Dependencies

- Add `@react-three/drei` (`OrbitControls`, `<Html>`). `three` and `@react-three/fiber` are already installed (Phase 0).
- Do **not** add `@react-three/postprocessing` in 1b. Bloom is a future enhancement; per-neuron additive sprites (§4) get the comp's look without a post pass.

### Imperative / reactive split (load-bearing, preserved from Phase 0)

- A single `useFrame` loop reads `useTraceStore.getState()` each frame and drives three.js objects **imperatively**: instanced neuron matrices and colors, pulse sprite positions, edge geometry, and camera easing. It never calls `setState` and never re-renders React.
- Only the **DOM overlays** (region/motor labels via drei `<Html>`, sensory grid, caption) subscribe reactively, and only to `envStep` (per-step, not per-frame), exactly as the panels do today.
- The Cloud/Flow toggle writes a target into a ref (or a dedicated lightweight store field); the `useFrame` loop eases the live `morph` value toward it. Flipping the toggle does not, by itself, re-render the canvas tree.

## 4. The Cloud treatment

### Layout (`buildCloudLayout(header)`)

Port the 3D branch of the comp's `_layout`. For each region, choose a shape by neuron count:
- `count <= 8` → **column** (small ring/stack).
- `count` is a perfect square → **grid**.
- otherwise → **phyllotaxis disc** (golden-angle `2.39996323`).

Regions are spread along X by index (`-1 .. 1`), with alternating Y and stepped Z offsets per region so clusters do not overlap in depth. The whole point set is **mean-centered** (subtract the centroid of all neurons) so the cloud orbits about its own middle. Each neuron returns `{region, idx, cloudPos:[x,y,z], r3}` where `r3` is a base radius hint by shape.

`clusterCentroids(neurons, 'cloudPos')` returns the per-region centroid, used for edge endpoints and label anchors.

### Neurons (rendering)

- One `InstancedMesh` of additive point-sprites: a small plane/sphere with a **soft radial-gradient alpha texture**, `blending: THREE.AdditiveBlending`, `depthWrite: false`, `toneMapped: false`. Color per instance = region hue (the `--c-<region>` palette).
- Per frame, for each neuron: compute `{sp, act} = neuronGlow(frame, region, idx, ti, T)`. Scale = base `r3` times a flash factor; on spike (`sp`), scale up briefly and add a white core flash whose intensity decays across the window fraction (`1 - frac`). Instance alpha/brightness = `min(1, act + flash*0.95)`, matching the comp.

### `neuronGlow(frame, region, idx, ti, T)`

Port the comp's `_neuronGlow`: base activity `0.06`, `+0.5` if spiking at `ti`, `+0.18` if spiking at `ti-1` (wrapped mod T). Returns `{sp, act}`. Pure, unit-tested.

### Camera

- drei `OrbitControls` with `autoRotate: true` (rate tuned to the comp's slow drift), damping enabled, and **user drag enabled** (the long-term "go big" hook). `enablePan: false`, sensible zoom clamps.
- Real `THREE.Fog` (or `fogExp2`) from the theme background color, so far neurons fade with depth (replaces the comp's depth-alpha hack).

### Pathways (`buildEdges`, `edgeState`)

- `buildEdges(header)` returns one edge per `header.pathways[]` with `{id, src, dst, gated}`. Endpoints are resolved at render time from the current (possibly morphing) centroids.
- `edgeState(pathwayFrame, gated)` ports `_edgeState`: returns `{inten, open, quiescent}`. `gate_open` may be a scalar or array; take the max as the fraction; `open = frac > 0.5`; `quiescent = gated && !open`.
- Edge rendering: additive line between endpoints; width/alpha grow with `inten`. When `quiescent`, draw **dashed and dimmed** instead.
- **Travelling pulses:** along each non-quiescent edge, a few small additive sprites move from src to dst; their count, size, and alpha scale with `inten`. Pulse phase advances with time (the comp uses `~0.05 * time + k/count + edge.off`).

### Labels

- Region labels: drei `<Html>` anchored at each cloud centroid, showing the region name and `N · rate` (rate from `frame.regions[id].rate`), styled with existing theme tokens (Space Grotesk / IBM Plex Mono, region color).
- Motor action labels: at each motor neuron, the action label plus a marker: `◂` if selected, `✕` if its gate is shut (`gate_open[a] <= 0.5`). Driven by `frame.router.gate_open` and `frame.task.action`.

## 5. Flow Map + camera morph

### Flow layout (`buildFlowLayout(header)`)

Port the 2D branch of the comp's `_layout`: the same per-region shapes (grid / disc / column), arranged left-to-right in a plane at `z = 0`, returning `{flowPos:[x,y,0]}` per neuron. `clusterCentroids(neurons, 'flowPos')` gives flow centroids for flow edges and labels.

### The morph scalar

- A single `morph` value in `[0,1]` (`0` = full Cloud, `1` = full Flow), stored in a ref / lightweight store field. The toggle sets a **target**; `useFrame` eases the live value toward it (e.g. critically-damped or simple exponential approach), so switching is a smooth transition, never a hard cut.
- **Neuron position** each frame = `lerp(cloudPos, flowPos, morph)`. A pure `lerpVec3` helper is unit-tested.
- **Camera:** as `morph → 1`, `autoRotate` ramps to off and the camera eases (position + target) to a locked front-on framing; orbit is disabled at `morph == 1`. As `morph → 0`, the camera eases back and auto-rotate resumes. The same scalar drives the camera ease.
- **Edges:** endpoints follow the live (lerped) centroids. The bezier control offset (arc "bow") blends in with `morph`: a straight 3D segment at Cloud, an arced 2D bezier at Flow. Pulses ride whatever the current curve is, so flow continues through the transition.

### Toggle UI (`CloudFlowToggle`)

- A small segmented `Cloud / Flow` control rendered **on the hero stage** (absolutely positioned, top-right of the canvas), not in the header. The header segmented-button cluster (theme / focus / dashboard) belongs to 1c.
- Clicking sets the morph target. The control reflects the current target as its active segment.

## 6. Overlays (DOM, over the canvas)

Absolutely positioned over the canvas, reactive to `envStep` only (not per-frame):

- **Sensory-input grid** (top-left): a `gridN × gridN` grid. `aggregateSensoryGrid(encoding)` sums `encoding.sensory_input.spikes` over the window to find the most-active agent cell and goal cell, returning `{agentCell, goalCell}` (or null when absent). The agent cell is filled (sensory hue, glow); the goal cell is ring-outlined (motor hue). Pure, unit-tested.
- **Hero caption** (top-left): `NEURON FIELD` plus a mode-dependent caption ("distributed neuron cloud · field spikes · auto-orbit" for Cloud; the `region → region → ...` flow string for Flow).

## 7. Background and dropped effects

- Background is the solid theme `--bg` plus fog. 
- The comp's faint background grid lines and trail-fade (translucent fill over the prior 2D frame) are **dropped** for 1b: trail-fade does not translate to a real-3D scene, and additive glow + flash decay + fog carry the motion feel. A subtle 3D grid is an optional later polish, out of scope here.

## 8. Defaults for deferred knobs

Hardcode in 1b (knob UI is later): `glowStrength = 1`, `showTrails` effectively off (no trail pass), `propagationSpeed = 1` (so window-advance and pulse timing match Phase 0).

## 9. Testing

- **Unit (Vitest):** every pure helper: `buildCloudLayout`, `buildFlowLayout`, `clusterCentroids`, `buildEdges`, `edgeState`, `neuronGlow`, `aggregateSensoryGrid`, `lerpVec3` / morph easing. Assert structure, counts, ordering, centering, the `gate_open > 0.5` rule, spike afterglow, and sensory aggregation against a small synthetic trace.
- **Component isolation:** keep `vi.mock("./hero/Hero", () => ({ Hero: () => null }))` in `App.test.tsx`. The real `<Canvas>` crashes jsdom (no WebGL / ResizeObserver), so the hero is never mounted in unit tests.
- **E2E (Playwright):** extend `e2e/smoke.spec.ts` to assert the hero canvas mounts on the real trace and that the on-stage Cloud/Flow toggle is present and switches the active segment. Visual fidelity is verified by manual sign-off.
- **Typecheck:** `npx tsc -b` as a required gate (Vitest/esbuild does not typecheck).

## 10. Acceptance criteria

1. On the real trace, the hero renders a live 3D neuron cloud: regions in distinct hues and shapes (grid/disc/column by count), neurons glowing and flashing on field spikes, fading with depth.
2. The camera auto-rotates and can be dragged to spin.
3. Pathway edges connect region clusters with travelling pulses scaled by intensity; gated-closed pathways render dashed and dim.
4. Region labels (name + `N · rate`) and motor action labels (selected `◂`, gated-shut `✕`) track the current step.
5. The on-stage Cloud/Flow toggle smoothly morphs neuron positions, edges, and camera between the 3D cloud and the flat flow map.
6. The sensory-input grid overlay and hero caption reflect the current step.
7. The hero is fully data-driven (no hardcoded region ids in render logic).
8. All pure helpers are unit-tested; `npm test` green, `npx tsc -b` clean, `npm run build` succeeds, Playwright smoke passes.

## 11. Risks / open notes

- **Edge morph fidelity:** blending a straight 3D line into a 2D bezier while endpoints also lerp is the trickiest piece; if it proves fiddly, an acceptable fallback is to cross-fade edge *opacity* across the morph (fade cloud edges out, flow edges in) rather than geometrically morphing a single curve. Decide during implementation; either satisfies criterion 5.
- **drei `<Html>` label volume:** many labels can cost layout; we have only one per region plus motor labels, so this is fine, but keep labels out of the per-frame path (update on `envStep`).
- **Camera control choice:** `OrbitControls` + manual camera ease is the baseline. If the morph ease fights `OrbitControls`, drei `CameraControls` (smooth programmatic `setLookAt`) is the fallback. Implementation detail, not a contract change.
