# Stage-2 Dashboard Design (L17)

**Date:** 2026-06-17
**Phase:** 2 — Multi-region brain
**Status:** Design locked — ready for layout/visual handoff to Claude Design
**Author:** brainstormed in session 2, Week 11

---

## 1. Purpose & scope

The base of a long-term **brain monitoring system**, not a one-off dashboard. It must:

- Give complete debug visibility into a running brain (no lapse in information verbosity).
- Be **presentable** — usable in front of other people / the capstone video, not raw data dumps.
- Be **extensible to a set of brains**: many brains over many iterations, trained for different
  tasks, with different region counts and neuron counts. The dashboard renders whatever a trace
  *declares* — it never hardcodes today's five regions.
- Be modular but concise and strongly built.

This spec covers the **architecture, data contract, and panel inventory**. It deliberately stops
short of layout geometry and visual conventions (color system, exact grid), which are handed to
**Claude Design**. See §8 for the handoff brief.

### Non-goals (YAGNI for this stage)

- Cross-run comparison / "compare 50 episodes" analytics (later; the file store enables it).
- Training-time live metrics dashboards (TensorBoard/W&B already cover scalar curves).
- Multi-agent (`B > 1`) visualization. A trace is one agent; multi-agent gets a `lane` field later.
- Standing up Redis. The contract is designed so it slots in later with zero changes (§3).

---

## 2. Architecture overview

```
  Brain.step()  ──►  Frame (transport-agnostic object)  ──►  TraceSink
                                                              ├─ FileSink        (always on — system of record)
                                                              ├─ WebSocketSink   ("toward streaming" — direct, no infra)
                                                              └─ RedisStreamSink (drop in when N-sims/M-viewers arrive)

  Dashboard (React + WebGL hero + D3/Recharts panels)  ◄── reads trace file OR WebSocket frames
```

**Layering principle:** files are the **datastore / system of record** (durable, diffable,
shareable, reproducible). Redis, when it arrives, is **live transport only** — never the store.
The `TraceSink` interface makes file vs. WebSocket vs. Redis a config swap, not a rewrite.

**Tech stack:** React + a custom viz layer. **WebGL** (pixi/regl/three.js) for the neuron-field
hero — chosen day one to avoid the SVG/canvas scaling wall (see §6). D3 for bespoke visuals
(network flow, grid overlay, router gates); Recharts for standard time-series (membrane traces,
rate curves). Grafana and live matplotlib were rejected: neither does the bespoke, presentable
visuals this platform needs.

---

## 3. Data contract

### 3.1 Serialization — JSONL

- One **trace header** per run (schema below) + an ordered list of **Frames**.
- On disk: **JSONL** — header as line 0 (or a `.meta.json` sidecar), one Frame per line.
- **The unlock:** appending a line to a file and sending a frame over WebSocket are the *identical
  operation on the identical object*. Batch and stream are the same code path. This is how we
  "build toward streaming" for free.
- Parquet export can be added later for bulk analytics; JSONL is the live/replay format.

### 3.2 `TraceSink` interface

A single interface with interchangeable implementations:

- `FileSink` — append each Frame as a JSONL line. Always on; the system of record.
- `WebSocketSink` — emit each Frame to connected clients. Built this stage (direct, no infra).
- `RedisStreamSink` — publish each Frame to a per-brain Redis Stream (durable, replayable,
  consumer groups). Added the day multi-process fan-out (N sims → M dashboards) is real.

### 3.3 Tier 1 — Trace Header (once per run)

Declares the brain's **topology** and run context. This is what makes the dashboard data-driven:
a different brain (more regions, different task) lights up the same UI with no code changes.
Panels read the header for structure; Frames for state.

```json
{
  "schema_version": "1.0",
  "brain": { "id": "five-region", "config_hash": "ab12…", "seed": 0, "T": 32 },
  "task":  { "type": "gridworld", "grid_n": 5,
             "action_labels": ["up","down","left","right"] },
  "regions": [
    { "id": "sensory",     "label": "Sensory Cortex",  "n_neurons": 64,  "role": "input",    "render": "dots" },
    { "id": "hippocampus", "label": "Hippocampus",     "n_neurons": 150, "role": "memory",   "render": "dots" },
    { "id": "prefrontal",  "label": "Prefrontal",      "n_neurons": 4,   "role": "planning", "render": "dots" },
    { "id": "router",      "label": "Thalamic Router", "n_neurons": 4,   "role": "control",  "render": "dots" },
    { "id": "motor",       "label": "Motor Cortex",    "n_neurons": 4,   "role": "output",   "render": "dots" }
  ],
  "pathways": [
    { "id": "sens_hippo", "src": "sensory",     "dst": "hippocampus", "gated": true,  "label": "store/recall" },
    { "id": "sens_pfc",   "src": "sensory",     "dst": "prefrontal",  "gated": false },
    { "id": "hippo_pfc",  "src": "hippocampus", "dst": "prefrontal",  "gated": true },
    { "id": "pfc_motor",  "src": "prefrontal",  "dst": "motor",       "gated": true, "label": "router-gated" }
  ]
}
```

- `regions[].render` — representation hint (`"dots" | "cloud" | "density"`), defaulting off
  `n_neurons` (see the scaling ladder, §6). A bigger region just declares its size and the hero
  picks the mode. No rewrite.

### 3.4 Tier 2 — Frame (one per environment step)

One `Brain.step()` = one Frame. Each field is either a cheap scalar summary (always on; drives
gauges/flow/router/task) or a `[T, …]` time-series over the inference window (drives traces/raster/
hero). Fields are annotated with the panel they feed.

```json
{
  "episode": 3, "step": 17, "t": 1.42,

  "task": {                                         // → Panel 3 (Task Overlay)
    "agent": [2,4], "goal": [0,1],
    "action": 2, "action_label": "left",
    "reward": -0.01, "return": -0.18,
    "terminated": false, "truncated": false
  },

  "regions": {                                      // → Panel 1 (Activity gauges)
    "sensory":     { "rate": 0.21, "spikes": 430, "active_frac": 0.34, "rate_t": [/*T*/] },
    "hippocampus": { "rate": 0.08, "spikes": 384, "active_frac": 0.12, "rate_t": [/*T*/] },
    "prefrontal":  { "rate": 0.55, "spikes": 70,  "active_frac": 0.75, "rate_t": [/*T*/] },
    "router":      { "rate": 0.40, "spikes": 51,  "active_frac": 0.50, "rate_t": [/*T*/] },
    "motor":       { "rate": 0.62, "spikes": 79,  "active_frac": 0.25, "rate_t": [/*T*/] }
  },

  "pathways": {                                     // → Panel 2 (Flow edge intensity)
    "sens_hippo": { "intensity": 0.21, "gate_open": 1.0 },
    "sens_pfc":   { "intensity": 0.21 },
    "hippo_pfc":  { "intensity": 0.08, "gate_open": 1.0 },
    "pfc_motor":  { "intensity": 0.33, "gate_open": [1,0,1,1] }   // per-action; = 1 - gate_closed
  },

  "router": {                                       // → Panel 4 (Router State)
    "gate_open":   [1,0,1,1],                        // which actions can pass (= 1 - gate_closed)
    "gate_open_t": [/* [T, n_actions] */],
    "utilities":   [0.7,0.1,0.5,0.6]                 // PFC action utilities, pre-gate (kept here so
                                                     //   the panel shows utility vs. what got gated)
  },

  "field": {                                        // → Hero (always on, lightweight: bits)
    "sensory":     { "spikes": [/* [T, 64]  */] },
    "hippocampus": { "spikes": [/* [T, 150] */] },
    "prefrontal":  { "spikes": [/* [T, 4]   */] },
    "router":      { "spikes": [/* [T, 4]   */] },
    "motor":       { "spikes": [/* [T, 4]   */] }
  },

  "detail": {                                       // → Panel 5 (Membrane traces) — opt-in per region
    "prefrontal": { "neuron_ids": [0,1,2,3],
                    "membrane": [/* [T, n] */],
                    "spikes":   [/* [T, n] */] }
  }
}
```

### 3.5 Verbosity tiers (controlled, never lossy by accident)

| Tier | Content | Cost | Default |
|---|---|---|---|
| Summary | `regions[].rate`, `pathways[].intensity`, `router.gate_open`, `task` | tiny | always on |
| `field` | per-neuron **spikes** `[T, n]`, all regions | bits (~7 KB/frame for 226 neurons) | always on |
| `detail` | per-neuron **membrane** + spikes `[T, n]` | heavier | opt-in via `record_detail: [...]` |

Full five-region `detail` for an episode is only ~1–2 MB, so everything *can* be recorded when
debugging — but you control what each live frame pays. No lapse in verbosity; just verbosity you
choose.

### 3.6 Contract conventions

- **`gate_open = 1 - gate_closed`.** `Brain` outputs `gate_closed` (a mask); the sink inverts it
  once at write time so the UI only ever reasons about "what's flowing."
- **One agent per trace** (`B = 1`). The sink fixes the batch index. Multi-agent → future `lane`.
- **`utilities` lives under `router`**, not `prefrontal` — the router panel needs utility-vs-gated
  side by side. (Locked per session decision.)

---

## 4. Panel inventory

The five required panels (L17 brief) plus the hero. All are **driven by the header topology**, not
hardcoded region names.

| # | Panel | Reads | Purpose |
|---|---|---|---|
| Hero | **Neuron Field** (centerpiece) | `field`, `pathways[].intensity`, header `regions`/`pathways` | Every neuron rendered with real spike data; watch signal propagate region→region |
| 1 | Region Activity Overview | `regions[].rate / active_frac / rate_t` | Spike-rate gauge per region; spot dead/runaway regions |
| 2 | Inter-Region Communication Flow | `pathways[].intensity / gate_open`, header `pathways` | Network diagram, edge intensity = signal flow |
| 3 | Task State Overlay | `task`, header `task` | Grid-world state synced to neural activity |
| 4 | Thalamic Router State | `router.gate_open(_t)`, `router.utilities` | Which action pathways are open/closed; utility vs. gated |
| 5 | Selected Neuron Membrane Traces | `detail[region].membrane / spikes / neuron_ids` | Per-neuron membrane V over the T-window |

---

## 5. The Neuron-Field Hero (centerpiece)

The differentiator: the whole brain is **226 neurons** (64 + 150 + 4 + 4 + 4) — small enough to
render *every neuron with its actual spikes*. The honest view and the "cool" view are the same
view. (Contrast the flashy systems that fake it because real cortex is ~10⁹ neurons.)

**Meaningful layout, not a blob:**

- Five **region clusters** in signal-flow order: sensory → hippocampus → prefrontal → router → motor.
- Each neuron: **glow = membrane V** (when `detail` present), **flash = spike** (from `field`),
  animated across the T-window, advancing per env step.
- **Inter-region edges pulse** with `pathways[].intensity` — watch a sensory pattern propagate →
  trigger hippocampal recall → light PFC → get gated by the router → fire motor. Timing bugs (dead
  region, stuck gate) become visible.
- The 64 **sensory neurons laid out as the grid itself**, tying the hero to Panel 3.

**Fusion role:** the hero is Panel 1 (activity) + Panel 2 (flow) combined as the centerpiece; the
other panels are instruments around it.

**Focus mode:** the hero expands to fill the stage ("fullscreens" naturally), pushing the other
panels out to **collapsible side rails** you can pull back. Layout shell = central stage + dockable
side panels + a bottom timeline scrubber.

**Build risk:** this is the highest design-effort, highest-risk element. Scaffold the static layout
+ one animated step first; wire real playback after the boring panels work. Timebox it.

---

## 6. Scaling — neuron-count ladder

The hero's **representation is a function of N**, declared per region in the header (`render`), so
adding regions / growing neuron counts in future brains does not require a rewrite.

| Engine / limit | Bites at | Notes |
|---|---|---|
| SVG / D3 dots | ~1,000 | DOM-bound. **Not** the hero engine. |
| Canvas 2D | ~10,000 | CPU draw calls. |
| **WebGL point sprites** | **~100K easily, ~1M with care** | One buffer, one draw call. **Build on this.** |
| Data/transport (`field`) | **~tens of thousands** | N×T as 1 bit: 10K→~40 KB/frame, 100K→~400 KB/frame. Caps before rendering, esp. live. |
| Human perception | ~2,000–5,000 | Beyond this, individual dots stop being legible — a density field is *correct*, not a compromise. |

**Representation ladder (auto-selected from `n_neurons`):**

- `N ≤ ~2K` → **dots**: individual neurons, glow + flash. The literal view. (All foreseeable
  near-term brains.)
- `N ~2K–100K` → **cloud**: WebGL point cloud, still per-neuron as a field; pan/zoom to inspect.
- `N > ~100K` → **density** (`render: "density"`): the *sink* stops shipping per-neuron `field`
  data and emits an aggregated population field (binned heatmap/density) with drill-down to a
  sampled subset.

**Verdict:** no regret. This is an snnTorch SNN on a single machine; realistic future iterations
land in the thousands–tens-of-thousands range — squarely in the WebGL per-neuron tier. The literal
view holds for everything plausible; the aggregate ladder covers the rest. Downsampling lives in
the *data contract*, not just the renderer — which is also exactly what neuroscience does at scale.

---

## 7. Build order (Saturday + beyond)

1. **`Frame` schema + `TraceSink` (FileSink)** — emit conforming JSONL from `Brain.run_episode`
   with `record=True`. The contract is the foundation; everything reads it.
2. **Dashboard shell** — layout stage + collapsible rails + bottom timeline scrubber; load a trace
   file and replay (batch first, per the brief).
3. **Boring panels first** — 1 (gauges), 3 (task overlay), 4 (router), 2 (flow), 5 (membrane).
   These de-risk the data wiring before the hero.
4. **Hero** — static layout + one animated step in Claude Design, then real playback.
5. **WebSocketSink** — same Frame object over the wire; flip the dashboard from file-replay to live.
6. **Later, on demand:** RedisStreamSink (multi-process fan-out), Parquet export (cross-run
   analytics), the `cloud`/`field` representations (only when a brain gets big enough to need them).

---

## 8. Handoff brief for Claude Design

What this spec deliberately leaves open, for Claude Design to define:

- **Layout geometry** — the panel grid around the hero; default sizes; the focus-mode expand/collapse
  behavior and the side-rail pull-back interaction; the bottom timeline scrubber.
- **Visual / color conventions** — encodings for spike rate, membrane voltage, gate open/closed,
  pathway intensity, reward sign; the overall palette and "presentable but information-dense"
  aesthetic; typography and density.
- **Hero visual treatment** — neuron glow/flash styling, region-cluster arrangement, edge-pulse
  animation, the sensory-as-grid layout.

Constraints to honor: data-driven from the header topology (no hardcoded regions); WebGL hero;
React + D3/Recharts panels; legible at high information density; works in focus mode and full
dashboard mode.
