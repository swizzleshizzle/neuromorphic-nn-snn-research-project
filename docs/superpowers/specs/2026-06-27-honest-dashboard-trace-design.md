# Honest Dashboard + Trained-Policy Trace Design

**Date:** 2026-06-27
**Status:** Approved (pending implementation plan)
**Session:** Week 12, Hands-On Track 1 (dashboard)
**Context:** v1 trains only a linear head on the frozen sensory concept (ADR-0001 Amendment 1). EXP-024 confirmed v1 navigates but is capped at ~30-50% by representational capacity; four of five regions are frozen spectators.

## 1. Problem

Two defects make the dashboard dishonest and unusable for "visualize a training trace":

1. **Recorded traces never solve the task.** `monitor/runner.py::record_episode` drives actions from `out["action"]` (the brain's internal degenerate motor readout from Session 1), NOT the trained `nn.Linear` head on the sensory concept. So every recorded trace shows the brain failing, regardless of how well the head trained. Both existing traces hit 0 goal-reaching frames.
2. **The trained policy cannot be recorded.** The trained policy runs `recall=False` (memory bypassed), but `build_frame` reads `out["recordings"][region]` for all five regions and KeyErrors when the hippocampus recording is absent under `recall=False`. The existing workaround records with `recall=True`, which runs a different (non-trained) policy.

Separately, the dashboards imply all five regions are learning, when only sensory feeds the policy. The visualization oversells what is functional.

## 2. Goals

- Record a trace of the **actual trained policy** (head-driven, `recall=False`) that **reaches the goal**.
- Make the dashboards **honest**: mark which regions are on the policy path vs spectators, data-driven.
- Add **PNG export** as a technical feature (the media/content track is dropped; no content framing).

Non-goals: engaging or unfreezing the brain (that is Track 2 / Option B); changing the learning architecture.

## 3. Components (dependency order)

### 3.1 Trace-path fix (`src/neuromorphic/monitor/runner.py`, `src/neuromorphic/training/reinforce.py`)

- Promote `_concept_rate(out)` to a public `concept_rate(out)` in `reinforce.py` (same definition: `out["concept"].mean(dim=0)[0]`); keep a thin `_concept_rate` alias if anything imports the private name.
- New `record_policy_episode(brain, head, env, sink, *, seed=0, action_labels=DEFAULT_ACTION_LABELS, max_steps=None, recall=False, policy_regions=("sensory",), generator=None) -> dict`:
  - Open the sink with a header that includes `policy_regions` (section 3.2).
  - Per step: `out = brain.step(obs, store=False, recall=recall, record=True, generator=generator)`; `action = int(head(concept_rate(out)).argmax())`; step the env with that action; pad bypassed recordings (below); `build_frame(out, ...)`; write.
  - Returns the same summary dict shape as `record_episode` (`steps`, `total_reward`, `reached_goal`).
- **Bypassed-region padding:** a helper (in `runner.py`) ensures every region in `schema.REGION_OUTPUT_KEY` has an entry in `out["recordings"]` before `build_frame`. For a missing region it injects a zero spike train shaped `[T, B, N]`, where `T` is taken from any present region's recording and `N` is that region's neuron count from the brain. The bypassed region then renders as silent. `build_frame` itself is unchanged.

### 3.2 Honest annotation mechanism (`src/neuromorphic/monitor/schema.py`, `dashboard/src/contract.ts`)

- `build_header(...)` gains a `policy_regions: list[str]` parameter (default `[]`) and includes it in the emitted header. `record_policy_episode` passes `list(policy_regions)`.
- The frontend `TraceHeader` interface gains `policy_regions?: string[]`.
- A region is a **spectator** iff its id is not in `policy_regions`. Both dashboards derive the spectator set this way (no hardcoded region ids), so it updates automatically when Track 2 puts more regions on the policy path.

### 3.3 Regenerate the goal-reaching trace (`experiments/023_week11_brain_training/run.py`)

- Replace the post-training `record_episode(brain, env, sink, recall=True, ...)` call with `record_policy_episode(brain, head, env, sink, recall=False, policy_regions=("sensory",), seed=args.seed, generator=gen)`. After training to 100% on the fixed goal, the greedy head solves it, so `outputs/week11_trained_trace.jsonl` becomes a goal-reaching trace (terminated frame present).

### 3.4 Batch matplotlib dashboard (`src/dashboard/multi_region_viz.py`)

- Read `header.get("policy_regions", [])`. On each per-region raster panel, tag the title: regions in `policy_regions` get "● on policy path"; spectators get "○ spectator (frozen)". Add a single figure caption noting that only policy-path regions learn in v1. PNG output is unchanged (export is inherent).

### 3.5 NEURO·SCOPE web (`dashboard/`)

- **Annotation:** `RegionActivity` (and/or the hero region labels) shows a small "spectator" badge for regions not in `header.policy_regions`. Reads the new contract field; data-driven.
- **Export PNG:** an "Export PNG" control that downloads the hero canvas as a PNG. The R3F `<Canvas>` is created with `gl={{ preserveDrawingBuffer: true }}` so `canvas.toDataURL("image/png")` yields a non-blank image; the button triggers a download. Compositing the DOM panels onto the same image is a stretch (noted, not required).

## 4. Testing

- **Python (TDD / integration):**
  - `record_policy_episode` on a brain plus a head trained on the fixed goal produces a trace with `reached_goal=True` and a terminated frame (integration smoke; head trained briefly in-test or via a tiny deterministic head).
  - The zero-pad: a `recall=False` `out` missing the hippocampus recording builds a frame without error, and the hippocampus field is all zeros with the correct `[T, N]` shape.
  - `build_header(..., policy_regions=["sensory"])` includes `policy_regions` in the header.
  - `multi_region_viz` renders without error from a trace carrying `policy_regions` and the spectator tags appear in the figure (assert on the title strings via the existing test harness).
- **NEURO·SCOPE:** `tsc -b` clean, `npm run build` ok, Playwright asserts the spectator badge renders and the Export PNG button is present and clickable; visual correctness by manual sign-off. Keep `vi.mock("./hero/Hero")` in `App.test`.

## 5. Scope and sequence

Build order: 3.1 trace-path fix → 3.2 header field → 3.3 regenerate trace → 3.4 batch annotation → 3.5 NEURO·SCOPE annotation → NEURO·SCOPE export (last). Export DOM compositing is explicitly stretch. Engaging the brain is Track 2, out of scope here.

## 6. Acceptance criteria

1. `record_policy_episode` records the trained head policy under `recall=False` and the regenerated `week11_trained_trace.jsonl` contains a goal-reaching (terminated) frame.
2. A `recall=False` trace builds frames without crashing; bypassed regions render silent (zeros).
3. The trace header carries `policy_regions`, and both dashboards mark non-policy regions as spectators with no hardcoded region ids.
4. The batch dashboard renders an honest figure (spectator tags + caption) from the regenerated trace.
5. NEURO·SCOPE shows the spectator badge and an Export PNG button that downloads a non-blank hero image.
6. Python suite green; NEURO·SCOPE `tsc -b` clean, build ok, Playwright green.
