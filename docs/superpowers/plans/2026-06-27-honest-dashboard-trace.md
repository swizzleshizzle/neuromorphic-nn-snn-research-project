# Honest Dashboard + Trained-Policy Trace Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Record a trace of the actual trained policy (head-driven, `recall=False`) that reaches the goal, make both dashboards honestly mark spectator vs policy-path regions, and add PNG export, without touching the frozen-brain learning architecture.

**Architecture:** A new `record_policy_episode` drives actions from the trained `nn.Linear` head (not the brain's internal readout) while recording region activity, padding the bypassed hippocampus recording with zeros so frames build under `recall=False`. A new `policy_regions` header field is the single data-driven source both dashboards read to mark spectators. EXP-023 is repointed at the new recorder so its trained trace becomes goal-reaching; `sync-trace.mjs` then serves that trace to the web app.

**Tech Stack:** Python 3.11, PyTorch, Gymnasium, matplotlib, pytest; NEURO·SCOPE = Vite + React + TS + Zustand + react-three-fiber, Vitest, Playwright.

**Design:** `docs/superpowers/specs/2026-06-27-honest-dashboard-trace-design.md`

## Global Constraints

- **Brain stays frozen.** Do not change the learning architecture, `recall=False`, or what trains. This adds a recording path, a header field, dashboard annotations, and export only.
- **Backward compatibility:** `record_episode` is unchanged; `build_header`'s new `policy_regions` parameter defaults to `[]` and existing callers keep working; the spectator annotation is absent when `policy_regions` is missing/empty.
- **Data-driven:** spectator status derives from `policy_regions` only. No hardcoded region ids in dashboard logic.
- **Media track is dropped:** export is a plain technical feature (PNG download). No content/video framing in code, comments, or UI copy.
- **monitor must not import `neuromorphic.training`** (layering): the recorder computes the concept inline rather than importing the trainer.
- **WebGL/DOM components** are verified by `npx tsc -b` + `npm run build` + Playwright + manual sign-off, not jsdom unit tests. Keep `vi.mock("./hero/Hero")` in `App.test.tsx`.
- **Commands** run from the repo root. Python: `.venv/Scripts/python.exe -m pytest <paths> -v`. Web: from `dashboard/`, `npx tsc -b`, `npm run build`, `npx vitest run`, `npm run e2e`.
- **Commit style:** plain present-tense (`feat:`/`test:`), no `Co-Authored-By`, no "Generated with" trailer, no em-dashes.

---

## File Structure

```
src/neuromorphic/training/reinforce.py     # MODIFY: public concept_rate (alias kept)
src/neuromorphic/monitor/schema.py         # MODIFY: build_header gains policy_regions
src/neuromorphic/monitor/runner.py         # MODIFY: record_policy_episode + zero-pad helper
src/neuromorphic/monitor/__init__.py       # MODIFY: export record_policy_episode
experiments/023_week11_brain_training/run.py  # MODIFY: record via record_policy_episode
src/dashboard/multi_region_viz.py          # MODIFY: region_tag helper + spectator annotation
tests/monitor/test_schema.py               # MODIFY: policy_regions in header
tests/monitor/test_runner.py               # MODIFY: record_policy_episode behavior
tests/dashboard/test_multi_region_viz.py   # MODIFY: region_tag + spectator tags
dashboard/src/contract.ts                  # MODIFY: policy_regions?: string[]
dashboard/src/panels/RegionActivity.tsx    # MODIFY: spectator badge
dashboard/scripts/sync-trace.mjs           # MODIFY: source = trained trace
dashboard/src/hero/Hero.tsx                # MODIFY: preserveDrawingBuffer
dashboard/src/hero/overlays/ExportButton.tsx  # NEW: Export PNG control
dashboard/e2e/smoke.spec.ts                # MODIFY: spectator badge + export button
```

---

### Task 1: build_header carries policy_regions

**Files:**
- Modify: `src/neuromorphic/monitor/schema.py`
- Test: `tests/monitor/test_schema.py`

**Interfaces:**
- Produces: `build_header(brain, *, seed, action_labels, task_type="gridworld", grid_n=None, policy_regions=None) -> dict` with `header["policy_regions"] == list(policy_regions or [])`.

- [ ] **Step 1: Write the failing test**

Add to `tests/monitor/test_schema.py`:

```python
def test_build_header_includes_policy_regions():
    from neuromorphic.brain import Brain
    from neuromorphic.monitor.schema import build_header

    brain = Brain(grid_n=5, seed=0)
    h = build_header(brain, seed=0, action_labels=["up", "right", "down", "left"], policy_regions=["sensory"])
    assert h["policy_regions"] == ["sensory"]
    # default is an empty list (backward compatible)
    h2 = build_header(brain, seed=0, action_labels=["up", "right", "down", "left"])
    assert h2["policy_regions"] == []
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/Scripts/python.exe -m pytest tests/monitor/test_schema.py::test_build_header_includes_policy_regions -v`
Expected: FAIL (`policy_regions` kwarg / key missing).

- [ ] **Step 3: Implement**

In `src/neuromorphic/monitor/schema.py`, change `build_header` to accept and emit `policy_regions`:

```python
def build_header(brain, *, seed: int, action_labels, task_type: str = "gridworld", grid_n: int | None = None, policy_regions=None) -> dict:
    """Build the once-per-run trace header declaring brain topology + run context."""
    grid_n = brain.grid_n if grid_n is None else grid_n
    regions = [
        {"id": rid, "label": label, "n_neurons": n, "role": role, "render": render_for_n(n)}
        for rid, label, n, role in region_specs(brain)
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "brain": {"id": "five-region", "config_hash": _config_hash(brain, seed), "seed": seed, "T": brain.T},
        "task": {"type": task_type, "grid_n": grid_n, "action_labels": list(action_labels)},
        "regions": regions,
        "pathways": [dict(p) for p in PATHWAYS],
        "policy_regions": list(policy_regions or []),
    }
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/Scripts/python.exe -m pytest tests/monitor/test_schema.py -v`
Expected: PASS (existing schema tests plus the new one).

- [ ] **Step 5: Commit**

```bash
git add src/neuromorphic/monitor/schema.py tests/monitor/test_schema.py
git commit -m "feat: trace header carries policy_regions"
```

---

### Task 2: record_policy_episode (head-driven, recall=False, zero-padded)

**Files:**
- Modify: `src/neuromorphic/training/reinforce.py`, `src/neuromorphic/monitor/runner.py`, `src/neuromorphic/monitor/__init__.py`
- Test: `tests/monitor/test_runner.py`

**Interfaces:**
- Consumes: `build_header` with `policy_regions` (Task 1); `build_frame`; `REGION_OUTPUT_KEY`, `region_specs` from schema.
- Produces:
  - `concept_rate(out) -> torch.Tensor` public in `reinforce.py` (`out["concept"].mean(dim=0)[0]`), with `_concept_rate = concept_rate` kept as an alias.
  - `record_policy_episode(brain, head, env, sink, *, seed=0, action_labels=DEFAULT_ACTION_LABELS, max_steps=None, recall=False, policy_regions=("sensory",), generator=None) -> dict` returning `{"steps", "total_reward", "reached_goal"}`. Actions come from `head` on the sensory concept; bypassed region recordings are zero-padded before `build_frame`.

- [ ] **Step 1: Write the failing test**

Add to `tests/monitor/test_runner.py`:

```python
def test_record_policy_episode_uses_head_and_pads_bypassed(tmp_path):
    import torch
    from neuromorphic.brain import Brain
    from neuromorphic.envs import GridWorldEnv
    from neuromorphic.monitor import FileSink, record_policy_episode
    from neuromorphic.monitor.frame import build_frame  # noqa: F401  (sanity import)
    from neuromorphic.training.reinforce import make_policy_head

    brain = Brain(grid_n=5, seed=0)
    head = make_policy_head(brain)
    # Force the head to always choose action 1 (right), regardless of input.
    with torch.no_grad():
        head.weight.zero_()
        head.bias.copy_(torch.tensor([0.0, 10.0, 0.0, 0.0]))

    env = GridWorldEnv(max_steps=6)
    sink = FileSink(tmp_path / "trace.jsonl")
    gen = torch.Generator().manual_seed(0)
    summary = record_policy_episode(brain, head, env, sink, seed=0, recall=False,
                                    policy_regions=("sensory",), generator=gen)

    assert summary["steps"] >= 1
    lines = [l for l in (tmp_path / "trace.jsonl").read_text().splitlines() if l.strip()]
    import json
    header = json.loads(lines[0]); frames = [json.loads(l) for l in lines[1:]]
    assert header["policy_regions"] == ["sensory"]
    # every action is the head's forced choice, not the brain's internal readout
    assert all(f["task"]["action"] == 1 for f in frames)
    # bypassed hippocampus recording is padded to a silent field of correct width
    n_hippo = brain.hippo.n_neurons
    spikes = frames[0]["field"]["hippocampus"]["spikes"]
    assert len(spikes[0]) == n_hippo
    assert all(v == 0 for row in spikes for v in row)
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/Scripts/python.exe -m pytest tests/monitor/test_runner.py::test_record_policy_episode_uses_head_and_pads_bypassed -v`
Expected: FAIL (`record_policy_episode` not importable).

- [ ] **Step 3: Implement**

In `src/neuromorphic/training/reinforce.py`, promote the concept helper (keep the private alias so `action_distribution` still works):

```python
def concept_rate(out: dict) -> torch.Tensor:
    """Mean firing rate of the sensory concept over the window, single agent. -> [concept]."""
    return out["concept"].mean(dim=0)[0]


# Backward-compatible alias (older callers import the private name).
_concept_rate = concept_rate
```

(Delete the old `def _concept_rate(...)` definition; `action_distribution` keeps calling `_concept_rate(out)`, now the alias.)

In `src/neuromorphic/monitor/runner.py`, add the zero-pad helper and the recorder (the recorder computes the concept inline so monitor does not import training):

```python
import torch

from neuromorphic.monitor.frame import build_frame
from neuromorphic.monitor.schema import REGION_OUTPUT_KEY, build_header, region_specs

# ... keep DEFAULT_ACTION_LABELS and record_episode unchanged ...


def _pad_bypassed_recordings(out: dict, brain) -> None:
    """Ensure every region has an output recording; zero-fill ones bypassed this step."""
    recs = out.setdefault("recordings", {})
    t_b = None
    for rid, key in REGION_OUTPUT_KEY.items():
        r = recs.get(rid)
        if r is not None and key in r:
            tens = r[key]
            t_b = (tens.shape[0], tens.shape[1])
            break
    if t_b is None:
        return
    T, B = t_b
    sizes = {rid: n for rid, _, n, _ in region_specs(brain)}
    for rid, key in REGION_OUTPUT_KEY.items():
        r = recs.get(rid)
        if r is None or key not in r:
            recs.setdefault(rid, {})[key] = torch.zeros(T, B, sizes[rid])


def record_policy_episode(
    brain,
    head,
    env,
    sink,
    *,
    seed: int = 0,
    action_labels=DEFAULT_ACTION_LABELS,
    max_steps: int | None = None,
    recall: bool = False,
    policy_regions=("sensory",),
    generator=None,
) -> dict:
    """Record one episode driven by the trained head (sensory concept -> action logits).

    Unlike ``record_episode`` (which uses the brain's internal ``out['action']``), this
    records the actual trained policy. Region activity is still captured via ``record=True``;
    regions bypassed under ``recall=False`` are zero-filled so frames build without error.
    """
    sink.open(build_header(brain, seed=seed, action_labels=action_labels, policy_regions=list(policy_regions)))
    obs, _ = env.reset(seed=seed)

    total_reward = 0.0
    reached_goal = False
    steps = 0
    limit = max_steps if max_steps is not None else getattr(env, "max_steps", 100)

    while steps < limit:
        out = brain.step(obs, store=False, recall=recall, record=True, generator=generator)
        concept = out["concept"].mean(dim=0)[0]  # inline (monitor must not import training)
        action = int(head(concept).argmax())
        _pad_bypassed_recordings(out, brain)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += float(reward)

        task = {
            "agent": [int(obs[0]), int(obs[1])],
            "goal": [int(obs[2]), int(obs[3])],
            "action": action,
            "action_label": action_labels[action],
            "reward": float(reward),
            "return": total_reward,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
        }
        frame = build_frame(out, episode=0, step=steps, t=float(steps), task=task,
                            store=False, recall=recall, grid_n=brain.grid_n)
        sink.write(frame)

        obs = next_obs
        steps += 1
        if terminated:
            reached_goal = True
            break
        if truncated:
            break

    sink.close()
    return {"steps": steps, "total_reward": total_reward, "reached_goal": reached_goal}
```

In `src/neuromorphic/monitor/__init__.py`, export the new function:

```python
from neuromorphic.monitor.runner import record_episode, record_policy_episode
```

and add `"record_policy_episode"` to `__all__`.

- [ ] **Step 4: Run to verify pass**

Run: `.venv/Scripts/python.exe -m pytest tests/monitor/test_runner.py -v`
Expected: PASS (existing runner tests plus the new one).

- [ ] **Step 5: Commit**

```bash
git add src/neuromorphic/training/reinforce.py src/neuromorphic/monitor/runner.py src/neuromorphic/monitor/__init__.py tests/monitor/test_runner.py
git commit -m "feat: record_policy_episode records the trained head policy"
```

---

### Task 3: Repoint EXP-023 and regenerate the goal-reaching trace

**Files:**
- Modify: `experiments/023_week11_brain_training/run.py`

**Interfaces:**
- Consumes: `record_policy_episode` (Task 2).

- [ ] **Step 1: Switch the recording call**

In `experiments/023_week11_brain_training/run.py`, update the import and the post-training recording. Change the import line:

```python
from neuromorphic.monitor import FileSink, record_policy_episode  # was: record_episode
```

Replace the recording block (the `record_episode(...)` call near the end) with:

```python
    # Record one trained episode for the dashboard, driven by the TRAINED HEAD
    # (recall=False, the actual policy). Bypassed regions render silent; only the
    # sensory region is on the policy path in v1.
    sink = FileSink(TRACE)
    summary = record_policy_episode(
        brain, head, env, sink, seed=args.seed, recall=False,
        policy_regions=("sensory",), generator=gen,
    )
    print(f"trained trace  -> {TRACE} (reached_goal={summary['reached_goal']})", flush=True)
```

- [ ] **Step 2: Regenerate the trace and verify it reaches the goal**

Run: `.venv/Scripts/python.exe experiments/023_week11_brain_training/run.py --episodes 150`
Expected: prints `trained: ... reached goal 100%` during eval and `reached_goal=True` for the recorded trace.

Verify the written trace has a terminated frame:

```bash
.venv/Scripts/python.exe -c "import json; ls=[l for l in open('outputs/week11_trained_trace.jsonl') if l.strip()]; fr=[json.loads(l) for l in ls[1:]]; print('frames', len(fr), 'terminated', sum(f['task']['terminated'] for f in fr), 'policy_regions', json.loads(ls[0])['policy_regions'])"
```
Expected: `terminated` >= 1 and `policy_regions ['sensory']`.

- [ ] **Step 3: Commit (code + regenerated trace)**

```bash
git add experiments/023_week11_brain_training/run.py outputs/week11_trained_trace.jsonl
git commit -m "feat: EXP-023 records the trained head policy as a goal-reaching trace"
```

---

### Task 4: Batch dashboard spectator annotation

**Files:**
- Modify: `src/dashboard/multi_region_viz.py`
- Test: `tests/dashboard/test_multi_region_viz.py`

**Interfaces:**
- Produces: `region_tag(region_id, policy_regions) -> str` returning `"● on policy path"` when the id is in `policy_regions`, else `"○ spectator (frozen)"`. Used to annotate each region raster.

- [ ] **Step 1: Write the failing test**

Add to `tests/dashboard/test_multi_region_viz.py`:

```python
def test_region_tag_marks_policy_vs_spectator():
    from dashboard.multi_region_viz import region_tag
    assert region_tag("sensory", ["sensory"]) == "● on policy path"
    assert region_tag("motor", ["sensory"]) == "○ spectator (frozen)"
    # empty/missing policy_regions -> no claim, treat as spectator-unknown (spectator string)
    assert region_tag("sensory", []) == "○ spectator (frozen)"
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/Scripts/python.exe -m pytest tests/dashboard/test_multi_region_viz.py::test_region_tag_marks_policy_vs_spectator -v`
Expected: FAIL (`region_tag` not defined).

- [ ] **Step 3: Implement**

In `src/dashboard/multi_region_viz.py`, add the helper near the styling helpers:

```python
def region_tag(region_id, policy_regions) -> str:
    """Honest label: is this region on the policy path, or a frozen spectator?"""
    return "● on policy path" if region_id in (policy_regions or []) else "○ spectator (frozen)"
```

Thread `policy_regions` into the raster annotation. Change `_raster` to accept a tag and append it to the ylabel:

```python
def _raster(ax, spikes, region, label, tag=""):
    """Spike raster for one region: x = time, y = neuron, tinted by region."""
    arr = np.asarray(spikes)
    color = PALETTE.get(region, TEXT)
    if arr.size:
        T, N = arr.shape
        ts, ns = np.nonzero(arr)
        ax.scatter(ts, ns, s=6, c=color, marker="s", linewidths=0)
        ax.set_xlim(-0.5, T - 0.5)
        ax.set_ylim(-0.5, max(N - 0.5, 0.5))
    _style_ax(ax, accent=color)
    suffix = f"\n{tag}" if tag else ""
    ax.set_ylabel(f"{label}\n(n={arr.shape[1] if arr.size else 0}){suffix}",
                  color=color, fontsize=8, rotation=0, ha="right", va="center")
    ax.yaxis.set_label_coords(-0.04, 0.5)
```

In `render_dashboard`, read `policy_regions` and pass the tag, and add a caption:

```python
    policy_regions = header.get("policy_regions", [])
    ...
    for i, rid in enumerate(order):
        ax = fig.add_subplot(left[i])
        _raster(ax, frame["field"][rid]["spikes"], rid, labels[rid], tag=region_tag(rid, policy_regions))
        ...
```

After the existing `fig.suptitle(...)`, add an honest caption (only when policy_regions is known):

```python
    if policy_regions:
        fig.text(0.10, 0.005,
                 "v1: only policy-path regions learn; spectators are frozen (run the brain to engage them).",
                 color=MUTED, fontsize=8, ha="left")
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/Scripts/python.exe -m pytest tests/dashboard/test_multi_region_viz.py -v`
Expected: PASS (existing dashboard tests plus the new one).

- [ ] **Step 5: Regenerate the batch figure from the goal-reaching trace**

Run: `.venv/Scripts/python.exe -m dashboard.multi_region_viz outputs/week11_trained_trace.jsonl`
Expected: writes `outputs/week11_trained_trace_dashboard.png` with the spectator tags and caption.

- [ ] **Step 6: Commit (code + figure)**

```bash
git add src/dashboard/multi_region_viz.py tests/dashboard/test_multi_region_viz.py outputs/week11_trained_trace_dashboard.png
git commit -m "feat: batch dashboard marks spectator vs policy-path regions"
```

---

### Task 5: NEURO·SCOPE spectator badge + serve the honest trace

**Files:**
- Modify: `dashboard/src/contract.ts`, `dashboard/src/panels/RegionActivity.tsx`, `dashboard/scripts/sync-trace.mjs`, `dashboard/e2e/smoke.spec.ts`

**Interfaces:**
- Consumes: the `policy_regions` header field (Task 1); the regenerated trained trace (Task 3).
- Produces: `TraceHeader.policy_regions?: string[]`; a `[data-spectator]` badge on regions not in `policy_regions`; the web app serves the goal-reaching trained trace.

- [ ] **Step 1: Extend the contract type**

In `dashboard/src/contract.ts`, add to `TraceHeader`:

```ts
export interface TraceHeader {
  schema_version: string;
  brain: { id: string; config_hash: string; seed: number; T: number };
  task: { type: string; grid_n: number; action_labels: string[] };
  regions: Region[];
  pathways: Pathway[];
  policy_regions?: string[];
}
```

- [ ] **Step 2: Spectator badge in RegionActivity**

In `dashboard/src/panels/RegionActivity.tsx`, compute the spectator set from the header and render a badge. Replace the label line:

```tsx
export function RegionActivity() {
  const header = useTraceStore((s) => s.header);
  const frame = useTraceStore((s) => s.frames[s.envStep]);
  if (!header) return null;
  const policy = header.policy_regions;

  return (
    <Panel kicker="PANEL 01 · ACTIVITY" title="Region Activity">
      {header.regions.map((r) => {
        const rs = frame?.regions[r.id];
        const rate = rs?.rate ?? 0;
        const hue = regionHue(r.id);
        const spectator = policy !== undefined && !policy.includes(r.id);
        return (
          <div key={r.id} style={{ display: "flex", alignItems: "center", gap: 8, padding: "7px 0" }}>
            <span style={{ width: 7, height: 7, borderRadius: "50%", background: hue, boxShadow: `0 0 6px ${hue}`, flex: "none" }} />
            <div style={{ flex: 1, minWidth: 0 }}>
              <div style={{ display: "flex", justifyContent: "space-between", font: "12px sans-serif" }}>
                <span>
                  {r.label}
                  {spectator && (
                    <span
                      data-spectator
                      style={{ marginLeft: 6, font: "8px monospace", color: "var(--text-faint)", border: "1px solid var(--edge)", borderRadius: 3, padding: "1px 4px" }}
                    >
                      spectator
                    </span>
                  )}
                </span>
                <span style={{ font: "11px monospace", color: "var(--text-dim)" }}>{rate.toFixed(2)}</span>
              </div>
              <svg width="100%" height="16" viewBox="0 0 100 16" preserveAspectRatio="none" style={{ marginTop: 3, display: "block" }}>
                <polyline points={sparkPoints(rs?.rate_t ?? [], 100, 16)} fill="none" stroke={hue} strokeWidth="1.2" />
              </svg>
              <div style={{ font: "9px monospace", color: "var(--text-faint)", marginTop: 2 }}>
                active {((rs?.active_frac ?? 0) * 100).toFixed(0)}% · {rs?.spikes ?? 0} spikes
              </div>
            </div>
          </div>
        );
      })}
    </Panel>
  );
}
```

(Keep the existing `sparkPoints` helper above unchanged.)

- [ ] **Step 3: Serve the goal-reaching trace**

In `dashboard/scripts/sync-trace.mjs`, change the source to the trained (goal-reaching, policy_regions-carrying) trace, keeping the public filename so no frontend URL changes:

```js
const src = resolve(here, "../../outputs/week11_trained_trace.jsonl");
const dst = resolve(here, "../public/week11_dashboard_trace.jsonl");
```

Run: `cd dashboard && npm run sync:trace`
Expected: `synced trace -> .../public/week11_dashboard_trace.jsonl`.

- [ ] **Step 4: e2e asserts the badge**

In `dashboard/e2e/smoke.spec.ts`, add inside the existing test (after the panel assertions):

```ts
  // spectator badges render for frozen regions (trace carries policy_regions)
  await expect(page.locator("[data-spectator]").first()).toBeVisible();
```

- [ ] **Step 5: Verify**

From `dashboard/`:
Run: `npx tsc -b` → clean.
Run: `npm run build` → succeeds.
Run: `npx vitest run` → green, App.test mock intact.
Run: `npm run e2e` → PASS (badge visible on the served trace).

- [ ] **Step 6: Commit**

```bash
git add dashboard/src/contract.ts dashboard/src/panels/RegionActivity.tsx dashboard/scripts/sync-trace.mjs dashboard/public/week11_dashboard_trace.jsonl dashboard/e2e/smoke.spec.ts
git commit -m "feat: NEURO-SCOPE spectator badge and honest goal-reaching trace"
```

---

### Task 6: NEURO·SCOPE Export PNG

**Files:**
- Modify: `dashboard/src/hero/Hero.tsx`
- Create: `dashboard/src/hero/overlays/ExportButton.tsx`
- Modify: `dashboard/e2e/smoke.spec.ts`

**Interfaces:**
- Produces: an `Export PNG` control (`[data-export-png]`) that downloads the hero canvas as a PNG. The hero `<Canvas>` is created with `gl={{ preserveDrawingBuffer: true }}` so `toDataURL` yields a non-blank image.

- [ ] **Step 1: Enable canvas readback**

In `dashboard/src/hero/Hero.tsx`, add `gl={{ preserveDrawingBuffer: true }}` to the `<Canvas>`:

```tsx
      <Canvas
        camera={{ position: [0, 0, 3.2], fov: 50 }}
        style={{ background: "var(--bg)" }}
        dpr={[1, 2]}
        gl={{ preserveDrawingBuffer: true }}
      >
        <Scene morphRef={morphRef} />
      </Canvas>
```

- [ ] **Step 2: Export control**

Create `dashboard/src/hero/overlays/ExportButton.tsx`:

```tsx
export function ExportButton() {
  const onExport = () => {
    const canvas = document.querySelector("canvas");
    if (!canvas) return;
    const url = canvas.toDataURL("image/png");
    const a = document.createElement("a");
    a.href = url;
    a.download = "neuroscope-hero.png";
    a.click();
  };
  return (
    <button
      data-export-png
      onClick={onExport}
      style={{
        position: "absolute",
        top: 14,
        right: 120,
        zIndex: 10,
        font: "600 10px/1 'Space Grotesk', sans-serif",
        color: "var(--text-dim)",
        background: "var(--panel)",
        border: "1px solid var(--edge)",
        borderRadius: 8,
        backdropFilter: "var(--blur)",
        padding: "6px 11px",
        cursor: "pointer",
      }}
    >
      Export PNG
    </button>
  );
}
```

Mount it in `Hero.tsx` as a sibling of `<Canvas>` (alongside the existing overlays):

```tsx
      <CloudFlowToggle />
      <ExportButton />
      <SensoryGrid />
      <HeroCaption />
```

(Add `import { ExportButton } from "./overlays/ExportButton";` at the top of `Hero.tsx`.)

- [ ] **Step 3: e2e asserts the button**

In `dashboard/e2e/smoke.spec.ts`, add inside the existing test:

```ts
  // export control is present and clickable
  const exportBtn = page.locator("[data-export-png]");
  await expect(exportBtn).toBeVisible();
  await exportBtn.click();
```

- [ ] **Step 4: Verify**

From `dashboard/`:
Run: `npx tsc -b` → clean.
Run: `npm run build` → succeeds.
Run: `npx vitest run` → green, App.test mock intact.
Run: `npm run e2e` → PASS (export button visible + click does not error).

Manual: `npm run dev`, click Export PNG, confirm a non-blank `neuroscope-hero.png` downloads.

- [ ] **Step 5: Commit**

```bash
git add dashboard/src/hero/Hero.tsx dashboard/src/hero/overlays/ExportButton.tsx dashboard/e2e/smoke.spec.ts
git commit -m "feat: NEURO-SCOPE hero PNG export"
```

---

## Self-Review

**Spec coverage:**
- Trace-path fix (head-driven recall=False + zero-pad): Tasks 1, 2. ✓
- Honest annotation mechanism (policy_regions header field): Task 1; consumed in Tasks 4, 5. ✓
- Regenerate goal-reaching trace: Task 3. ✓
- Batch dashboard honest annotation: Task 4. ✓
- NEURO·SCOPE annotation: Task 5. ✓
- NEURO·SCOPE PNG export: Task 6. ✓
- Media-as-technical-only, brain-frozen, data-driven, layering (no monitor→training import): Global Constraints + Task 2 inline-concept note. ✓
- Export DOM compositing: correctly left out (stretch, not a task). ✓

**Placeholder scan:** none. Every code step shows complete code.

**Type/name consistency:** `policy_regions` is named identically across `build_header` (Task 1), `record_policy_episode` (Task 2), the batch dashboard (Task 4), and the contract/`RegionActivity` (Task 5). `concept_rate` (Task 2) preserves the `_concept_rate` alias that `action_distribution` calls. `region_tag` (Task 4) signature matches its test. The `[data-spectator]` and `[data-export-png]` hooks match their e2e selectors.

**Shared-code touches:** `build_header` gains an optional param (default `[]`, existing callers unaffected); `record_episode` is untouched; `reinforce` keeps the `_concept_rate` alias. No regressions expected.
