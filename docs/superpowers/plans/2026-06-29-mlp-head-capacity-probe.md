# EXP-025 Head Capacity Probe Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Determine whether the ~30-50% held-out navigation cap is caused by the linear policy head or the frozen sensory encoder, by swapping a one-hidden-layer MLP head in for the linear head and comparing held-out success across a paired multi-seed sweep.

**Architecture:** Make the policy head pluggable in `reinforce.py` (linear default stays byte-identical; add an MLP option). Thread a `head_type` option through the existing EXP-024 generalization harness. Add an EXP-025 runner that sweeps `{linear, mlp} x {shaped, sparse} x 5 seeds` and an aggregator that emits a paired evidence table. The brain stays frozen throughout; only the head changes.

**Tech Stack:** Python, PyTorch (`torch.nn`), pytest. snntorch-based `Brain` (frozen feature extractor). REINFORCE trainer in `neuromorphic.training`.

## Global Constraints

- Commit messages: plain, imperative, no `Co-Authored-By` / AI trailers, no em-dashes in prose or docs.
- Run Python via the repo venv: `.venv/Scripts/python.exe` (Windows).
- The `head_type="linear"` path must stay byte-identical to today's behavior (preserves EXP-023/EXP-024 as the baseline of record). It must remain an `nn.Linear` instance so existing tests reading `head.in_features`, `head.out_features`, and `head.weight` keep passing.
- The brain is frozen in v1 (`recall=False`, runs under `no_grad`); no encoder changes in this work.
- Concept/action dims come from the brain (`brain.content`, `brain.n_actions`), never hardcoded.
- Keep the existing 202-test suite green.

---

### Task 1: Pluggable policy head (linear default + MLP option)

**Files:**
- Modify: `src/neuromorphic/training/reinforce.py` (`make_policy_head`, type hints on `action_distribution`, `greedy_action`, `policy_parameters`, `train_episode`)
- Test: `tests/training/test_reinforce.py`

**Interfaces:**
- Consumes: `Brain` with `.content` (int, 64) and `.n_actions` (int, 4).
- Produces: `make_policy_head(brain, head_type: str = "linear", hidden: int = 128) -> nn.Module`.
  - `head_type="linear"` -> `nn.Linear(brain.content, brain.n_actions)` (unchanged default).
  - `head_type="mlp"` -> `nn.Sequential(nn.Linear(brain.content, hidden), nn.ReLU(), nn.Linear(hidden, brain.n_actions))`.
  - Any unknown `head_type` raises `ValueError`.
  - All downstream functions accept any `nn.Module` head (only `head(concept_rate)` and `head.parameters()` are used).

- [ ] **Step 1: Write the failing tests**

Add to `tests/training/test_reinforce.py`:

```python
import pytest
import torch.nn as nn


def test_make_policy_head_linear_is_unchanged_default():
    brain = Brain(grid_n=5, seed=0)
    head = make_policy_head(brain)
    assert isinstance(head, nn.Linear)
    assert head.in_features == brain.content
    assert head.out_features == brain.n_actions


def test_make_policy_head_mlp_shape_and_forward():
    brain = Brain(grid_n=5, seed=0)
    head = make_policy_head(brain, head_type="mlp", hidden=128)
    assert isinstance(head, nn.Sequential)
    x = torch.zeros(brain.content)
    out = head(x)
    assert out.shape == (brain.n_actions,)
    assert len(list(head.parameters())) > 0


def test_make_policy_head_rejects_unknown_type():
    brain = Brain(grid_n=5, seed=0)
    with pytest.raises(ValueError):
        make_policy_head(brain, head_type="transformer")


def test_train_episode_updates_mlp_head_but_not_the_frozen_brain():
    brain = Brain(grid_n=5, seed=0)
    head = make_policy_head(brain, head_type="mlp", hidden=128)
    env = GridWorldEnv()
    opt = torch.optim.Adam(policy_parameters(head), lr=1e-2)
    params_before = [p.detach().clone() for p in head.parameters()]
    sensory_before = brain.sensory.fc1.weight.detach().clone()

    train_episode(
        brain, head, env, opt, gamma=0.99, baseline=0.0,
        generator=torch.Generator().manual_seed(0), max_steps=10,
    )

    changed = any(
        not torch.equal(b, a.detach())
        for b, a in zip(params_before, head.parameters())
    )
    assert changed
    assert torch.equal(sensory_before, brain.sensory.fc1.weight.detach())
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/training/test_reinforce.py -k "make_policy_head or mlp_head" -v`
Expected: FAIL (TypeError: unexpected keyword argument `head_type`, and `nn`/`pytest` usage).

- [ ] **Step 3: Implement the pluggable head**

In `src/neuromorphic/training/reinforce.py`, replace `make_policy_head` and widen the head type hints from `nn.Linear` to `nn.Module`:

```python
def make_policy_head(brain, head_type: str = "linear", hidden: int = 128) -> nn.Module:
    """A trainable actor head: sensory concept (``brain.content`` dims) -> action logits.

    ``head_type="linear"`` is the v1 default (a single ``nn.Linear``); ``"mlp"`` adds one
    ReLU hidden layer of width ``hidden`` to test whether a nonlinear readout extracts more
    from the frozen sensory concept (EXP-025). The brain stays frozen either way.
    """
    if head_type == "linear":
        return nn.Linear(brain.content, brain.n_actions)
    if head_type == "mlp":
        return nn.Sequential(
            nn.Linear(brain.content, hidden),
            nn.ReLU(),
            nn.Linear(hidden, brain.n_actions),
        )
    raise ValueError(f"unknown head_type {head_type!r} (expected 'linear' or 'mlp')")
```

Change the head parameter annotations on `action_distribution`, `greedy_action`, `policy_parameters`, and `train_episode` from `head: nn.Linear` to `head: nn.Module`. Update `policy_parameters` docstring to read "the head only" (already correct) and keep its body `return head.parameters()`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/training/test_reinforce.py -v`
Expected: PASS (new tests plus all pre-existing reinforce tests, including `test_policy_head_shapes` and `test_train_episode_updates_head_but_not_the_frozen_brain`).

- [ ] **Step 5: Commit**

```bash
git add src/neuromorphic/training/reinforce.py tests/training/test_reinforce.py
git commit -m "feat: pluggable policy head with optional MLP (EXP-025)"
```

---

### Task 2: Thread head_type through the generalization harness

**Files:**
- Modify: `src/neuromorphic/training/generalization.py` (`GenConfig`, `run_generalization`)
- Test: `tests/training/test_generalization.py`

**Interfaces:**
- Consumes: `make_policy_head(brain, head_type, hidden)` from Task 1.
- Produces: `GenConfig` gains `head_type: str = "linear"` and `hidden: int = 128`; `run_generalization(cfg)` builds the head via `make_policy_head(brain, head_type=cfg.head_type, hidden=cfg.hidden)`. The returned summary's `config` block includes `head_type` and `hidden` (already true via `asdict(cfg)`).

- [ ] **Step 1: Write the failing tests**

Add to `tests/training/test_generalization.py` (match the existing imports there; add any missing ones):

```python
from neuromorphic.training.generalization import GenConfig, run_generalization


def test_genconfig_defaults_to_linear_head():
    cfg = GenConfig()
    assert cfg.head_type == "linear"
    assert cfg.hidden == 128


def test_run_generalization_mlp_head_records_head_type(tmp_path):
    cfg = GenConfig(
        seed=0, episodes=3, n_heldout=2, max_steps=8,
        head_type="mlp", hidden=32, tag="smoke_mlp", out_dir=tmp_path,
    )
    summary = run_generalization(cfg)
    assert summary["config"]["head_type"] == "mlp"
    assert summary["config"]["hidden"] == 32
    assert "train" in summary["eval"] and "heldout" in summary["eval"]


def test_run_generalization_is_deterministic_for_fixed_seed_and_head(tmp_path):
    base = dict(seed=1, episodes=3, n_heldout=2, max_steps=8, head_type="linear")
    a = run_generalization(GenConfig(**base, tag="det_a", out_dir=tmp_path))
    b = run_generalization(GenConfig(**base, tag="det_b", out_dir=tmp_path))
    assert a["eval"] == b["eval"]
    assert a["train_goals"] == b["train_goals"]
    assert a["heldout_goals"] == b["heldout_goals"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/training/test_generalization.py -k "head_type or mlp or deterministic" -v`
Expected: FAIL (`GenConfig` has no `head_type` field -> TypeError on construction / AttributeError).

- [ ] **Step 3: Implement the pass-through**

In `src/neuromorphic/training/generalization.py`:

Add two fields to `GenConfig` (place them next to the other model-ish fields, before `tag`):

```python
    head_type: str = "linear"
    hidden: int = 128
```

In `run_generalization`, change the head construction line:

```python
    head = make_policy_head(brain, head_type=cfg.head_type, hidden=cfg.hidden)
```

(`make_policy_head` is already imported from `neuromorphic.training.reinforce`.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/training/test_generalization.py -v`
Expected: PASS (new tests plus all pre-existing generalization tests).

- [ ] **Step 5: Commit**

```bash
git add src/neuromorphic/training/generalization.py tests/training/test_generalization.py
git commit -m "feat: head_type option in generalization harness (EXP-025)"
```

---

### Task 3: EXP-025 sweep runner and aggregator

**Files:**
- Create: `experiments/025_head_capacity/run.py`
- Create: `experiments/025_head_capacity/aggregate.py`
- Test: `tests/training/test_head_capacity_aggregate.py`

**Interfaces:**
- Consumes: `GenConfig`, `run_generalization` (Task 2).
- Produces:
  - `experiments/025_head_capacity/run.py` with `build_configs(seeds, episodes, out_dir) -> list[GenConfig]` (the 2 heads x 2 regimes x len(seeds) sweep) and a `main()` that runs them, writing per-run files tagged `f"{regime}_{head_type}_seed{seed}"` into `out_dir`.
  - `experiments/025_head_capacity/aggregate.py` with `aggregate(summaries: list[dict]) -> dict` and `format_table(agg: dict) -> str`. `aggregate` groups by `(head_type, regime)` and computes `mean`/`spread` (max-min) of held-out and train success across seeds; `format_table` renders a markdown table.

- [ ] **Step 1: Write the failing test for the aggregator**

Create `tests/training/test_head_capacity_aggregate.py`:

```python
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "exp025_aggregate", ROOT / "experiments" / "025_head_capacity" / "aggregate.py"
)
agg_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agg_mod)


def _summary(head_type, shaping, heldout_success, train_success):
    return {
        "config": {"head_type": head_type, "shaping": shaping},
        "eval": {
            "train": {"success_rate": train_success},
            "heldout": {"success_rate": heldout_success},
        },
    }


def test_aggregate_groups_by_head_and_regime():
    summaries = [
        _summary("linear", True, 0.4, 0.5),
        _summary("linear", True, 0.6, 0.7),
        _summary("mlp", True, 0.8, 0.9),
        _summary("mlp", True, 0.9, 1.0),
    ]
    agg = agg_mod.aggregate(summaries)
    assert agg[("linear", "shaped")]["heldout_mean"] == 0.5
    assert agg[("linear", "shaped")]["heldout_spread"] == 0.2
    assert agg[("mlp", "shaped")]["heldout_mean"] == 0.85


def test_format_table_is_markdown_with_both_heads():
    summaries = [
        _summary("linear", False, 0.3, 0.4),
        _summary("mlp", False, 0.5, 0.6),
    ]
    table = agg_mod.format_table(agg_mod.aggregate(summaries))
    assert "linear" in table and "mlp" in table
    assert "sparse" in table
    assert table.count("|") >= 6  # at least a header row of cells
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/training/test_head_capacity_aggregate.py -v`
Expected: FAIL (`aggregate.py` does not exist -> import error).

- [ ] **Step 3: Implement the aggregator**

Create `experiments/025_head_capacity/aggregate.py`:

```python
"""Aggregate EXP-025 per-run summaries into a paired head-vs-regime evidence table."""

from __future__ import annotations


def _regime(shaping: bool) -> str:
    return "shaped" if shaping else "sparse"


def aggregate(summaries: list[dict]) -> dict:
    """Group summaries by (head_type, regime); mean and spread (max-min) of success."""
    groups: dict[tuple[str, str], dict[str, list[float]]] = {}
    for s in summaries:
        key = (s["config"]["head_type"], _regime(s["config"]["shaping"]))
        g = groups.setdefault(key, {"heldout": [], "train": []})
        g["heldout"].append(s["eval"]["heldout"]["success_rate"])
        g["train"].append(s["eval"]["train"]["success_rate"])

    out: dict[tuple[str, str], dict[str, float]] = {}
    for key, vals in groups.items():
        h, t = vals["heldout"], vals["train"]
        out[key] = {
            "n": len(h),
            "heldout_mean": sum(h) / len(h),
            "heldout_spread": max(h) - min(h),
            "train_mean": sum(t) / len(t),
            "train_spread": max(t) - min(t),
        }
    return out


def format_table(agg: dict) -> str:
    """Render the aggregate as a markdown table sorted by (regime, head_type)."""
    lines = [
        "| regime | head | n | heldout mean | heldout spread | train mean |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for (head, regime) in sorted(agg, key=lambda k: (k[1], k[0])):
        m = agg[(head, regime)]
        lines.append(
            f"| {regime} | {head} | {m['n']} | {m['heldout_mean']:.0%} | "
            f"{m['heldout_spread']:.0%} | {m['train_mean']:.0%} |"
        )
    return "\n".join(lines)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/training/test_head_capacity_aggregate.py -v`
Expected: PASS.

- [ ] **Step 5: Implement the sweep runner**

Create `experiments/025_head_capacity/run.py`:

```python
"""EXP-025 - head capacity probe (MLP head vs linear head).

Sweeps {linear, mlp} x {shaped, sparse} x seeds, paired by seed (identical goal split
per seed), and aggregates held-out success into a markdown evidence table. The brain stays
frozen (ADR-0001); only the policy head changes.

Run (repo root, venv active):
    .venv/Scripts/python.exe experiments/025_head_capacity/run.py
    .venv/Scripts/python.exe experiments/025_head_capacity/run.py --seeds 0 1 2 3 4 --episodes 600
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import torch  # noqa: E402

from neuromorphic.training.generalization import GenConfig, run_generalization

torch.set_num_threads(1)

HERE = Path(__file__).resolve().parent
_agg_spec = importlib.util.spec_from_file_location("exp025_aggregate", HERE / "aggregate.py")
aggregate_mod = importlib.util.module_from_spec(_agg_spec)
_agg_spec.loader.exec_module(aggregate_mod)


def build_configs(seeds: list[int], episodes: int, out_dir: Path) -> list[GenConfig]:
    """The 2 heads x 2 regimes x seeds sweep, each tagged uniquely by regime/head/seed."""
    configs = []
    for head_type in ("linear", "mlp"):
        for shaping in (True, False):
            regime = "shaped" if shaping else "sparse"
            for seed in seeds:
                configs.append(GenConfig(
                    seed=seed, episodes=episodes, shaping=shaping,
                    head_type=head_type, hidden=128,
                    tag=f"{regime}_{head_type}_seed{seed}", out_dir=out_dir,
                ))
    return configs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EXP-025 head capacity probe")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--episodes", type=int, default=600)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = HERE / "outputs"
    configs = build_configs(args.seeds, args.episodes, out_dir)
    summaries = []
    for i, cfg in enumerate(configs, 1):
        print(f"[{i}/{len(configs)}] {cfg.tag} ...", flush=True)
        summaries.append(run_generalization(cfg))

    agg = aggregate_mod.aggregate(summaries)
    table = aggregate_mod.format_table(agg)
    out_dir.mkdir(parents=True, exist_ok=True)
    # tuple keys are not JSON-serializable; join into "head|regime" strings
    agg_json = {f"{head}|{regime}": v for (head, regime), v in agg.items()}
    (out_dir / "025_summary.json").write_text(json.dumps(agg_json, indent=2))
    (out_dir / "025_table.md").write_text(table + "\n")
    print("\n" + table, flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Smoke-test the runner end to end (tiny)**

Run: `.venv/Scripts/python.exe experiments/025_head_capacity/run.py --seeds 0 1 --episodes 3`
Expected: prints `[1/8] shaped_linear_seed0 ...` through `[8/8] sparse_mlp_seed1 ...`, then a markdown table with 4 data rows; creates `experiments/025_head_capacity/outputs/025_summary.json` and `025_table.md`. (This is a wiring smoke test, not the real result.)

- [ ] **Step 7: Commit**

```bash
git add experiments/025_head_capacity/run.py experiments/025_head_capacity/aggregate.py tests/training/test_head_capacity_aggregate.py
git commit -m "feat: EXP-025 head capacity sweep runner and aggregator"
```

---

### Task 4: Run the full sweep and write the verdict

**Files:**
- Create: `experiments/025_head_capacity/FINDINGS.md`
- Create (generated): `experiments/025_head_capacity/outputs/025_summary.json`, `025_table.md`
- Modify (conditional): `docs/adr/0001-multi-region-training-strategy.md`

**Interfaces:**
- Consumes: the Task 3 runner.
- Produces: the evidence table, the verdict (head-limited vs encoder-limited), and the recommended next step.

- [ ] **Step 1: Run the full sweep**

Run: `.venv/Scripts/python.exe experiments/025_head_capacity/run.py`
Expected: 20 runs ([1/20]..[20/20]), then the markdown table; `outputs/025_summary.json` and `025_table.md` written. (Runtime is minutes; do not interrupt.)

- [ ] **Step 2: Apply the verdict rule**

Read `outputs/025_table.md`. Apply the spec's rule: a lift requires the MLP held-out mean to exceed `linear heldout_mean + linear heldout_spread` in at least the sparse regime.
- Lift -> verdict is "linear head was the limiter".
- No lift -> verdict is "frozen encoder is the wall".
- MLP mean lands inside the linear band (ambiguous) -> note that the planned escalation is a sparse-regime width sweep `{32, 64, 128, 256}` before concluding; do not overclaim.

- [ ] **Step 3: Write FINDINGS.md**

Create `experiments/025_head_capacity/FINDINGS.md` containing: the question, the embedded evidence table (paste from `025_table.md`), the verdict per Step 2, the paired-seed method note (5 seeds, identical goal split per seed), and the recommended Week-13 next step (engine unfreeze if encoder-limited; extract-more-from-head if head-limited). No em-dashes.

- [ ] **Step 4: Conditionally amend ADR-0001**

Only if the verdict is decisive (clear lift or clear no-lift, not ambiguous): append a one-paragraph amendment to `docs/adr/0001-multi-region-training-strategy.md` stating the EXP-025 result and what it implies for the next step. If ambiguous, skip this step and say so in FINDINGS.md.

- [ ] **Step 5: Run the full test suite**

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: all pass (>= 202 + the new EXP-025 tests).

- [ ] **Step 6: Commit**

```bash
git add experiments/025_head_capacity/FINDINGS.md experiments/025_head_capacity/outputs docs/adr/0001-multi-region-training-strategy.md
git commit -m "feat: EXP-025 head capacity findings and verdict"
```

---

## Self-Review

**Spec coverage:**
- Pluggable head (linear default byte-identical + MLP) -> Task 1.
- `head_type` threaded through harness, summary records it -> Task 2.
- Paired determinism guard -> Task 2 Step 1 (`test_run_generalization_is_deterministic...`).
- 2 heads x 2 regimes x 5 seeds sweep -> Task 3 `build_configs` + Task 4 Step 1.
- Aggregator with mean +/- spread, markdown table -> Task 3.
- Verdict logic / band rule -> Task 4 Step 2.
- FINDINGS.md + conditional ADR amendment -> Task 4 Steps 3-4.
- Per-seed-tagged outputs so no run overwrites another -> Task 3 `build_configs` tag.
- Keep 202 suite green -> Task 4 Step 5.
- Out-of-scope items (no unfreeze, no dashboard, no scaling, no width sweep unless ambiguous) -> respected; width sweep only referenced as conditional escalation in Task 4 Step 2.

**Placeholder scan:** No TBD/TODO; every code step shows full code; every run step shows the command and expected output.

**Type consistency:** `make_policy_head(brain, head_type, hidden)` signature identical in Tasks 1, 2, 3. `GenConfig.head_type`/`.hidden` consistent across Tasks 2-3. `aggregate`/`format_table` signatures consistent across Task 3 test, aggregator, and runner. Aggregate group key `(head_type, regime)` consistent between `aggregate`, `format_table`, and the runner's JSON-key join.
