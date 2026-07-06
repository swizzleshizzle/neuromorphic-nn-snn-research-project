# EXP-026 Sensory Pre-Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Pre-train the frozen sensory encoder to predict goal-relative displacement so its 64-d concept becomes a decodable state code, then re-run the generalization eval to test whether engaging the encoder lifts the EXP-025 held-out cap.

**Architecture:** A new `training/pretrain.py` supervised-trains `brain.sensory` (backprop through the spiking encoder via a scratch `Linear(64->2)` displacement readout). The generalization harness gains an opt-in, default-off `pretrain_sensory` hook that runs pre-training per seed before the RL loop; the existing `no_grad` RL path then freezes the encoder. A new EXP-026 runner sweeps the paired eval and reports the two-stage result.

**Tech Stack:** Python, PyTorch (`nn.Linear`, Adam, MSE), snntorch spiking encoder (surrogate gradients), pytest.

## Global Constraints

- Commit messages: plain, imperative, no `Co-Authored-By` / AI trailers, no em-dashes in prose or docs.
- Run Python via the repo venv: `.venv/Scripts/python.exe` (Windows). Tests: `.venv/Scripts/python.exe -m pytest <paths> -v`.
- `pretrain_sensory=False` (the default) must leave `run_generalization` byte-identical (EXP-023/024/025 reproducible). Guard by running the pre-training only under `if cfg.pretrain_sensory:`.
- Concept/action dims come from the brain (`brain.content == 64`, `brain.sensory` is the encoder); grid side is `cfg.size`. Never hardcode.
- The encoder is trained during pre-training, then frozen for RL (the existing `no_grad` path in `action_distribution` handles the freeze — no code change there).
- Determinism: every run is fully seeded by `cfg.seed` (encoder init, state split, Poisson encoding, goal split, head training).
- Keep the existing suite green (228 tests on `main`).

---

### Task 1: Pre-training pure helpers (targets, state enumeration, split)

**Files:**
- Create: `src/neuromorphic/training/pretrain.py`
- Test: `tests/training/test_pretrain.py`

**Interfaces:**
- Produces:
  - `displacement_target(obs: torch.Tensor, grid_n: int) -> torch.Tensor` — `[B, 4]` obs
    `(ax, ay, gx, gy)` -> `[B, 2]` float `((gx-ax), (gy-ay)) / (grid_n - 1)`, range `[-1, 1]`.
  - `enumerate_states(grid_n: int) -> torch.Tensor` — all `(agent, goal)` cell pairs with
    `agent != goal`, shape `[grid_n**2 * (grid_n**2 - 1), 4]`, dtype long.
  - `split_states(states, frac_heldout: float, seed: int) -> tuple[Tensor, Tensor]` —
    deterministic `(train, heldout)` row split.

- [ ] **Step 1: Write the failing tests**

Create `tests/training/test_pretrain.py`:

```python
import torch

from neuromorphic.training.pretrain import (
    displacement_target,
    enumerate_states,
    split_states,
)


def test_displacement_target_normalized_vectors():
    obs = torch.tensor([[0, 0, 4, 4], [4, 0, 0, 0], [2, 2, 2, 2]])
    d = displacement_target(obs, grid_n=5)
    assert d.shape == (3, 2)
    assert torch.allclose(d[0], torch.tensor([1.0, 1.0]))    # (4-0, 4-0)/4
    assert torch.allclose(d[1], torch.tensor([-1.0, 0.0]))   # (0-4, 0-0)/4
    assert torch.allclose(d[2], torch.tensor([0.0, 0.0]))    # same cell


def test_enumerate_states_count_and_no_self_pairs():
    states = enumerate_states(grid_n=5)
    assert states.shape == (5**2 * (5**2 - 1), 4)   # 600
    agent = states[:, :2]
    goal = states[:, 2:]
    assert not torch.any((agent == goal).all(dim=1))


def test_split_states_disjoint_sized_deterministic():
    states = enumerate_states(grid_n=5)
    tr_a, he_a = split_states(states, frac_heldout=0.2, seed=0)
    tr_b, he_b = split_states(states, frac_heldout=0.2, seed=0)
    assert he_a.shape[0] == round(states.shape[0] * 0.2)   # 120
    assert tr_a.shape[0] + he_a.shape[0] == states.shape[0]
    assert torch.equal(tr_a, tr_b) and torch.equal(he_a, he_b)   # deterministic
    # disjoint: no held-out row appears in train
    tr_set = {tuple(r.tolist()) for r in tr_a}
    assert all(tuple(r.tolist()) not in tr_set for r in he_a)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/training/test_pretrain.py -v`
Expected: FAIL (module `neuromorphic.training.pretrain` does not exist).

- [ ] **Step 3: Implement the helpers**

Create `src/neuromorphic/training/pretrain.py`:

```python
"""Supervised pre-training for the sensory encoder (EXP-026).

Pre-trains ``SensoryCortex`` so its concept code linearly exposes goal-relative
displacement ``(gx-ax, gy-ay)``. A scratch ``Linear(concept -> 2)`` readout shapes the
encoder via backprop through the spiking hierarchy; the readout is discarded afterward.
The encoder is then frozen for the RL policy (ADR-0001 Amendment 2 follow-up).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from neuromorphic.regions.sensory_cortex import encode_gridworld


def displacement_target(obs: torch.Tensor, grid_n: int) -> torch.Tensor:
    """``[B, 4]`` obs (ax, ay, gx, gy) -> ``[B, 2]`` normalized (gx-ax, gy-ay) / (grid_n-1)."""
    ax, ay, gx, gy = obs[:, 0], obs[:, 1], obs[:, 2], obs[:, 3]
    disp = torch.stack([gx - ax, gy - ay], dim=1).float()
    return disp / (grid_n - 1)


def enumerate_states(grid_n: int) -> torch.Tensor:
    """All (agent, goal) cell pairs with agent != goal -> ``[M, 4]`` long tensor."""
    cells = [(x, y) for x in range(grid_n) for y in range(grid_n)]
    rows = [
        [ax, ay, gx, gy]
        for (ax, ay) in cells
        for (gx, gy) in cells
        if (ax, ay) != (gx, gy)
    ]
    return torch.tensor(rows, dtype=torch.long)


def split_states(states: torch.Tensor, frac_heldout: float, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministic (train, heldout) row split of ``states`` by ``seed``."""
    n = states.shape[0]
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    n_held = round(n * frac_heldout)
    return states[perm[n_held:]], states[perm[:n_held]]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/training/test_pretrain.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/neuromorphic/training/pretrain.py tests/training/test_pretrain.py
git commit -m "feat: sensory pre-training helpers (displacement target, state split)"
```

---

### Task 2: Differentiable encoder batch + the pre-training loop

**Files:**
- Modify: `src/neuromorphic/training/pretrain.py`
- Test: `tests/training/test_pretrain.py`

**Interfaces:**
- Consumes: `displacement_target`, `enumerate_states`, `split_states` (Task 1);
  `encode_gridworld` and `SensoryCortex` from `neuromorphic.regions.sensory_cortex`.
- Produces:
  - `concept_rate_batch(sensory, obs, grid_n, T=32, generator=None) -> torch.Tensor` — encode a
    `[B, 4]` obs batch (`encode_gridworld` -> `sensory.forward`, **with grad**) and return the
    mean-over-`T` concept firing rate `[B, concept]`. Must be differentiable w.r.t. the encoder
    weights (does NOT wrap the encoder in `no_grad`).
  - `pretrain_sensory(sensory, *, grid_n, epochs=200, lr=1e-3, frac_heldout=0.2, seed=0, T=32, generator=None, freeze_encoder=False) -> dict` —
    trains a scratch `Linear(sensory.concept -> 2)` readout on `concept_rate_batch` toward
    `displacement_target` (MSE, Adam over encoder+readout params, or readout-only if
    `freeze_encoder=True`). Returns `{"train_disp_error": float, "heldout_disp_error": float,
    "epochs": int, "freeze_encoder": bool}` where the errors are mean-absolute error on the
    respective state split. The readout is discarded (not returned).

- [ ] **Step 1: Write the failing tests**

Add to `tests/training/test_pretrain.py`:

```python
from neuromorphic.regions.sensory_cortex import SensoryCortex
from neuromorphic.training.pretrain import concept_rate_batch, pretrain_sensory


def _fresh_sensory(seed=0):
    # n_obs = 2 * grid_n**2 for a 5x5 grid = 50
    return SensoryCortex(n_obs=2 * 5 * 5, seed=seed)


def test_concept_rate_batch_shape_and_differentiable():
    sensory = _fresh_sensory()
    obs = torch.tensor([[0, 0, 4, 4], [1, 2, 3, 0]])
    gen = torch.Generator().manual_seed(0)
    rate = concept_rate_batch(sensory, obs, grid_n=5, T=8, generator=gen)
    assert rate.shape == (2, sensory.concept)
    # gradient must reach the encoder weights (the load-bearing requirement)
    rate.sum().backward()
    assert sensory.fc1.weight.grad is not None
    assert torch.any(sensory.fc1.weight.grad != 0)


def test_pretrain_sensory_learns_and_changes_encoder():
    sensory = _fresh_sensory(seed=0)
    w1_before = sensory.fc1.weight.detach().clone()
    info = pretrain_sensory(
        sensory, grid_n=5, epochs=30, lr=1e-3, frac_heldout=0.2, seed=0, T=8,
        generator=torch.Generator().manual_seed(0),
    )
    assert set(info) == {"train_disp_error", "heldout_disp_error", "epochs", "freeze_encoder"}
    assert info["epochs"] == 30
    assert info["freeze_encoder"] is False
    assert torch.isfinite(torch.tensor(info["heldout_disp_error"]))
    # encoder weights moved
    assert not torch.equal(w1_before, sensory.fc1.weight.detach())


def test_pretrain_sensory_freeze_encoder_leaves_weights_unchanged():
    sensory = _fresh_sensory(seed=0)
    w1_before = sensory.fc1.weight.detach().clone()
    info = pretrain_sensory(
        sensory, grid_n=5, epochs=20, lr=1e-3, seed=0, T=8,
        generator=torch.Generator().manual_seed(0), freeze_encoder=True,
    )
    assert info["freeze_encoder"] is True
    # only the scratch readout trained; the encoder is untouched
    assert torch.equal(w1_before, sensory.fc1.weight.detach())


def test_pretrain_sensory_is_deterministic():
    def run():
        s = _fresh_sensory(seed=1)
        return pretrain_sensory(
            s, grid_n=5, epochs=20, lr=1e-3, seed=1, T=8,
            generator=torch.Generator().manual_seed(1),
        )["heldout_disp_error"]
    assert run() == run()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/training/test_pretrain.py -k "concept_rate or pretrain_sensory" -v`
Expected: FAIL (`concept_rate_batch` / `pretrain_sensory` not defined).

- [ ] **Step 3: Implement**

Append to `src/neuromorphic/training/pretrain.py`:

```python
def concept_rate_batch(sensory, obs, grid_n, T: int = 32, generator=None) -> torch.Tensor:
    """Encode ``[B, 4]`` obs -> mean-over-T concept rate ``[B, concept]``, differentiable.

    Calls the encoder directly (NOT via ``brain.step``, which wraps it in ``no_grad``) so
    gradients reach ``sensory.fc1`` / ``fc2`` during pre-training.
    """
    spikes = encode_gridworld(obs, grid_n, T=T, generator=generator)   # [T, B, N_obs]
    concept = sensory(spikes)                                          # [T, B, concept]
    return concept.mean(dim=0)                                         # [B, concept]


def _disp_mae(readout, sensory, obs, grid_n, T, generator) -> float:
    with torch.no_grad():
        pred = readout(concept_rate_batch(sensory, obs, grid_n, T=T, generator=generator))
        return float((pred - displacement_target(obs, grid_n)).abs().mean())


def pretrain_sensory(
    sensory, *, grid_n, epochs: int = 200, lr: float = 1e-3, frac_heldout: float = 0.2,
    seed: int = 0, T: int = 32, generator=None, freeze_encoder: bool = False,
) -> dict:
    """Pre-train ``sensory`` so its concept linearly decodes goal-relative displacement.

    Trains a scratch ``Linear(concept -> 2)`` readout (and the encoder unless
    ``freeze_encoder``) with MSE + Adam over the train state split; reports mean-absolute
    displacement error on the train and held-out splits. The readout is discarded.
    """
    torch.manual_seed(seed)
    states = enumerate_states(grid_n)
    train_states, heldout_states = split_states(states, frac_heldout, seed)

    readout = nn.Linear(sensory.concept, 2)
    params = list(readout.parameters())
    if not freeze_encoder:
        params += list(sensory.parameters())
    opt = torch.optim.Adam(params, lr=lr)

    for _ in range(epochs):
        rate = concept_rate_batch(sensory, train_states, grid_n, T=T, generator=generator)
        pred = readout(rate)
        loss = ((pred - displacement_target(train_states, grid_n)) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

    return {
        "train_disp_error": _disp_mae(readout, sensory, train_states, grid_n, T, generator),
        "heldout_disp_error": _disp_mae(readout, sensory, heldout_states, grid_n, T, generator),
        "epochs": epochs,
        "freeze_encoder": freeze_encoder,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/training/test_pretrain.py -v`
Expected: PASS (all pretrain tests). Note: `test_pretrain_sensory_learns_and_changes_encoder`
asserts the encoder weights move; if it is flaky at 30 epochs, that is a real signal the loop
is not learning — do not weaken the assertion, investigate the loss.

- [ ] **Step 5: Commit**

```bash
git add src/neuromorphic/training/pretrain.py tests/training/test_pretrain.py
git commit -m "feat: sensory encoder pre-training loop (displacement readout)"
```

---

### Task 3: Wire pre-training into the generalization harness

**Files:**
- Modify: `src/neuromorphic/training/generalization.py` (`GenConfig`, `run_generalization`)
- Test: `tests/training/test_generalization.py`

**Interfaces:**
- Consumes: `pretrain_sensory` (Task 2).
- Produces: `GenConfig` gains `pretrain_sensory: bool = False`, `pretrain_epochs: int = 200`,
  `pretrain_lr: float = 1e-3`. When `pretrain_sensory=True`, `run_generalization` pre-trains
  `brain.sensory` before the RL loop and stores the returned dict under `summary["pretrain"]`
  (else `summary["pretrain"] is None`).

- [ ] **Step 1: Write the failing tests**

Add to `tests/training/test_generalization.py`:

```python
def test_genconfig_defaults_no_pretrain():
    cfg = GenConfig()
    assert cfg.pretrain_sensory is False
    assert cfg.pretrain_epochs == 200
    assert cfg.pretrain_lr == 1e-3


def test_run_generalization_without_pretrain_has_null_pretrain(tmp_path):
    cfg = GenConfig(seed=0, episodes=2, n_heldout=2, max_steps=8, tag="no_pt", out_dir=tmp_path)
    summary = run_generalization(cfg)
    assert summary["pretrain"] is None


def test_run_generalization_with_pretrain_records_gate_and_changes_encoder(tmp_path):
    import torch
    from neuromorphic.brain import Brain

    random_w1 = Brain(grid_n=5, seed=0).sensory.fc1.weight.detach().clone()
    cfg = GenConfig(
        seed=0, episodes=2, n_heldout=2, max_steps=8,
        pretrain_sensory=True, pretrain_epochs=10, tag="pt", out_dir=tmp_path,
    )
    summary = run_generalization(cfg)
    assert summary["pretrain"] is not None
    assert "heldout_disp_error" in summary["pretrain"]
    assert summary["config"]["pretrain_sensory"] is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/training/test_generalization.py -k "pretrain" -v`
Expected: FAIL (`GenConfig` has no `pretrain_sensory` field; `summary` has no `pretrain` key).

- [ ] **Step 3: Implement**

In `src/neuromorphic/training/generalization.py`:

Add the import near the top (with the other `neuromorphic.training` imports):

```python
from neuromorphic.training.pretrain import pretrain_sensory
```

Add three fields to `GenConfig` (next to the other model-ish fields, before `tag`):

```python
    pretrain_sensory: bool = False
    pretrain_epochs: int = 200
    pretrain_lr: float = 1e-3
```

In `run_generalization`, insert the hook between `brain = Brain(...)` and
`head = make_policy_head(...)`:

```python
    brain = Brain(grid_n=cfg.size, seed=cfg.seed)
    pretrain_info = None
    if cfg.pretrain_sensory:
        pretrain_info = pretrain_sensory(
            brain.sensory, grid_n=cfg.size, epochs=cfg.pretrain_epochs, lr=cfg.pretrain_lr,
            seed=cfg.seed, generator=torch.Generator().manual_seed(cfg.seed),
        )
    head = make_policy_head(brain, head_type=cfg.head_type, hidden=cfg.hidden)
```

Add `"pretrain": pretrain_info,` to the `summary` dict (e.g. right after the `"eval"` entry).

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/training/test_generalization.py -v`
Expected: PASS (new + all pre-existing generalization tests, including the byte-identity
determinism test which exercises the `pretrain_sensory=False` default path).

- [ ] **Step 5: Commit**

```bash
git add src/neuromorphic/training/generalization.py tests/training/test_generalization.py
git commit -m "feat: pretrain_sensory hook in generalization harness"
```

---

### Task 4: EXP-026 runner (paired sweep + random-encoder reference)

**Files:**
- Create: `experiments/026_sensory_pretrain/run.py`
- Test: `tests/training/test_sensory_pretrain_run.py`

**Interfaces:**
- Consumes: `GenConfig`, `run_generalization` (Task 3); `pretrain_sensory` (Task 2); the
  EXP-025 aggregator (`experiments/025_head_capacity/aggregate.py`).
- Produces: `experiments/026_sensory_pretrain/run.py` with
  `build_configs(seeds, episodes, out_dir) -> list[GenConfig]` (linear head, `pretrain_sensory=True`,
  `{shaped, sparse} x seeds`, tags suffixed `_pt`) and a `main()` that runs the sweep, prints the
  Stage-1 gate metrics per seed plus a Stage-2 held-out table, and writes
  `outputs/026_summary.json` / `026_table.md`.

- [ ] **Step 1: Write the failing test**

Create `tests/training/test_sensory_pretrain_run.py`:

```python
import importlib.util as ilu
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_spec = ilu.spec_from_file_location(
    "exp026_run", ROOT / "experiments" / "026_sensory_pretrain" / "run.py"
)
run_mod = ilu.module_from_spec(_spec)
_spec.loader.exec_module(run_mod)


def test_build_configs_pretrain_linear_both_regimes(tmp_path):
    cfgs = run_mod.build_configs([0, 1], episodes=5, out_dir=tmp_path)
    assert len(cfgs) == 4   # 2 regimes x 2 seeds, linear only
    assert all(c.pretrain_sensory is True for c in cfgs)
    assert all(c.head_type == "linear" for c in cfgs)
    assert all(c.tag.endswith("_pt") for c in cfgs)
    assert {c.shaping for c in cfgs} == {True, False}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/training/test_sensory_pretrain_run.py -v`
Expected: FAIL (`run.py` does not exist -> import error).

- [ ] **Step 3: Implement the runner**

Create `experiments/026_sensory_pretrain/run.py`:

```python
"""EXP-026 - sensory pre-training (engage the encoder).

Two-stage test of ADR-0001 Amendment 2 (the frozen encoder is the cap):
  Stage 1 - pre-train the sensory encoder to decode goal-relative displacement; gate on the
            held-out displacement error vs a random-encoder reference.
  Stage 2 - freeze the pre-trained encoder, train the linear policy head, and compare held-out
            navigation success against the EXP-025 random-encoder band (shaped 23%, sparse 27%).

Run (repo root, venv active):
    .venv/Scripts/python.exe experiments/026_sensory_pretrain/run.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import torch  # noqa: E402

from neuromorphic.brain import Brain
from neuromorphic.training.generalization import GenConfig, run_generalization
from neuromorphic.training.pretrain import pretrain_sensory

torch.set_num_threads(1)

HERE = Path(__file__).resolve().parent
_agg_spec = importlib.util.spec_from_file_location(
    "exp025_aggregate", HERE.parent / "025_head_capacity" / "aggregate.py"
)
aggregate_mod = importlib.util.module_from_spec(_agg_spec)
_agg_spec.loader.exec_module(aggregate_mod)


def build_configs(seeds, episodes, out_dir):
    """Linear head, pretrain_sensory=True, {shaped, sparse} x seeds; tags suffixed _pt."""
    configs = []
    for shaping in (True, False):
        regime = "shaped" if shaping else "sparse"
        for seed in seeds:
            configs.append(GenConfig(
                seed=seed, episodes=episodes, shaping=shaping, head_type="linear",
                pretrain_sensory=True, tag=f"{regime}_linear_seed{seed}_pt", out_dir=out_dir,
            ))
    return configs


def _run_one(cfg):
    torch.set_num_threads(1)
    return run_generalization(cfg)


def _random_reference(seed):
    """Stage-1 reference: can a linear readout decode displacement from a RANDOM encoder?"""
    torch.set_num_threads(1)
    brain = Brain(grid_n=5, seed=seed)
    info = pretrain_sensory(
        brain.sensory, grid_n=5, epochs=200, lr=1e-3, seed=seed,
        generator=torch.Generator().manual_seed(seed), freeze_encoder=True,
    )
    return seed, info["heldout_disp_error"]


def parse_args():
    p = argparse.ArgumentParser(description="EXP-026 sensory pre-training")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--episodes", type=int, default=600)
    p.add_argument("--workers", type=int, default=10)
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = HERE / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    configs = build_configs(args.seeds, args.episodes, out_dir)
    workers = max(1, min(args.workers, len(configs)))

    print(f"running {len(configs)} pretrain configs across {workers} workers ...", flush=True)
    summaries = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_run_one, cfg): cfg for cfg in configs}
        for done, fut in enumerate(as_completed(futures), 1):
            cfg = futures[fut]
            summaries.append(fut.result())
            print(f"[{done}/{len(configs)}] {cfg.tag} done", flush=True)

    # Stage 1 gate: pre-trained vs random-encoder held-out displacement error, per seed
    print("\n=== Stage 1: displacement decode error (held-out states) ===", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        rand_ref = dict(ex.map(_random_reference, args.seeds))
    pt_err = {}
    for s in summaries:
        seed = s["config"]["seed"]
        pt_err.setdefault(seed, s["pretrain"]["heldout_disp_error"])
    for seed in args.seeds:
        print(f"  seed {seed}: pretrained {pt_err[seed]:.3f}  vs  random {rand_ref[seed]:.3f}",
              flush=True)

    # Stage 2 table: held-out navigation success (reuse the EXP-025 aggregator)
    agg = aggregate_mod.aggregate(summaries)
    table = aggregate_mod.format_table(agg)
    agg_json = {f"{head}|{regime}": v for (head, regime), v in agg.items()}
    (out_dir / "026_summary.json").write_text(json.dumps(
        {"stage2": agg_json, "stage1": {"pretrained": pt_err, "random": rand_ref}}, indent=2))
    (out_dir / "026_table.md").write_text(table + "\n")
    print("\n=== Stage 2: held-out navigation success ===\n" + table, flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/training/test_sensory_pretrain_run.py -v`
Expected: PASS.

- [ ] **Step 5: Smoke-test the runner end to end (tiny)**

Run: `.venv/Scripts/python.exe experiments/026_sensory_pretrain/run.py --seeds 0 1 --episodes 3`
Expected: prints `[1/4]`..`[4/4]` done, a Stage-1 block (`seed 0: pretrained X vs random Y`), and
a Stage-2 4-row table; creates `experiments/026_sensory_pretrain/outputs/026_summary.json` and
`026_table.md`. (Wiring smoke test at tiny epochs/episodes — not the real result. Note: 200
pretrain epochs run even in the smoke test since `pretrain_epochs` is not reduced by `--episodes`;
if the smoke run is slow, pass a short-epoch config is out of scope here, just let it run.)

- [ ] **Step 6: Commit**

```bash
git add experiments/026_sensory_pretrain/run.py tests/training/test_sensory_pretrain_run.py
git commit -m "feat: EXP-026 sensory pre-training runner (two-stage)"
```

---

### Task 5: Run the full experiment and read the two-stage verdict (with Mike)

**Files:**
- Create (generated): `experiments/026_sensory_pretrain/outputs/026_table.md`, `026_summary.json`
- Update: the Week-13/14 obsidian note (Session 3 findings)
- Modify (conditional): `docs/adr/0001-multi-region-training-strategy.md`

- [ ] **Step 1: Run the full experiment**

Run: `.venv/Scripts/python.exe experiments/026_sensory_pretrain/run.py`
Expected: 10 configs (2 regimes x 5 seeds) + the 5 random references; Stage-1 per-seed error
lines and the Stage-2 table; outputs written. (Runtime: pre-training adds ~minutes per seed on
top of the ~1h RL sweep; do not interrupt.)

- [ ] **Step 2: Apply the Stage-1 gate**

Read the Stage-1 block. Gate passes when the pre-trained held-out displacement error is clearly
below the random-encoder reference and low in absolute terms (target: pre-trained held-out MAE
< ~0.75 normalized units). If the gate fails (pre-training did not make displacement decodable),
STOP and report - Stage 2 is uninterpretable; the pre-training objective or hyperparameters need
revisiting before any encoder-cap conclusion.

- [ ] **Step 3: Apply the Stage-2 band rule (with Mike)**

Compare the pre-trained held-out success (`026_table.md`) against the EXP-025 random-encoder
linear band (shaped 23%, sparse 27% held-out; spreads from `experiments/025_head_capacity`).
- Clears the band -> engaging the encoder lifted the cap; Option B validated; regions now
  specialize through learning -> ablation studies become the next meaningful work.
- Does not clear the band -> the bottleneck is deeper than the sensory encoder -> rethink.

- [ ] **Step 4: Write up + conditional ADR amendment**

Append a Session-3 findings block to the Week-13/14 obsidian note with the Stage-1 gate table,
the Stage-2 held-out table, and the verdict. Amend ADR-0001 (Amendment 3) only if the result is
decisive.

- [ ] **Step 5: Full suite**

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: all pass.

---

## Self-Review

**Spec coverage:**
- Displacement objective + normalized target -> Task 1 (`displacement_target`).
- Differentiable encoder batch (gradients reach the encoder) -> Task 2 (`concept_rate_batch` +
  its differentiability test).
- Pre-training loop, discarded readout, gate metrics -> Task 2 (`pretrain_sensory`).
- Enumerable state data + held-out split for the gate -> Task 1 (`enumerate_states`,
  `split_states`).
- Random-encoder reference for the gate -> Task 2 (`freeze_encoder`) + Task 4 (`_random_reference`).
- In-harness, per-seed, default-off integration + byte-identity -> Task 3.
- Paired sweep, both regimes, linear head, reuse EXP-025 aggregator -> Task 4.
- Two-stage verdict + writeup + conditional ADR -> Task 5.
- Determinism by seed -> Task 2 determinism test; Task 3 exercises the default byte-identity path.

**Placeholder scan:** No TBD/TODO; every code step shows full code; every run step shows the
command and expected output.

**Type consistency:** `pretrain_sensory(sensory, *, grid_n, epochs, lr, frac_heldout, seed, T,
generator, freeze_encoder) -> dict` identical across Task 2 (def), Task 3 (harness call), and
Task 4 (`_random_reference` call). `concept_rate_batch(sensory, obs, grid_n, T, generator)`
consistent between Task 2 def and its use inside `pretrain_sensory`. `GenConfig.pretrain_sensory/
pretrain_epochs/pretrain_lr` consistent across Tasks 3-4. The gate dict keys
(`train_disp_error`, `heldout_disp_error`, `epochs`, `freeze_encoder`) match between Task 2's
return, its tests, and Task 4's `summary["pretrain"]["heldout_disp_error"]` access.
