# EXP-027 Encoder Characterization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Characterize the EXP-026-engaged sensory encoder - show it carries task-structured activity the frozen spectators lack (cross-region decodability), quantify how distributed/load-bearing it is (geometry + causal dropout-on-navigation), with checkpoint-save so trained models reload instead of recomputing.

**Architecture:** A new `analysis/probes.py` of pure, brain/training-agnostic primitives (probes, geometry, activity extraction with zero-fill for bypassed regions). Lightweight checkpoint save/load added to the training path. An `experiments/027_encoder_characterization/` driver runs Component A (encoder-only decodability + geometry, seconds/seed) and Component B (dropout-on-navigation via a `MaskedHead` on reloaded checkpoints), paired across the 12 EXP-026 seeds.

**Tech Stack:** Python, PyTorch (`torch.linalg`, `nn.Linear`, Adam), snntorch spiking Brain, pytest.

## Global Constraints

- Commit messages: plain, imperative, no `Co-Authored-By` / AI trailers, no em-dashes in prose or docs.
- Run Python via the repo venv: `.venv/Scripts/python.exe` (Windows). Tests: `.venv/Scripts/python.exe -m pytest <paths> -v`.
- No edits to existing src behavior: `checkpoint`/probe additions are opt-in and default-off; the `pretrain_sensory=False` / no-checkpoint paths stay byte-identical (EXP-023/024/025/026 reproducible). Do NOT edit `greedy_action`, `evaluate`, `render_dashboard`, or the JSONL trace contract.
- Recordings are keyed by the `Brain._regions` names: `sensory`, `hippocampus`, `prefrontal`, `router`, `motor` (NOT `pfc`/`sensory_cortex`). `brain.step(record=True)["recordings"][region]` is a dict `{signal_key: [T,B,N]}`; a bypassed region (hippocampus under `recall=False`) yields an EMPTY dict - `region_rate_matrix` must zero-fill, never index-crash.
- `ridge_probe` is EXPLICIT L2 (augmented system or normal equations), a shared lambda across regions - NOT bare `torch.linalg.lstsq` (that is unregularized OLS).
- Per-action optimality uses the `GridWorldEnv` transition: action a -> next cell `(clip(ax+dx,0,N-1), clip(ay+dy,0,N-1))` with `_DELTAS=((0,-1),(1,0),(0,1),(-1,0))`; optimal iff it strictly reduces Manhattan distance to the goal. Chance is the shuffle-null band, not a literal 25%.
- Determinism: everything seeded by the run seed (encoder init, state split, probe fits, masks).
- Keep the existing suite green (239 tests on `main`).

## File structure

- `src/neuromorphic/analysis/__init__.py`, `src/neuromorphic/analysis/probes.py` - primitives (Tasks 1-2).
- `src/neuromorphic/training/checkpoints.py` - `save_trained` / `load_trained` (Task 3).
- `src/neuromorphic/training/generalization.py` - opt-in `checkpoint_path` on `GenConfig`/`run_generalization` (Task 3).
- `experiments/027_encoder_characterization/{probe.py,dropout_eval.py,run.py,aggregate.py}` (Tasks 4-6).
- `tests/training/test_probes.py`, `tests/training/test_checkpoints.py`, `tests/training/test_dropout_eval.py`.

---

### Task 1: Probe + geometry primitives (pure numeric)

**Files:**
- Create: `src/neuromorphic/analysis/__init__.py` (empty), `src/neuromorphic/analysis/probes.py`
- Test: `tests/training/test_probes.py`

**Interfaces:**
- Produces (all pure, torch-only):
  - `optimal_action_targets(states, grid_n) -> torch.Tensor` `[M,4]` float 0/1: entry (i,a)=1 iff action a strictly reduces Manhattan distance from `states[i]` agent to goal, using the clipped `_DELTAS` transition.
  - `ridge_probe(X_tr, Y_tr, X_te, Y_te, lam) -> dict` with `r2` (held-out R2, averaged over targets), `weights` `[N, Ytarget]`. Explicit L2.
  - `peraction_probe(X_tr, Y_tr, X_te, Y_te) -> dict` with per-action `acc` `[4]` and `mean_acc` (Y are 0/1 per-action labels; one linear+sigmoid probe per action).
  - `shuffle_null(fit_fn, X_tr, Y_tr, X_te, Y_te, *, n, seed) -> dict` with `mean`, `hi` (permute Y_tr rows, refit, collect the metric; empirical chance band).
  - `pca_reduce(X_tr, X_te, k) -> (Xtr_k, Xte_k)` fit PCA on train, apply to both.
  - `participation_ratio(X) -> float` = `(sum lam)^2 / sum(lam^2)` of the covariance eigenvalues.
  - `unit_importance(X, Y) -> torch.Tensor` `[N]` ranking (abs ridge weight magnitude summed over targets).
  - `keepk_curve(X_tr, Y_tr, X_te, Y_te, order, ks, lam) -> list[dict]` R2 keeping the top-k units in `order`.

- [ ] **Step 1: Write the failing tests**

Create `tests/training/test_probes.py`:

```python
import torch

from neuromorphic.analysis.probes import (
    optimal_action_targets, ridge_probe, peraction_probe, shuffle_null,
    pca_reduce, participation_ratio, unit_importance, keepk_curve,
)


def test_optimal_action_targets_known_cases():
    # actions: 0=up(0,-1) 1=right(1,0) 2=down(0,1) 3=left(-1,0); optimal reduces manhattan
    states = torch.tensor([[0, 0, 4, 4], [4, 4, 0, 0], [2, 2, 2, 0]])
    T = optimal_action_targets(states, grid_n=5)
    assert T.shape == (3, 4)
    # agent(0,0) goal(4,4): right and down reduce distance; up/left clip or increase
    assert T[0].tolist() == [0.0, 1.0, 1.0, 0.0]
    # agent(4,4) goal(0,0): up and left reduce
    assert T[1].tolist() == [1.0, 0.0, 0.0, 1.0]
    # agent(2,2) goal(2,0): only up reduces (goal directly above)
    assert T[2].tolist() == [1.0, 0.0, 0.0, 0.0]


def test_ridge_probe_recovers_linear_signal_and_is_regularized():
    torch.manual_seed(0)
    N, M = 8, 200
    W = torch.randn(N, 2)
    X = torch.randn(M, N)
    Y = X @ W + 0.01 * torch.randn(M, 2)
    out = ridge_probe(X[:150], Y[:150], X[150:], Y[150:], lam=1e-2)
    assert out["r2"] > 0.95            # recovers the linear map
    assert out["weights"].shape == (N, 2)
    # ridge shrinks vs lambda=0 (OLS): larger lambda -> smaller weight norm
    small = ridge_probe(X[:150], Y[:150], X[150:], Y[150:], lam=1e-3)["weights"].norm()
    big = ridge_probe(X[:150], Y[:150], X[150:], Y[150:], lam=1e2)["weights"].norm()
    assert big < small


def test_peraction_probe_shape_and_range():
    torch.manual_seed(0)
    X = torch.randn(120, 16)
    Y = (torch.randn(120, 4) > 0).float()
    out = peraction_probe(X[:90], Y[:90], X[90:], Y[90:])
    assert out["acc"].shape == (4,)
    assert 0.0 <= float(out["mean_acc"]) <= 1.0


def test_shuffle_null_band_is_near_chance():
    torch.manual_seed(0)
    X = torch.randn(120, 8)
    Y = torch.randn(120, 2)   # no real signal
    def fit(a, b, c, d): return ridge_probe(a, b, c, d, lam=1e-2)["r2"]
    null = shuffle_null(fit, X[:90], Y[:90], X[90:], Y[90:], n=10, seed=0)
    assert null["hi"] < 0.3   # permuted-label R2 stays near zero


def test_participation_ratio_bounds():
    torch.manual_seed(0)
    iso = torch.randn(200, 10)                      # ~isotropic -> PR near 10
    rank1 = torch.randn(200, 1) @ torch.randn(1, 10)  # rank-1 -> PR near 1
    assert participation_ratio(iso) > 5.0
    assert participation_ratio(rank1) < 2.0


def test_keepk_curve_monotone_top_units():
    torch.manual_seed(0)
    N = 12
    X = torch.randn(200, N)
    Y = X[:, :3] @ torch.randn(3, 1)   # only first 3 units carry signal
    order = unit_importance(X[:150], Y[:150])
    curve = keepk_curve(X[:150], Y[:150], X[150:], Y[150:], order=order, ks=[1, 3, 12], lam=1e-2)
    r2 = {c["k"]: c["r2"] for c in curve}
    assert r2[3] > 0.8 and r2[3] >= r2[1] - 1e-6   # top-3 recover most signal
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/training/test_probes.py -v`
Expected: FAIL (module `neuromorphic.analysis.probes` does not exist).

- [ ] **Step 3: Implement the primitives**

Create `src/neuromorphic/analysis/__init__.py` (empty). Create `src/neuromorphic/analysis/probes.py`:

```python
"""Linear-probe + representation-geometry primitives for encoder characterization (EXP-027)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.optim import Adam

from neuromorphic.envs.gridworld import manhattan

_DELTAS = ((0, -1), (1, 0), (0, 1), (-1, 0))  # up, right, down, left (matches GridWorldEnv)


def optimal_action_targets(states: torch.Tensor, grid_n: int) -> torch.Tensor:
    """[M,4] float: (i,a)=1 iff action a strictly reduces Manhattan distance to the goal."""
    M = states.shape[0]
    out = torch.zeros(M, 4)
    for i in range(M):
        ax, ay, gx, gy = (int(v) for v in states[i])
        d0 = manhattan((ax, ay), (gx, gy))
        for a, (dx, dy) in enumerate(_DELTAS):
            nx = min(max(ax + dx, 0), grid_n - 1)
            ny = min(max(ay + dy, 0), grid_n - 1)
            if manhattan((nx, ny), (gx, gy)) < d0:
                out[i, a] = 1.0
    return out


def _r2(pred: torch.Tensor, y: torch.Tensor) -> float:
    ss_res = ((y - pred) ** 2).sum()
    ss_tot = ((y - y.mean(dim=0)) ** 2).sum()
    return float(1.0 - ss_res / (ss_tot + 1e-12))


def ridge_probe(X_tr, Y_tr, X_te, Y_te, lam: float) -> dict:
    """Explicit L2-penalized linear probe (normal equations); held-out R2 + weights."""
    N = X_tr.shape[1]
    A = X_tr.T @ X_tr + lam * torch.eye(N)
    W = torch.linalg.solve(A, X_tr.T @ Y_tr)   # [N, Ytarget]
    return {"r2": _r2(X_te @ W, Y_te), "weights": W}


def peraction_probe(X_tr, Y_tr, X_te, Y_te, *, epochs: int = 200, lr: float = 1e-2) -> dict:
    """Four independent linear+sigmoid probes; per-action held-out accuracy."""
    torch.manual_seed(0)
    probe = nn.Linear(X_tr.shape[1], 4)
    opt = Adam(probe.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        opt.zero_grad()
        loss_fn(probe(X_tr), Y_tr).backward()
        opt.step()
    with torch.no_grad():
        pred = (probe(X_te) > 0).float()
        acc = (pred == Y_te).float().mean(dim=0)   # [4]
    return {"acc": acc, "mean_acc": float(acc.mean())}


def shuffle_null(fit_fn, X_tr, Y_tr, X_te, Y_te, *, n: int, seed: int) -> dict:
    """Permute train labels, refit n times; empirical chance band (mean, hi=max)."""
    vals = []
    for i in range(n):
        g = torch.Generator().manual_seed(seed + i)
        perm = torch.randperm(Y_tr.shape[0], generator=g)
        vals.append(float(fit_fn(X_tr, Y_tr[perm], X_te, Y_te)))
    t = torch.tensor(vals)
    return {"mean": float(t.mean()), "hi": float(t.max())}


def pca_reduce(X_tr, X_te, k: int):
    """Fit PCA on train (center + top-k right singular vectors), apply to both."""
    mu = X_tr.mean(dim=0, keepdim=True)
    U, S, Vh = torch.linalg.svd(X_tr - mu, full_matrices=False)
    comp = Vh[:k].T                       # [N, k]
    return (X_tr - mu) @ comp, (X_te - mu) @ comp


def participation_ratio(X) -> float:
    """(sum lam)^2 / sum(lam^2) of the covariance spectrum; ~effective dimensionality."""
    Xc = X - X.mean(dim=0, keepdim=True)
    cov = (Xc.T @ Xc) / max(X.shape[0] - 1, 1)
    lam = torch.linalg.eigvalsh(cov).clamp(min=0)
    return float((lam.sum() ** 2) / ((lam ** 2).sum() + 1e-12))


def unit_importance(X, Y, *, lam: float = 1e-2) -> torch.Tensor:
    """Rank units by summed abs ridge weight; returns unit indices, most important first."""
    W = ridge_probe(X, Y, X, Y, lam=lam)["weights"]   # [N, Yt]
    score = W.abs().sum(dim=1)
    return torch.argsort(score, descending=True)


def keepk_curve(X_tr, Y_tr, X_te, Y_te, order, ks, lam: float) -> list:
    """Held-out R2 keeping only the top-k units (by `order`) for each k in ks."""
    out = []
    for k in ks:
        idx = order[:k]
        out.append({"k": int(k), "r2": ridge_probe(X_tr[:, idx], Y_tr, X_te[:, idx], Y_te, lam=lam)["r2"]})
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/training/test_probes.py -v`
Expected: PASS (7 tests).

- [ ] **Step 5: Commit**

```bash
git add src/neuromorphic/analysis/ tests/training/test_probes.py
git commit -m "feat: linear-probe and geometry primitives for encoder characterization"
```

---

### Task 2: region_rate_matrix + task_targets (activity extraction with zero-fill)

**Files:**
- Modify: `src/neuromorphic/analysis/probes.py`
- Test: `tests/training/test_probes.py`

**Interfaces:**
- Consumes: `Brain.step`, `pretrain.displacement_target`, `optimal_action_targets` (Task 1).
- Produces:
  - `region_rate_matrix(brain, states, *, region_key, signal_key, width, recall=False, T=32, generator=None) -> torch.Tensor` `[M, width]` mean-over-T rate; **zero-fills** `[M, width]` when the region produced no recording (bypassed hippocampus under `recall=False`).
  - `task_targets(states, grid_n) -> dict` with `displacement` `[M,2]` (`pretrain.displacement_target`) and `optimal_action` `[M,4]` (`optimal_action_targets`).
  - `REGION_SIGNALS: dict[str, tuple[str, int]]` mapping region_key -> (signal_key, width): `sensory`->`("concept",64)` (plus a `sensory_hidden`->`("hidden",128)` alias handled by callers), `prefrontal`->`("utility",4)`, `prefrontal_state`->`("state",100)`, `router`->`("gate",4)`, `motor`->`("action",4)`, `hippocampus`->`("population",150)`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/training/test_probes.py`:

```python
from neuromorphic.brain import Brain
from neuromorphic.analysis.probes import region_rate_matrix, task_targets


def test_region_rate_matrix_sensory_shape_and_state_dependent():
    brain = Brain(grid_n=5, seed=0)
    states = torch.tensor([[0, 0, 4, 4], [4, 4, 0, 0], [1, 2, 3, 0]])
    gen = torch.Generator().manual_seed(0)
    R = region_rate_matrix(brain, states, region_key="sensory", signal_key="concept",
                           width=64, recall=False, T=16, generator=gen)
    assert R.shape == (3, 64)
    assert not torch.allclose(R[0], R[1])   # different states -> different concept rate


def test_region_rate_matrix_zero_fills_bypassed_hippocampus():
    brain = Brain(grid_n=5, seed=0)
    states = torch.tensor([[0, 0, 4, 4], [2, 2, 1, 1]])
    R = region_rate_matrix(brain, states, region_key="hippocampus", signal_key="population",
                           width=150, recall=False, T=16, generator=torch.Generator().manual_seed(0))
    assert R.shape == (2, 150)
    assert torch.count_nonzero(R) == 0   # bypassed -> zero-filled, not a crash


def test_task_targets_shapes():
    states = torch.tensor([[0, 0, 4, 4], [4, 0, 0, 4]])
    t = task_targets(states, grid_n=5)
    assert t["displacement"].shape == (2, 2)
    assert t["optimal_action"].shape == (2, 4)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/training/test_probes.py -k "region_rate_matrix or task_targets" -v`
Expected: FAIL (functions not defined).

- [ ] **Step 3: Implement**

Append to `src/neuromorphic/analysis/probes.py`:

```python
from neuromorphic.training.pretrain import displacement_target

REGION_SIGNALS = {
    "sensory": ("concept", 64),
    "sensory_hidden": ("hidden", 128),
    "prefrontal": ("utility", 4),
    "prefrontal_state": ("state", 100),
    "router": ("gate", 4),
    "motor": ("action", 4),
    "hippocampus": ("population", 150),
}


def region_rate_matrix(brain, states, *, region_key, signal_key, width,
                       recall=False, T=32, generator=None) -> torch.Tensor:
    """[M, width] mean-over-T rate for one region signal; zero-filled if the region is bypassed.

    Uses a single batched brain.step(record=True) so all regions come from the same forward
    pass; the region name is the Brain._regions key (sensory/hippocampus/prefrontal/router/motor),
    NOT a dashboard/output alias. A bypassed region (hippocampus under recall=False) has an empty
    recordings dict -> return zeros instead of index-crashing.
    """
    obs = states if torch.is_tensor(states) else torch.tensor(states)
    out = brain.step(obs, store=False, recall=recall, record=True, generator=generator)
    rec = out["recordings"].get(region_key, {})           # {} for a bypassed region
    train = rec.get(signal_key) if isinstance(rec, dict) else None
    if train is None:
        return torch.zeros(obs.shape[0], width)
    return train.mean(dim=0)                               # [T,M,N] -> [M,N]


def task_targets(states, grid_n) -> dict:
    return {
        "displacement": displacement_target(states, grid_n),
        "optimal_action": optimal_action_targets(states, grid_n),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/training/test_probes.py -v`
Expected: PASS (10 tests). If `brain.step` rejects a batched `[M,4]` obs, that is a real integration issue - fix the call (e.g. ensure obs dtype/shape), do NOT loop per-state silently without noting it in the report.

- [ ] **Step 5: Commit**

```bash
git add src/neuromorphic/analysis/probes.py tests/training/test_probes.py
git commit -m "feat: region_rate_matrix (zero-fills bypassed regions) and task_targets"
```

---

### Task 3: Trained-model checkpoint save/load

**Files:**
- Create: `src/neuromorphic/training/checkpoints.py`
- Modify: `src/neuromorphic/training/generalization.py` (`GenConfig`, `run_generalization`)
- Test: `tests/training/test_checkpoints.py`

**Interfaces:**
- Produces:
  - `save_trained(path, brain, head, cfg_dict) -> None` - torch.save `{sensory_state, head_state, config}`.
  - `load_trained(path, *, grid_n, seed) -> tuple[brain, head]` - rebuild `Brain(grid_n, seed)`, load `sensory` state, rebuild the head via `make_policy_head`, load head state.
  - `GenConfig.checkpoint_path: str | None = None`; when set, `run_generalization` saves after training. Default `None` = byte-identical.

- [ ] **Step 1: Write the failing tests**

Create `tests/training/test_checkpoints.py`:

```python
import torch

from neuromorphic.brain import Brain
from neuromorphic.training.reinforce import make_policy_head, greedy_action
from neuromorphic.training.checkpoints import save_trained, load_trained
from neuromorphic.training.generalization import GenConfig, run_generalization


def test_checkpoint_roundtrip_reproduces_eval(tmp_path):
    brain = Brain(grid_n=5, seed=0)
    head = make_policy_head(brain)
    path = tmp_path / "ckpt.pt"
    save_trained(path, brain, head, {"grid_n": 5, "seed": 0})
    b2, h2 = load_trained(path, grid_n=5, seed=0)
    gen1 = torch.Generator().manual_seed(3)
    gen2 = torch.Generator().manual_seed(3)
    a1 = greedy_action(brain, head, [0, 0, 4, 4], generator=gen1)
    a2 = greedy_action(b2, h2, [0, 0, 4, 4], generator=gen2)
    assert a1 == a2
    # sensory + head weights round-trip exactly
    assert torch.equal(brain.sensory.fc1.weight, b2.sensory.fc1.weight)
    assert torch.equal(head.weight, h2.weight)


def test_genconfig_default_no_checkpoint(tmp_path):
    cfg = GenConfig()
    assert cfg.checkpoint_path is None


def test_run_generalization_writes_checkpoint_when_set(tmp_path):
    ckpt = tmp_path / "run.pt"
    cfg = GenConfig(seed=0, episodes=2, n_heldout=2, max_steps=8,
                    pretrain_sensory=True, pretrain_epochs=5,
                    checkpoint_path=str(ckpt), tag="ck", out_dir=tmp_path)
    run_generalization(cfg)
    assert ckpt.exists()
    b2, h2 = load_trained(ckpt, grid_n=5, seed=0)
    # the saved encoder was pretrained, so it must differ from a fresh random Brain(seed=0)
    fresh = Brain(grid_n=5, seed=0)
    assert not torch.equal(b2.sensory.fc1.weight, fresh.sensory.fc1.weight)
    assert h2.in_features == b2.content   # head reads the 64-d concept
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/training/test_checkpoints.py -v`
Expected: FAIL (module `checkpoints` missing; `GenConfig` has no `checkpoint_path`).

- [ ] **Step 3: Implement**

Create `src/neuromorphic/training/checkpoints.py`:

```python
"""Save/load a trained (sensory encoder, policy head) so downstream analysis reloads them."""

from __future__ import annotations

from pathlib import Path

import torch

from neuromorphic.brain import Brain
from neuromorphic.training.reinforce import make_policy_head


def save_trained(path, brain, head, cfg_dict: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "sensory_state": brain.sensory.state_dict(),
        "head_state": head.state_dict(),
        "config": cfg_dict,
    }, path)


def load_trained(path, *, grid_n, seed):
    ckpt = torch.load(path, weights_only=True)
    brain = Brain(grid_n=grid_n, seed=seed)
    brain.sensory.load_state_dict(ckpt["sensory_state"])
    head_type = ckpt["config"].get("head_type", "linear")
    hidden = ckpt["config"].get("hidden", 128)
    head = make_policy_head(brain, head_type=head_type, hidden=hidden)
    head.load_state_dict(ckpt["head_state"])
    return brain, head
```

In `src/neuromorphic/training/generalization.py`: add `checkpoint_path: str | None = None` to `GenConfig` (near `tag`); import `save_trained`; at the very end of `run_generalization` (after `eval_*` and before/after building `summary`), add:

```python
    if cfg.checkpoint_path is not None:
        from neuromorphic.training.checkpoints import save_trained
        save_trained(cfg.checkpoint_path, brain, head,
                     {"head_type": cfg.head_type, "hidden": cfg.hidden,
                      "seed": cfg.seed, "grid_n": cfg.size})
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/training/test_checkpoints.py tests/training/test_generalization.py -v`
Expected: PASS (new checkpoint tests + all pre-existing generalization tests, incl. the byte-identity determinism test on the no-checkpoint path).

- [ ] **Step 5: Commit**

```bash
git add src/neuromorphic/training/checkpoints.py src/neuromorphic/training/generalization.py tests/training/test_checkpoints.py
git commit -m "feat: trained sensory+head checkpoint save/load"
```

---

### Task 4: Component A - cross-region decodability + geometry (encoder-only)

**Files:**
- Create: `experiments/027_encoder_characterization/__init__.py` (empty), `experiments/027_encoder_characterization/probe.py`
- Test: `tests/training/test_encoder_characterization.py`

**Interfaces:**
- Consumes: `pretrain_sensory`, `enumerate_states`, `split_states` (pretrain); `region_rate_matrix`, `task_targets`, `REGION_SIGNALS`, `ridge_probe`, `peraction_probe`, `shuffle_null`, `pca_reduce`, `participation_ratio`, `unit_importance`, `keepk_curve` (probes).
- Produces: `characterize_seed(seed, *, grid_n=5, pretrain_epochs=200, lam=1e-2, T=32) -> dict` - pre-trains one encoder, then on the encoder's OWN held-out `split_states(seed)` states computes, for each region in `REGION_SIGNALS`: displacement R2 + its shuffle-null hi + PCA-matched (k=4,8) R2, and per-action mean accuracy + shuffle-null; plus concept geometry (participation_ratio, keepk_curve on displacement, fraction-of-units-for-90%-R2). Returns a nested dict keyed by region/target.

- [ ] **Step 1: Write the failing test**

Create `tests/training/test_encoder_characterization.py`:

```python
import importlib.util as ilu
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_spec = ilu.spec_from_file_location(
    "exp027_probe", ROOT / "experiments" / "027_encoder_characterization" / "probe.py")
probe_mod = ilu.module_from_spec(_spec)
_spec.loader.exec_module(probe_mod)


def test_characterize_seed_smoke_and_specialization_direction():
    res = probe_mod.characterize_seed(0, grid_n=5, pretrain_epochs=30, T=8)
    # every region present, hippocampus zero-filled -> displacement R2 ~ 0
    assert "sensory" in res["regions"] and "hippocampus" in res["regions"]
    assert res["regions"]["hippocampus"]["displacement_r2"] < 0.1
    # trained sensory concept beats its own shuffle-null band on displacement
    s = res["regions"]["sensory"]
    assert s["displacement_r2"] > s["displacement_null_hi"]
    # geometry present
    assert 1.0 <= res["geometry"]["participation_ratio"] <= 64.0
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/training/test_encoder_characterization.py -v`
Expected: FAIL (`probe.py` does not exist).

- [ ] **Step 3: Implement `experiments/027_encoder_characterization/probe.py`**

```python
"""EXP-027 Component A: cross-region decodability + concept geometry (encoder-only, per seed)."""

from __future__ import annotations

import torch

from neuromorphic.brain import Brain
from neuromorphic.training.pretrain import pretrain_sensory, enumerate_states, split_states
from neuromorphic.analysis.probes import (
    REGION_SIGNALS, region_rate_matrix, task_targets, ridge_probe, peraction_probe,
    shuffle_null, pca_reduce, participation_ratio, unit_importance, keepk_curve,
)


def characterize_seed(seed, *, grid_n=5, pretrain_epochs=200, lam=1e-2, T=32) -> dict:
    gen = torch.Generator().manual_seed(seed)
    brain = Brain(grid_n=grid_n, seed=seed)
    pretrain_sensory(brain.sensory, grid_n=grid_n, epochs=pretrain_epochs, seed=seed,
                     generator=torch.Generator().manual_seed(seed))
    # probe on the encoder's OWN held-out split (states it never trained on)
    tr, te = split_states(enumerate_states(grid_n), frac_heldout=0.2, seed=seed)
    Ttr, Tte = task_targets(tr, grid_n), task_targets(te, grid_n)

    regions = {}
    for region_key, (signal_key, width) in REGION_SIGNALS.items():
        Xtr = region_rate_matrix(brain, tr, region_key=region_key.split("_")[0],
                                 signal_key=signal_key, width=width, recall=False, T=T, generator=gen)
        Xte = region_rate_matrix(brain, te, region_key=region_key.split("_")[0],
                                 signal_key=signal_key, width=width, recall=False, T=T, generator=gen)
        disp = ridge_probe(Xtr, Ttr["displacement"], Xte, Tte["displacement"], lam=lam)["r2"]
        null = shuffle_null(lambda a, b, c, d: ridge_probe(a, b, c, d, lam=lam)["r2"],
                            Xtr, Ttr["displacement"], Xte, Tte["displacement"], n=10, seed=seed)
        pca = {}
        for k in (4, 8):
            if width > k:
                xtr_k, xte_k = pca_reduce(Xtr, Xte, k)
                pca[k] = ridge_probe(xtr_k, Ttr["displacement"], xte_k, Tte["displacement"], lam=lam)["r2"]
        act = peraction_probe(Xtr, Ttr["optimal_action"], Xte, Tte["optimal_action"])["mean_acc"]
        regions[region_key] = {"displacement_r2": disp, "displacement_null_hi": null["hi"],
                               "displacement_pca": pca, "optimal_action_acc": act}

    # concept geometry (aim 2A)
    Cte = region_rate_matrix(brain, te, region_key="sensory", signal_key="concept",
                             width=64, recall=False, T=T, generator=gen)
    Ctr = region_rate_matrix(brain, tr, region_key="sensory", signal_key="concept",
                             width=64, recall=False, T=T, generator=gen)
    order = unit_importance(Ctr, Ttr["displacement"], lam=lam)
    curve = keepk_curve(Ctr, Ttr["displacement"], Cte, Tte["displacement"],
                        order=order, ks=[1, 2, 4, 8, 16, 32, 64], lam=lam)
    return {"seed": seed, "regions": regions,
            "geometry": {"participation_ratio": participation_ratio(Cte), "keepk": curve}}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/training/test_encoder_characterization.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add experiments/027_encoder_characterization/__init__.py experiments/027_encoder_characterization/probe.py tests/training/test_encoder_characterization.py
git commit -m "feat: EXP-027 Component A cross-region decodability + geometry"
```

---

### Task 5: Component B - MaskedHead + dropout-on-navigation

**Files:**
- Create: `experiments/027_encoder_characterization/dropout_eval.py`
- Test: `tests/training/test_dropout_eval.py`

**Interfaces:**
- Consumes: `load_trained` (Task 3); `evaluate` (generalization); `region_rate_matrix`/`unit_importance`/`task_targets` (probes); `split_states`/`enumerate_states` (pretrain).
- Produces:
  - `class MaskedHead(nn.Module)` wrapping a head; `forward(x)` = `head(x * mask)`; `set_mask(mask)`.
  - `random_mask(width, k, seed) -> torch.Tensor` (k zeros), `importance_mask(order, k, mode) -> torch.Tensor` (zero top-k or bottom-k of `order`).
  - `dropout_curve(brain, head, goals, *, grid_n, ks, n_random=5, size=5, max_steps=100) -> dict` - held-out `evaluate` success for random-k (averaged over draws), top-k, bottom-k masks. The importance `order` is computed from the SAME reloaded `brain`'s concept via `unit_importance` on displacement (so the ranked units are the units being masked).

- [ ] **Step 1: Write the failing tests**

Create `tests/training/test_dropout_eval.py`:

```python
import importlib.util as ilu
from pathlib import Path
import torch

from neuromorphic.brain import Brain
from neuromorphic.training.reinforce import make_policy_head, greedy_action

ROOT = Path(__file__).resolve().parents[2]
_spec = ilu.spec_from_file_location(
    "exp027_dropout", ROOT / "experiments" / "027_encoder_characterization" / "dropout_eval.py")
de = ilu.module_from_spec(_spec)
_spec.loader.exec_module(de)


def test_masked_head_k0_matches_unmasked():
    brain = Brain(grid_n=5, seed=0)
    head = make_policy_head(brain)
    mh = de.MaskedHead(head)
    mh.set_mask(torch.ones(64))
    g1 = torch.Generator().manual_seed(1)
    g2 = torch.Generator().manual_seed(1)
    assert greedy_action(brain, head, [0, 0, 4, 4], generator=g1) == \
           greedy_action(brain, mh, [0, 0, 4, 4], generator=g2)


def test_random_mask_zeros_k_units():
    m = de.random_mask(64, 10, seed=0)
    assert m.shape == (64,)
    assert int((m == 0).sum()) == 10


def test_importance_mask_top_vs_bottom_disjoint():
    order = torch.arange(64)   # 0 most important
    top = de.importance_mask(order, 8, mode="top")
    bot = de.importance_mask(order, 8, mode="bottom")
    assert int((top == 0).sum()) == 8 and int((bot == 0).sum()) == 8
    assert int(((top == 0) & (bot == 0)).sum()) == 0   # disjoint masked sets
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/training/test_dropout_eval.py -v`
Expected: FAIL (`dropout_eval.py` does not exist).

- [ ] **Step 3: Implement `experiments/027_encoder_characterization/dropout_eval.py`**

```python
"""EXP-027 Component B: causal dropout-on-navigation via a MaskedHead on the concept."""

from __future__ import annotations

import torch
import torch.nn as nn

from neuromorphic.training.generalization import evaluate
from neuromorphic.training.pretrain import enumerate_states, split_states
from neuromorphic.analysis.probes import region_rate_matrix, task_targets, unit_importance


class MaskedHead(nn.Module):
    """Wrap a trained head; zero a subset of concept units before it reads them."""

    def __init__(self, head, mask=None):
        super().__init__()
        self.head = head
        self.mask = torch.ones(head.in_features) if mask is None else mask

    def set_mask(self, mask):
        self.mask = mask

    def forward(self, x):
        return self.head(x * self.mask.to(x.dtype))


def random_mask(width, k, seed) -> torch.Tensor:
    m = torch.ones(width)
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(width, generator=g)[:k]
    m[idx] = 0.0
    return m


def importance_mask(order, k, mode) -> torch.Tensor:
    m = torch.ones(len(order))
    idx = order[:k] if mode == "top" else order[len(order) - k:]
    m[idx] = 0.0
    return m


def dropout_curve(brain, head, goals, *, grid_n, ks, n_random=5, size=5, max_steps=100) -> dict:
    """Held-out nav success under random-k / top-k / bottom-k concept masking (importance from THIS brain)."""
    tr, _ = split_states(enumerate_states(grid_n), frac_heldout=0.2, seed=0)
    Xtr = region_rate_matrix(brain, tr, region_key="sensory", signal_key="concept",
                             width=head.in_features, recall=False, T=brain.T,
                             generator=torch.Generator().manual_seed(0))
    order = unit_importance(Xtr, task_targets(tr, grid_n)["displacement"])
    mh = MaskedHead(head)

    def succ():
        return evaluate(brain, mh, goals, size=size, start=(0, 0), max_steps=max_steps,
                        generator=torch.Generator().manual_seed(0)).success_rate

    out = {"random": {}, "top": {}, "bottom": {}}
    for k in ks:
        rs = []
        for j in range(n_random):
            mh.set_mask(random_mask(head.in_features, k, seed=j))
            rs.append(succ())
        out["random"][k] = sum(rs) / len(rs)
        mh.set_mask(importance_mask(order, k, "top")); out["top"][k] = succ()
        mh.set_mask(importance_mask(order, k, "bottom")); out["bottom"][k] = succ()
    return out
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/training/test_dropout_eval.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add experiments/027_encoder_characterization/dropout_eval.py tests/training/test_dropout_eval.py
git commit -m "feat: EXP-027 Component B MaskedHead dropout-on-navigation"
```

---

### Task 6: Driver + aggregation

**Files:**
- Create: `experiments/027_encoder_characterization/run.py`, `experiments/027_encoder_characterization/aggregate.py`
- Test: `tests/training/test_encoder_characterization.py` (add aggregation test)

**Interfaces:**
- Consumes: `characterize_seed` (Task 4); `run_generalization`/`GenConfig` with `checkpoint_path`, `load_trained`, `dropout_curve` (Tasks 3, 5).
- Produces: `aggregate.py` with `aggregate_regions(per_seed) -> dict` (paired win-fraction of sensory-concept vs each region on displacement + optimal-action, sign test, mean/spread across seeds) and `format_matrix(agg) -> str` (markdown region x target decodability matrix). `run.py` `main()` fans out over the 12 seeds: Component A always; Component B mints one checkpoint per seed via `run_generalization(pretrain_sensory=True, checkpoint_path=..., shaping=True)` then `load_trained` + `dropout_curve`. Writes `outputs/027_summary.json`, `027_matrix.md`, `027_dropout.md`.

- [ ] **Step 1: Write the failing test**

Add to `tests/training/test_encoder_characterization.py`:

```python
import importlib.util as ilu2
_aspec = ilu2.spec_from_file_location(
    "exp027_agg", ROOT / "experiments" / "027_encoder_characterization" / "aggregate.py")
agg_mod = ilu2.module_from_spec(_aspec)
_aspec.loader.exec_module(agg_mod)


def test_aggregate_regions_paired_winfraction():
    per_seed = [
        {"regions": {"sensory": {"displacement_r2": 0.8, "optimal_action_acc": 0.9},
                     "motor": {"displacement_r2": 0.2, "optimal_action_acc": 0.5}}},
        {"regions": {"sensory": {"displacement_r2": 0.7, "optimal_action_acc": 0.85},
                     "motor": {"displacement_r2": 0.3, "optimal_action_acc": 0.55}}},
    ]
    agg = agg_mod.aggregate_regions(per_seed)
    assert agg["motor"]["displacement_win_fraction"] == 1.0   # sensory beats motor both seeds
    assert agg["motor"]["n"] == 2
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/training/test_encoder_characterization.py -k aggregate -v`
Expected: FAIL (`aggregate.py` missing).

- [ ] **Step 3: Implement `aggregate.py` then `run.py`**

`experiments/027_encoder_characterization/aggregate.py`:

```python
"""Aggregate EXP-027 per-seed characterization into paired region contrasts vs sensory."""

from __future__ import annotations


def aggregate_regions(per_seed: list) -> dict:
    """For each non-sensory region, paired win-fraction of sensory-concept over it, per target."""
    regions = [r for r in per_seed[0]["regions"] if r != "sensory"]
    out = {}
    for reg in regions:
        d_wins = a_wins = n = 0
        for s in per_seed:
            sen, oth = s["regions"]["sensory"], s["regions"][reg]
            d_wins += int(sen["displacement_r2"] > oth["displacement_r2"])
            a_wins += int(sen["optimal_action_acc"] > oth["optimal_action_acc"])
            n += 1
        out[reg] = {"n": n, "displacement_win_fraction": d_wins / n,
                    "optimal_action_win_fraction": a_wins / n}
    return out
```

`experiments/027_encoder_characterization/run.py`: `ProcessPoolExecutor` over `--seeds` (default 0..11); per seed call `characterize_seed` (Component A); if `--dropout`, `run_generalization(GenConfig(seed=s, pretrain_sensory=True, shaping=True, checkpoint_path=outputs/ck_s.pt, tag=..., out_dir=outputs))` then `load_trained` + `dropout_curve(brain, head, heldout_goals, grid_n=5, ks=[2,4,8,16,32])`. Aggregate with `aggregate_regions`; write `027_summary.json` (per-seed + aggregate + dropout), `027_matrix.md` (region x target win-fractions), `027_dropout.md`. Mirror `experiments/026_sensory_pretrain/run.py`'s pool + `torch.set_num_threads(1)` structure.

- [ ] **Step 4: Run the test + a tiny smoke**

Run: `.venv/Scripts/python.exe -m pytest tests/training/test_encoder_characterization.py -v`
Then: `.venv/Scripts/python.exe experiments/027_encoder_characterization/run.py --seeds 0 1 --pretrain-epochs 20`
Expected: tests PASS; smoke prints per-seed lines + a region x target matrix, writes `outputs/027_summary.json` and `027_matrix.md`. (Component A only unless `--dropout` given.)

- [ ] **Step 5: Commit**

```bash
git add experiments/027_encoder_characterization/run.py experiments/027_encoder_characterization/aggregate.py tests/training/test_encoder_characterization.py
git commit -m "feat: EXP-027 driver and paired region aggregation"
```

---

### Task 7: Run the full characterization and write the verdict (with Mike)

**Files:**
- Create (generated): `outputs/027_summary.json`, `027_matrix.md`, `027_dropout.md`
- Update: the Week-14 obsidian note (Session 2 findings)
- Modify (conditional): `docs/adr/0001-multi-region-training-strategy.md`

- [ ] **Step 1: Run Component A (fast) across the 12 seeds**

Run: `.venv/Scripts/python.exe experiments/027_encoder_characterization/run.py --seeds 0 1 2 3 4 5 6 7 8 9 10 11`
Expected: the region x target decodability matrix with paired win-fractions + the concept geometry; seconds-to-minutes per seed.

- [ ] **Step 2: Run Component B (dropout curve)**

Run: `.venv/Scripts/python.exe experiments/027_encoder_characterization/run.py --seeds 0 1 2 3 4 5 6 7 8 9 10 11 --dropout`
Expected: mints 12 checkpoints (~1h one-time) then the random/top/bottom dropout curves.

- [ ] **Step 3: Read the two verdicts (with Mike)**

Specialization: does the trained sensory concept beat every frozen spectator on both targets, paired, surviving the shuffle-null band and the PCA-matched (k=4,8) control? Report the gradient (concentrated in sensory, attenuated in PFC/router/motor, hippocampus zero-filled/bypassed). Distributedness: do geometry (participation ratio, keep-k, 90%-R2 fraction) and the dropout curve (graceful vs cliff; top-k vs bottom-k gap) AGREE on distributed-vs-brittle?

- [ ] **Step 4: Write up + conditional ADR**

Append a Session-2 findings block to the Week-14 obsidian note with the decodability matrix, geometry, and dropout curves. Amend ADR-0001 only if the specialization + distributedness verdict is decisive.

- [ ] **Step 5: Full suite**

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: all pass.

---

## Self-Review

**Spec coverage:** primitives (probes/geometry) -> Task 1; activity extraction with zero-fill + targets -> Task 2; checkpoint save/load (the new infra) -> Task 3; Component A decodability matrix + controls + geometry -> Task 4; Component B MaskedHead dropout with importance from the reloaded encoder -> Task 5; driver + paired aggregation -> Task 6; run + two-verdict writeup + conditional ADR -> Task 7. Honesty controls (shuffle-null, PCA-matched-k, zero-filled hippocampus, per-action-not-25%) are in Tasks 1-2/4. Leakage fix (probe held-out = encoder pretraining held-out split) is in Task 4 `characterize_seed`. Importance-ordering-matches-masked-encoder fix is in Task 5 `dropout_curve` (order from the same reloaded brain).

**Placeholder scan:** Tasks 1-5 carry full code; Task 6's `run.py` is described against the tested primitives + the EXP-026 driver it mirrors (no new logic, pure orchestration); no TBD/TODO.

**Type consistency:** `region_rate_matrix(brain, states, *, region_key, signal_key, width, recall, T, generator)` identical across Tasks 2/4/5. `ridge_probe(X_tr,Y_tr,X_te,Y_te,lam)` and `unit_importance(X,Y,lam)` consistent Tasks 1/4/5. `MaskedHead(head)`/`set_mask`/`random_mask(width,k,seed)`/`importance_mask(order,k,mode)` consistent Task 5. `save_trained(path,brain,head,cfg_dict)` / `load_trained(path,*,grid_n,seed)` consistent Tasks 3/5/6. `GenConfig.checkpoint_path` consistent Tasks 3/6.
```
