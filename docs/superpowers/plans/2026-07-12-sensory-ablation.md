# EXP-028 Sensory-Code Ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the EXP-028 dose-response ablation: degrade the pre-trained sensory concept (Gaussian noise + structural unit-drop), re-train the linear policy head against the degraded code, and measure how held-out navigation falls with dose.

**Architecture:** A head-wrapper (`AblatedConcept`, mirroring EXP-027's `MaskedHead`) perturbs the concept vector before the trained head reads it, so both training and eval go through the ablated channel with no change to `reinforce.py`. Two default-off hooks on `GenConfig` (`ablation`, `load_encoder_path`) let the existing generalization harness (a) load a cached frozen encoder instead of re-pretraining and (b) wrap the head with an ablation. A driver mints 12 pretrained encoders once, then sweeps (operator, dose, seed) re-training only the cheap linear head.

**Tech Stack:** Python, PyTorch, snntorch (already installed). Reuses `neuromorphic.analysis.probes` (`region_rate_matrix`, `unit_importance`, `task_targets`), `neuromorphic.training.generalization`, `neuromorphic.training.checkpoints`.

## Global Constraints

- Run Python via `.venv/Scripts/python.exe` (Windows venv).
- Plain commit messages; NO `Co-Authored-By` trailer; NO em-dashes in messages.
- New `GenConfig` hooks MUST be default-off and byte-identical when unused (same discipline as `pretrain_sensory`, `entropy_beta`).
- `GenConfig` must stay JSON-serializable (`run_generalization` does `json.dumps(asdict(cfg)...)`), so no tensors/callables in config fields. Importance ordering passes as `list[int]`.
- 12 seeds for any real sweep (EXP-026 lesson: n=5 lies).
- Base branch: `week14-encoder-characterization` at `809ba8f`. Experiment outputs are gitignored.
- Mirror EXP-026/027 structure and naming; do not restructure existing modules.

---

### Task 1: `AblatedConcept` wrapper + `AblationSpec`

**Files:**
- Create: `src/neuromorphic/analysis/ablate.py`
- Test: `tests/training/test_ablate.py`

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces:
  - `AblationSpec(kind: str, dose: float, mode: str = "random", seed: int = 0)` dataclass. `kind` in `{"gaussian","unitdrop"}`; `mode` in `{"random","top"}` (unitdrop only).
  - `AblatedConcept(head: nn.Module, spec: AblationSpec | None, *, width: int, order: list[int] | None = None)` — `nn.Module` wrapping `head`; `forward(x)` perturbs `x` (shape `[width]`) then returns `head(x)`. Identity (returns `head(x)` unchanged) when `spec is None` or `spec.dose == 0`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/training/test_ablate.py
import torch
import torch.nn as nn
from neuromorphic.analysis.ablate import AblationSpec, AblatedConcept


def _head(width=8, actions=4):
    torch.manual_seed(0)
    return nn.Linear(width, actions)


def test_dose_zero_is_identity_gaussian():
    head = _head()
    x = torch.randn(8)
    wrapped = AblatedConcept(head, AblationSpec("gaussian", dose=0.0), width=8)
    assert torch.equal(wrapped(x), head(x))


def test_none_spec_is_identity():
    head = _head()
    x = torch.randn(8)
    wrapped = AblatedConcept(head, None, width=8)
    assert torch.equal(wrapped(x), head(x))


def test_gaussian_perturbs_input():
    head = _head()
    x = torch.zeros(8)
    wrapped = AblatedConcept(head, AblationSpec("gaussian", dose=0.5, seed=1), width=8)
    # With a nonzero dose the effective input differs from the clean one, so output moves.
    assert not torch.equal(wrapped(x), head(x))


def test_unitdrop_random_zeros_expected_count():
    head = nn.Identity()
    x = torch.ones(8)
    wrapped = AblatedConcept(head, AblationSpec("unitdrop", dose=0.25, mode="random", seed=3), width=8)
    out = wrapped(x)
    assert int((out == 0).sum()) == 2  # round(0.25 * 8)


def test_unitdrop_top_drops_most_important_units():
    head = nn.Identity()
    x = torch.ones(8)
    order = [7, 6, 5, 4, 3, 2, 1, 0]  # most-important-first
    wrapped = AblatedConcept(head, AblationSpec("unitdrop", dose=0.25, mode="top"), width=8, order=order)
    out = wrapped(x)
    zeroed = set(int(i) for i in torch.nonzero(out == 0).flatten())
    assert zeroed == {7, 6}


def test_gaussian_is_reproducible_for_fixed_seed():
    head = nn.Identity()
    x = torch.zeros(8)
    a = AblatedConcept(head, AblationSpec("gaussian", dose=0.5, seed=5), width=8)(x)
    b = AblatedConcept(head, AblationSpec("gaussian", dose=0.5, seed=5), width=8)(x)
    assert torch.equal(a, b)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/training/test_ablate.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'neuromorphic.analysis.ablate'`

- [ ] **Step 3: Write the implementation**

```python
# src/neuromorphic/analysis/ablate.py
"""EXP-028: degrade the sensory concept before the policy head reads it.

Two operators: additive Gaussian noise (continuous fidelity dose) and structural
unit-drop (zero a fraction of concept units; random or most-important-first).
Wrapping the head means both training and eval flow through the ablated channel
with no change to reinforce.py (mirrors EXP-027's MaskedHead).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class AblationSpec:
    """A single dose of concept corruption. JSON-safe (scalars only)."""

    kind: str          # "gaussian" | "unitdrop"
    dose: float        # gaussian: sigma; unitdrop: fraction of units to zero
    mode: str = "random"  # unitdrop only: "random" | "top"
    seed: int = 0      # reproducible noise / random mask


class AblatedConcept(nn.Module):
    """Wrap a policy ``head``; perturb the concept vector before ``head`` reads it."""

    def __init__(self, head, spec: AblationSpec | None, *, width: int, order=None):
        super().__init__()
        self.head = head
        self.spec = spec
        self.width = width
        self._gen = None
        self._mask = None
        if spec is not None and spec.dose > 0 and spec.kind == "unitdrop":
            k = round(spec.dose * width)
            mask = torch.ones(width)
            if spec.mode == "top":
                if order is None:
                    raise ValueError("unitdrop mode='top' requires an importance order")
                idx = torch.tensor(list(order)[:k], dtype=torch.long)
            else:
                g = torch.Generator().manual_seed(spec.seed)
                idx = torch.randperm(width, generator=g)[:k]
            mask[idx] = 0.0
            self._mask = mask
        if spec is not None and spec.dose > 0 and spec.kind == "gaussian":
            self._gen = torch.Generator().manual_seed(spec.seed)

    def forward(self, x):
        spec = self.spec
        if spec is None or spec.dose == 0:
            return self.head(x)
        if spec.kind == "gaussian":
            x = x + spec.dose * torch.randn(x.shape, generator=self._gen)
        elif spec.kind == "unitdrop":
            x = x * self._mask.to(x.dtype)
        else:
            raise ValueError(f"unknown ablation kind {spec.kind!r}")
        return self.head(x)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/training/test_ablate.py -q`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add src/neuromorphic/analysis/ablate.py tests/training/test_ablate.py
git commit -m "feat: EXP-028 AblatedConcept wrapper (gaussian + unit-drop)"
```

---

### Task 2: default-off `ablation` + `load_encoder_path` hooks on the generalization harness

**Files:**
- Modify: `src/neuromorphic/training/generalization.py` (`GenConfig` dataclass; `run_generalization`)
- Test: `tests/training/test_ablation_hook.py`

**Interfaces:**
- Consumes: `AblationSpec`, `AblatedConcept` from Task 1.
- Produces: `GenConfig` gains three default-off fields:
  - `ablation: AblationSpec | None = None`
  - `ablation_order: list[int] | None = None` (importance ranking for `unitdrop` `mode="top"`)
  - `load_encoder_path: str | None = None` (load a cached sensory encoder instead of pretraining)
  - `run_generalization` behavior unchanged when all three are unset; `summary["config"]` stays JSON-serializable.

- [ ] **Step 1: Write the failing tests**

```python
# tests/training/test_ablation_hook.py
import json
from dataclasses import asdict

import torch

from neuromorphic.analysis.ablate import AblationSpec
from neuromorphic.training.generalization import GenConfig, run_generalization


def _cfg(tmp_path, **kw):
    return GenConfig(seed=0, episodes=5, n_heldout=4, size=5,
                     out_dir=tmp_path, tag="t", **kw)


def test_default_off_is_json_serializable_and_unwrapped(tmp_path):
    summary = run_generalization(_cfg(tmp_path))
    # summary must round-trip through JSON (no tensors/callables leaked into config)
    json.loads(json.dumps(summary["config"]))
    assert summary["config"]["ablation"] is None


def test_ablation_config_serializes(tmp_path):
    spec = AblationSpec("gaussian", dose=0.2, seed=0)
    summary = run_generalization(_cfg(tmp_path, ablation=spec))
    dumped = json.loads(json.dumps(summary["config"]))
    assert dumped["ablation"]["kind"] == "gaussian"
    assert dumped["ablation"]["dose"] == 0.2


def test_load_encoder_path_skips_pretrain(tmp_path):
    # mint an encoder checkpoint via the existing pretrain+save path
    ck = str(tmp_path / "enc.pt")
    run_generalization(_cfg(tmp_path, pretrain_sensory=True, checkpoint_path=ck))
    # loading it must skip pretraining (pretrain info stays None) and still run
    summary = run_generalization(_cfg(tmp_path, load_encoder_path=ck))
    assert summary["pretrain"] is None
    assert "heldout" in summary["eval"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/training/test_ablation_hook.py -q`
Expected: FAIL (`TypeError: __init__() got an unexpected keyword argument 'ablation'`)

- [ ] **Step 3: Add the config fields**

In `src/neuromorphic/training/generalization.py`, add an import near the top (after the existing `from neuromorphic.training.pretrain import pretrain_sensory` line):

```python
from neuromorphic.analysis.ablate import AblatedConcept, AblationSpec
```

Then in the `GenConfig` dataclass, after the `checkpoint_path: str | None = None` line, add:

```python
    ablation: AblationSpec | None = None
    ablation_order: "list[int] | None" = None
    load_encoder_path: str | None = None
```

- [ ] **Step 4: Wire the encoder-load and ablation hooks into `run_generalization`**

Replace the encoder setup block:

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

with:

```python
    brain = Brain(grid_n=cfg.size, seed=cfg.seed)
    pretrain_info = None
    if cfg.load_encoder_path is not None:
        ckpt = torch.load(cfg.load_encoder_path, weights_only=True)
        brain.sensory.load_state_dict(ckpt["sensory_state"])
    elif cfg.pretrain_sensory:
        pretrain_info = pretrain_sensory(
            brain.sensory, grid_n=cfg.size, epochs=cfg.pretrain_epochs, lr=cfg.pretrain_lr,
            seed=cfg.seed, generator=torch.Generator().manual_seed(cfg.seed),
        )
    head = make_policy_head(brain, head_type=cfg.head_type, hidden=cfg.hidden)
    if cfg.ablation is not None:
        head = AblatedConcept(head, cfg.ablation, width=brain.content, order=cfg.ablation_order)
```

- [ ] **Step 5: Keep the summary JSON-serializable**

`asdict(cfg)` recurses `AblationSpec` into a plain dict already, and `ablation_order` is a `list[int]`, so no tensors leak. No further change needed. Verify by reading the existing `summary = {...}` block still uses `asdict(cfg)`.

- [ ] **Step 6: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/training/test_ablation_hook.py -q`
Expected: PASS (3 passed)

- [ ] **Step 7: Regression-check the wider training suite**

Run: `.venv/Scripts/python.exe -m pytest tests/training -q`
Expected: PASS (all previously-passing tests still green)

- [ ] **Step 8: Commit**

```bash
git add src/neuromorphic/training/generalization.py tests/training/test_ablation_hook.py
git commit -m "feat: EXP-028 default-off ablation and cached-encoder hooks on GenConfig"
```

---

### Task 3: dose-curve aggregation

**Files:**
- Create: `experiments/028_sensory_ablation/__init__.py` (empty)
- Create: `experiments/028_sensory_ablation/aggregate.py`
- Test: `tests/training/test_ablation_aggregate.py`

**Interfaces:**
- Consumes: nothing from other tasks (pure functions over plain dicts).
- Produces:
  - `aggregate_curve(cells: list[dict]) -> dict` — each cell is `{"operator": str, "dose": float, "seed": int, "heldout_success": float}` where `operator` is one of `"gaussian"`, `"unitdrop_random"`, `"unitdrop_top"`. Returns `{operator: {dose: mean_success_across_seeds}}`.
  - `format_curve(curve: dict) -> str` — markdown, one column per operator, one row per dose (union of doses, ascending).

- [ ] **Step 1: Write the failing tests**

```python
# tests/training/test_ablation_aggregate.py
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parents[2] / "experiments" / "028_sensory_ablation"
spec = importlib.util.spec_from_file_location("exp028_agg", HERE / "aggregate.py")
agg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agg)


def test_aggregate_curve_means_across_seeds():
    cells = [
        {"operator": "gaussian", "dose": 0.0, "seed": 0, "heldout_success": 0.4},
        {"operator": "gaussian", "dose": 0.0, "seed": 1, "heldout_success": 0.6},
        {"operator": "gaussian", "dose": 0.2, "seed": 0, "heldout_success": 0.2},
    ]
    curve = agg.aggregate_curve(cells)
    assert curve["gaussian"][0.0] == 0.5
    assert curve["gaussian"][0.2] == 0.2


def test_format_curve_has_row_per_dose_and_operator_columns():
    curve = {"gaussian": {0.0: 0.5, 0.2: 0.2}, "unitdrop_top": {0.0: 0.5}}
    md = agg.format_curve(curve)
    assert "gaussian" in md and "unitdrop_top" in md
    assert "| 0.0 |" in md and "| 0.2 |" in md
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/training/test_ablation_aggregate.py -q`
Expected: FAIL (`FileNotFoundError` / module load error for `aggregate.py`)

- [ ] **Step 3: Write the implementation**

```python
# experiments/028_sensory_ablation/__init__.py  (empty file)
```

```python
# experiments/028_sensory_ablation/aggregate.py
"""Aggregate EXP-028 ablation cells into per-operator dose-response curves."""

from __future__ import annotations


def aggregate_curve(cells: list[dict]) -> dict:
    """cells -> {operator: {dose: mean heldout success across seeds}}."""
    buckets: dict = {}
    for c in cells:
        buckets.setdefault(c["operator"], {}).setdefault(c["dose"], []).append(
            c["heldout_success"]
        )
    return {
        op: {dose: sum(v) / len(v) for dose, v in sorted(doses.items())}
        for op, doses in buckets.items()
    }


def format_curve(curve: dict) -> str:
    """Markdown table: one row per dose (ascending union), one column per operator."""
    operators = sorted(curve)
    doses = sorted({d for op in operators for d in curve[op]})
    header = "| dose | " + " | ".join(operators) + " |"
    sep = "| --- | " + " | ".join("---" for _ in operators) + " |"
    rows = []
    for dose in doses:
        cells = []
        for op in operators:
            v = curve[op].get(dose)
            cells.append("-" if v is None else f"{v:.0%}")
        rows.append(f"| {dose} | " + " | ".join(cells) + " |")
    return "\n".join([header, sep, *rows]) + "\n"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/training/test_ablation_aggregate.py -q`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add experiments/028_sensory_ablation/__init__.py experiments/028_sensory_ablation/aggregate.py tests/training/test_ablation_aggregate.py
git commit -m "feat: EXP-028 dose-curve aggregation"
```

---

### Task 4: driver (mint encoders once, sweep operator x dose x seed)

**Files:**
- Create: `experiments/028_sensory_ablation/run.py`
- Test: `tests/training/test_ablation_run_smoke.py`

**Interfaces:**
- Consumes: `aggregate_curve`/`format_curve` (Task 3); `GenConfig`/`run_generalization` with the Task 2 hooks; `AblationSpec` (Task 1); `region_rate_matrix`, `unit_importance`, `task_targets` from `neuromorphic.analysis.probes`; `enumerate_states`, `split_states` from `neuromorphic.training.pretrain`.
- Produces:
  - `mint_encoder(seed, *, grid_n, out_dir, episodes, pretrain_epochs) -> str` — pretrains+saves one encoder checkpoint, returns its path.
  - `importance_order(ckpt_path, *, grid_n) -> list[int]` — most-important-concept-units-first ranking for that encoder (for `unitdrop` `mode="top"`).
  - `run_cell(seed, operator, dose, *, ckpt_path, order, grid_n, episodes) -> dict` — one ablation cell, returns `{"operator","dose","seed","heldout_success"}`.
  - `main()` — CLI: `--seeds`, `--grid-n`, `--episodes`, `--pretrain-epochs`, `--workers`; writes `outputs/028_curve.md` + `outputs/028_summary.json`.

- [ ] **Step 1: Write the failing smoke test**

```python
# tests/training/test_ablation_run_smoke.py
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parents[2] / "experiments" / "028_sensory_ablation"
spec = importlib.util.spec_from_file_location("exp028_run", HERE / "run.py")
run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run)


def test_mint_and_cell_smoke(tmp_path):
    # tiny config: mint one encoder, take its importance order, run one gaussian cell
    ck = run.mint_encoder(0, grid_n=5, out_dir=tmp_path, episodes=3, pretrain_epochs=2)
    assert Path(ck).exists()
    order = run.importance_order(ck, grid_n=5)
    assert len(order) == 64 and sorted(order) == list(range(64))
    cell = run.run_cell(0, "gaussian", 0.0, ckpt_path=ck, order=order, grid_n=5, episodes=3)
    assert cell["operator"] == "gaussian" and cell["dose"] == 0.0
    assert 0.0 <= cell["heldout_success"] <= 1.0
```

- [ ] **Step 2: Run the smoke test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/training/test_ablation_run_smoke.py -q`
Expected: FAIL (module load error / missing `mint_encoder`)

- [ ] **Step 3: Write the driver**

```python
# experiments/028_sensory_ablation/run.py
"""EXP-028 driver: sensory-code ablation dose-response.

Mint 12 pretrained+frozen sensory encoders once, then for each (operator, dose, seed)
reload the cached encoder and re-train ONLY the linear policy head against the ablated
concept, measuring held-out navigation success. Operators: gaussian noise, unit-drop
random, unit-drop top-k (most-important-first, importance from that encoder).

Run (repo root, venv active):
    .venv/Scripts/python.exe experiments/028_sensory_ablation/run.py --seeds 0 1 2 3 4 5 6 7 8 9 10 11
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import torch

from neuromorphic.analysis.ablate import AblationSpec
from neuromorphic.analysis.probes import region_rate_matrix, task_targets, unit_importance
from neuromorphic.training.generalization import GenConfig, run_generalization
from neuromorphic.training.checkpoints import load_trained
from neuromorphic.training.pretrain import enumerate_states, split_states

torch.set_num_threads(1)

HERE = Path(__file__).resolve().parent

_agg_spec = importlib.util.spec_from_file_location("exp028_agg", HERE / "aggregate.py")
aggregate_mod = importlib.util.module_from_spec(_agg_spec)
_agg_spec.loader.exec_module(aggregate_mod)

GAUSS_DOSES = [0.0, 0.05, 0.1, 0.2, 0.4, 0.8]
DROP_DOSES = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9]


def mint_encoder(seed, *, grid_n, out_dir, episodes, pretrain_epochs) -> str:
    torch.set_num_threads(1)
    out_dir = Path(out_dir)
    ck = out_dir / f"028_enc_{seed}.pt"
    cfg = GenConfig(
        seed=seed, episodes=episodes, shaping=True, pretrain_sensory=True,
        pretrain_epochs=pretrain_epochs, checkpoint_path=str(ck), size=grid_n,
        tag=f"exp028_mint_seed{seed}", out_dir=out_dir,
    )
    run_generalization(cfg)
    return str(ck)


def importance_order(ckpt_path, *, grid_n) -> list[int]:
    """Most-important concept-unit ranking for the cached encoder (train states)."""
    brain, _ = load_trained(ckpt_path, grid_n=grid_n)
    tr, _ = split_states(enumerate_states(grid_n), frac_heldout=0.2, seed=0)
    X = region_rate_matrix(brain, tr, region_key="sensory", signal_key="concept",
                           width=brain.content, recall=False, T=brain.T,
                           generator=torch.Generator().manual_seed(0))
    order = unit_importance(X, task_targets(tr, grid_n)["displacement"])
    return [int(i) for i in order]


def run_cell(seed, operator, dose, *, ckpt_path, order, grid_n, episodes) -> dict:
    torch.set_num_threads(1)
    if operator == "gaussian":
        spec, ab_order = AblationSpec("gaussian", dose=dose, seed=seed), None
    elif operator == "unitdrop_random":
        spec, ab_order = AblationSpec("unitdrop", dose=dose, mode="random", seed=seed), None
    elif operator == "unitdrop_top":
        spec, ab_order = AblationSpec("unitdrop", dose=dose, mode="top", seed=seed), order
    else:
        raise ValueError(f"unknown operator {operator!r}")
    cfg = GenConfig(
        seed=seed, episodes=episodes, shaping=True, size=grid_n,
        load_encoder_path=ckpt_path, ablation=spec, ablation_order=ab_order,
        tag=f"exp028_{operator}_d{dose}_s{seed}", out_dir=Path(ckpt_path).parent,
    )
    summary = run_generalization(cfg)
    return {"operator": operator, "dose": dose, "seed": seed,
            "heldout_success": summary["eval"]["heldout"]["success_rate"]}


def _cells_for_seed(seed):
    out = []
    for d in GAUSS_DOSES:
        out.append(("gaussian", d, seed))
    for d in DROP_DOSES:
        out.append(("unitdrop_random", d, seed))
        out.append(("unitdrop_top", d, seed))
    return out


def parse_args():
    p = argparse.ArgumentParser(description="EXP-028 sensory-code ablation")
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(12)))
    p.add_argument("--grid-n", type=int, default=5)
    p.add_argument("--episodes", type=int, default=600)
    p.add_argument("--pretrain-epochs", type=int, default=200)
    p.add_argument("--workers", type=int, default=6)
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = HERE / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    workers = max(1, min(args.workers, len(args.seeds)))

    print(f"Minting {len(args.seeds)} encoders across {workers} workers ...", flush=True)
    ckpts, orders = {}, {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(mint_encoder, s, grid_n=args.grid_n, out_dir=out_dir,
                          episodes=args.episodes, pretrain_epochs=args.pretrain_epochs): s
                for s in args.seeds}
        for fut in as_completed(futs):
            s = futs[fut]
            ckpts[s] = fut.result()
            print(f"  minted encoder seed {s}", flush=True)
    for s in args.seeds:
        orders[s] = importance_order(ckpts[s], grid_n=args.grid_n)

    jobs = [job for s in args.seeds for job in _cells_for_seed(s)]
    print(f"Sweeping {len(jobs)} ablation cells across {workers} workers ...", flush=True)
    cells = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(run_cell, s, op, d, ckpt_path=ckpts[s], order=orders[s],
                          grid_n=args.grid_n, episodes=args.episodes): (op, d, s)
                for (op, d, s) in jobs}
        for done, fut in enumerate(as_completed(futs), 1):
            cells.append(fut.result())
            if done % 12 == 0 or done == len(jobs):
                print(f"  [{done}/{len(jobs)}] cells done", flush=True)

    curve = aggregate_mod.aggregate_curve(cells)
    curve_md = aggregate_mod.format_curve(curve)
    (out_dir / "028_curve.md").write_text(curve_md)
    (out_dir / "028_summary.json").write_text(json.dumps(
        {"config": {"seeds": args.seeds, "grid_n": args.grid_n, "episodes": args.episodes},
         "cells": cells, "curve": curve}, indent=2))
    print("\n=== ablation dose-response (mean held-out success across seeds) ===\n" + curve_md,
          flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the smoke test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/training/test_ablation_run_smoke.py -q`
Expected: PASS (1 passed)

- [ ] **Step 5: Full training-suite regression**

Run: `.venv/Scripts/python.exe -m pytest tests/training -q`
Expected: PASS (all green)

- [ ] **Step 6: Commit**

```bash
git add experiments/028_sensory_ablation/run.py tests/training/test_ablation_run_smoke.py
git commit -m "feat: EXP-028 driver (mint encoders once, sweep operator x dose x seed)"
```

---

## Notes for the runner (post-build)

- The real sweep is `12 encoders + (6 gaussian + 6x2 unitdrop) x 12 = 216 head-trainings`. Multi-hour; run on the laptop over SSH (`ssh mlgbr@192.168.50.62`), tee to `outputs/028_run.log`, tune `--workers` to spare the gaming CPU.
- `load_trained(ckpt_path, grid_n=...)` now defaults `seed` from the checkpoint config (Task-adjacent fix already on the branch), so `importance_order` does not need to pass `seed`.
- After results land: fill the EXP-028 writeup, decide ADR-0001 Amendment 5, update `project-exp027-build-state` / add an EXP-028 memory.
