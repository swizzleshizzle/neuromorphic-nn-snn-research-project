# Entropy Bonus Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an optional entropy bonus to the REINFORCE objective so the higher-capacity MLP head stops collapsing to a zero-entropy one-hot policy, unblocking a fair EXP-025 re-run.

**Architecture:** Add `entropy_beta` (default 0.0) to `train_episode` and subtract `beta * sum_t H(pi_t)` from the loss; the default path stays byte-identical. Thread `entropy_beta` through `GenConfig` and the EXP-025 runner. Re-run the sweep with `entropy_beta=0.01`.

**Tech Stack:** Python, PyTorch (`torch.distributions.Categorical` entropy), pytest.

## Global Constraints

- Commit messages: plain, imperative, no `Co-Authored-By` / AI trailers, no em-dashes in prose or docs.
- Run Python via the repo venv: `.venv/Scripts/python.exe` (Windows). Tests: `.venv/Scripts/python.exe -m pytest <paths> -v`.
- The `entropy_beta=0.0` default path must stay byte-identical to today (preserves EXP-023/024). Achieve this by leaving the existing loss statement untouched and only adding the bonus under `if entropy_beta:`.
- The brain stays frozen (`recall=False`, `no_grad`); no encoder changes.
- Concept/action dims come from the brain, never hardcoded.
- Keep the existing suite green (214 tests as of branch `week13-head-capacity`).

---

### Task 1: entropy_beta in train_episode

**Files:**
- Modify: `src/neuromorphic/training/reinforce.py` (`train_episode`)
- Test: `tests/training/test_reinforce.py`

**Interfaces:**
- Produces: `train_episode(..., entropy_beta: float = 0.0)`. Loss becomes
  `policy_loss - entropy_beta * torch.stack(entropies).sum()`; with `entropy_beta=0.0`
  the executed loss statement is unchanged. Returned stats dict is unchanged.

- [ ] **Step 1: Write the failing tests**

Add to `tests/training/test_reinforce.py`:

```python
def test_train_episode_entropy_beta_default_matches_no_bonus():
    """entropy_beta=0.0 must reproduce the exact loss (EXP-023/024 byte-identity)."""
    def run(beta):
        brain = Brain(grid_n=5, seed=0)
        head = make_policy_head(brain)
        env = GridWorldEnv(max_steps=10)
        opt = torch.optim.Adam(policy_parameters(head), lr=1e-2)
        return train_episode(
            brain, head, env, opt, gamma=0.99, baseline=0.0,
            generator=torch.Generator().manual_seed(0), max_steps=10,
            entropy_beta=beta,
        )["loss"]
    assert run(0.0) == run(0.0)  # deterministic
    # explicit default path == explicitly passing 0.0
    brain = Brain(grid_n=5, seed=0)
    head = make_policy_head(brain)
    env = GridWorldEnv(max_steps=10)
    opt = torch.optim.Adam(policy_parameters(head), lr=1e-2)
    loss_default = train_episode(
        brain, head, env, opt, generator=torch.Generator().manual_seed(0), max_steps=10,
    )["loss"]
    assert loss_default == run(0.0)


def test_train_episode_entropy_bonus_lowers_loss():
    """With entropy > 0, a positive beta subtracts a positive term -> lower loss."""
    def loss_for(beta):
        brain = Brain(grid_n=5, seed=0)
        head = make_policy_head(brain)
        env = GridWorldEnv(max_steps=10)
        opt = torch.optim.Adam(policy_parameters(head), lr=1e-2)
        return train_episode(
            brain, head, env, opt, gamma=0.99, baseline=0.0,
            generator=torch.Generator().manual_seed(0), max_steps=10,
            entropy_beta=beta,
        )
    base = loss_for(0.0)
    bonus = loss_for(0.5)
    assert base["mean_entropy"] > 0.0
    assert bonus["loss"] < base["loss"]


def test_train_episode_with_bonus_updates_head():
    brain = Brain(grid_n=5, seed=0)
    head = make_policy_head(brain, head_type="mlp", hidden=128)
    env = GridWorldEnv(max_steps=10)
    opt = torch.optim.Adam(policy_parameters(head), lr=1e-2)
    before = [p.detach().clone() for p in head.parameters()]
    train_episode(
        brain, head, env, opt, generator=torch.Generator().manual_seed(0),
        max_steps=10, entropy_beta=0.01,
    )
    assert any(not torch.equal(b, a.detach()) for b, a in zip(before, head.parameters()))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/training/test_reinforce.py -k "entropy_beta or entropy_bonus or with_bonus" -v`
Expected: FAIL (`train_episode` has no `entropy_beta` kwarg -> TypeError).

- [ ] **Step 3: Implement**

In `src/neuromorphic/training/reinforce.py`, add the keyword-only param to `train_episode`'s signature (after `max_steps`):

```python
    max_steps: int | None = None,
    entropy_beta: float = 0.0,
) -> dict:
```

Then, leaving the existing loss line exactly as-is, add the bonus immediately after it:

```python
    loss = -(torch.stack(log_probs) * advantages).sum()
    if entropy_beta:
        loss = loss - entropy_beta * torch.stack(entropies).sum()
```

(Do not alter the `entropies` collection or the `mean_entropy` stat.) Update the
`train_episode` docstring with one line noting the optional `-beta*sum_t H` entropy bonus.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/training/test_reinforce.py -v`
Expected: PASS (new + all pre-existing reinforce tests).

- [ ] **Step 5: Commit**

```bash
git add src/neuromorphic/training/reinforce.py tests/training/test_reinforce.py
git commit -m "feat: optional entropy bonus in REINFORCE train_episode"
```

---

### Task 2: entropy_beta through the generalization harness

**Files:**
- Modify: `src/neuromorphic/training/generalization.py` (`GenConfig`, `run_generalization`)
- Test: `tests/training/test_generalization.py`

**Interfaces:**
- Consumes: `train_episode(..., entropy_beta=...)` from Task 1.
- Produces: `GenConfig.entropy_beta: float = 0.0`, passed to `train_episode` in the loop;
  recorded in the summary `config` via `asdict`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/training/test_generalization.py`:

```python
def test_genconfig_defaults_entropy_beta_zero():
    assert GenConfig().entropy_beta == 0.0


def test_run_generalization_records_entropy_beta(tmp_path):
    cfg = GenConfig(
        seed=0, episodes=3, n_heldout=2, max_steps=8,
        entropy_beta=0.01, tag="smoke_beta", out_dir=tmp_path,
    )
    summary = run_generalization(cfg)
    assert summary["config"]["entropy_beta"] == 0.01
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/training/test_generalization.py -k "entropy_beta" -v`
Expected: FAIL (`GenConfig` has no `entropy_beta` field).

- [ ] **Step 3: Implement**

In `src/neuromorphic/training/generalization.py`, add to `GenConfig` (next to `head_type` / `hidden`):

```python
    entropy_beta: float = 0.0
```

In `run_generalization`, add `entropy_beta=cfg.entropy_beta` to the `train_episode` call:

```python
        stats = train_episode(
            brain, head, env, opt, gamma=cfg.gamma, baseline=baseline,
            generator=gen, max_steps=cfg.max_steps, entropy_beta=cfg.entropy_beta,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/training/test_generalization.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/neuromorphic/training/generalization.py tests/training/test_generalization.py
git commit -m "feat: entropy_beta option in generalization harness"
```

---

### Task 3: entropy_beta plumbing in the EXP-025 runner

**Files:**
- Modify: `experiments/025_head_capacity/run.py` (`build_configs`, `parse_args`, `main`)
- Test: `tests/training/test_head_capacity_aggregate.py` (add a build_configs test)

**Interfaces:**
- Consumes: `GenConfig.entropy_beta` (Task 2).
- Produces: `build_configs(seeds, episodes, out_dir, entropy_beta=0.0)`; when
  `entropy_beta > 0` the per-run tag gains a suffix (e.g. `_b01`) so re-run outputs do not
  overwrite the zero-beta EXP-025 baseline. A `--entropy-beta` CLI arg (default 0.01)
  drives the re-run.

- [ ] **Step 1: Write the failing test**

Add to `tests/training/test_head_capacity_aggregate.py` (reuse the `importlib` loader
pattern already at the top of the file to load `run.py`; if only `aggregate.py` is loaded
there, add a sibling loader for `run.py`):

```python
import importlib.util as _ilu
from pathlib import Path as _P
_run_spec = _ilu.spec_from_file_location(
    "exp025_run", _P(__file__).resolve().parents[2] / "experiments" / "025_head_capacity" / "run.py"
)
run_mod = _ilu.module_from_spec(_run_spec)
_run_spec.loader.exec_module(run_mod)


def test_build_configs_sets_entropy_beta_and_tag_suffix(tmp_path):
    cfgs = run_mod.build_configs([0], episodes=5, out_dir=tmp_path, entropy_beta=0.01)
    assert len(cfgs) == 4  # 2 heads x 2 regimes x 1 seed
    assert all(c.entropy_beta == 0.01 for c in cfgs)
    assert all(c.tag.endswith("_b01") for c in cfgs)


def test_build_configs_zero_beta_has_no_suffix(tmp_path):
    cfgs = run_mod.build_configs([0], episodes=5, out_dir=tmp_path, entropy_beta=0.0)
    assert all(not c.tag.endswith("_b01") for c in cfgs)
    assert all(c.entropy_beta == 0.0 for c in cfgs)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/training/test_head_capacity_aggregate.py -k "build_configs" -v`
Expected: FAIL (`build_configs` takes no `entropy_beta`).

- [ ] **Step 3: Implement**

In `experiments/025_head_capacity/run.py`, update `build_configs`:

```python
def build_configs(seeds, episodes, out_dir, entropy_beta=0.0):
    """The 2 heads x 2 regimes x seeds sweep, tagged uniquely; _b01 suffix when beta>0."""
    suffix = "_b01" if entropy_beta else ""
    configs = []
    for head_type in ("linear", "mlp"):
        for shaping in (True, False):
            regime = "shaped" if shaping else "sparse"
            for seed in seeds:
                configs.append(GenConfig(
                    seed=seed, episodes=episodes, shaping=shaping,
                    head_type=head_type, hidden=128, entropy_beta=entropy_beta,
                    tag=f"{regime}_{head_type}_seed{seed}{suffix}", out_dir=out_dir,
                ))
    return configs
```

Add to `parse_args`:

```python
    p.add_argument("--entropy-beta", type=float, default=0.01,
                   help="entropy-bonus coefficient for the re-run (0 disables)")
```

In `main`, pass it through:

```python
    configs = build_configs(args.seeds, args.episodes, out_dir, entropy_beta=args.entropy_beta)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/training/test_head_capacity_aggregate.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add experiments/025_head_capacity/run.py tests/training/test_head_capacity_aggregate.py
git commit -m "feat: entropy_beta plumbing in EXP-025 runner"
```

---

### Task 4: Re-run EXP-025 with the entropy bonus and re-read the verdict (with Mike)

**Files:**
- Create (generated): `experiments/025_head_capacity/outputs/025_table.md` (+ per-run `_b01` files)
- Update: the Week-13 obsidian note (Session 2 findings)
- Modify (conditional): `docs/adr/0001-multi-region-training-strategy.md`

- [ ] **Step 1: Run the entropy-regularized sweep**

Run: `.venv/Scripts/python.exe experiments/025_head_capacity/run.py --entropy-beta 0.01`
Expected: 20 runs across workers (~1h), `_b01`-tagged per-run files + refreshed
`025_summary.json` / `025_table.md`.

- [ ] **Step 2: Read the collapse signal**

Confirm the MLP no longer collapses: per-seed final-block entropy should be > 0 across
seeds (the zero-entropy runs from the first sweep should be gone). If seeds still collapse,
this is Branch A3 (optimization still unstable) -> escalate to LR drop / advantage
normalization before any capacity verdict.

- [ ] **Step 3: Apply the verdict rule (with Mike)**

Band rule as before: MLP held-out mean vs linear `mean + spread`, sparse first.
- MLP now beats linear -> the linear head was the limiter (Branch A1).
- MLP trains stably but does not beat linear -> frozen encoder is the wall (Branch A2)
  -> next step is unfreeze/pre-train the sensory region.

- [ ] **Step 4: Write up + conditional ADR amendment**

Append a Session-2 findings block to the Week-13 obsidian note with the refreshed table
and the resolved (or escalated) verdict. Amend ADR-0001 only if the verdict is now decisive.

- [ ] **Step 5: Full suite**

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: all pass.

---

## Self-Review

**Spec coverage:** entropy_beta on train_episode (T1); byte-identical default (T1 Step 1
test + `if entropy_beta:` guard); GenConfig pass-through + recorded (T2); runner plumbing +
tag suffix + CLI (T3); re-run + verdict + writeup (T4). All spec items mapped.

**Placeholder scan:** no TBD/TODO; every code step shows full code; every run step shows
command + expected output.

**Type consistency:** `entropy_beta: float` consistent across `train_episode` (T1),
`GenConfig` (T2), and `build_configs` (T3). `build_configs` gains a keyword arg with a
default, so the existing zero-arg call sites and Task-3 tests agree. Tag suffix `_b01` used
consistently in the T3 implementation and its tests.
