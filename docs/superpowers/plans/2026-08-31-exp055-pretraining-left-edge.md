# EXP-055 Pretraining Left Edge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure the policy and the encoder's distance structure at 1, 2, 3 and 5 pretraining epochs, the window where two independent instruments say all the movement happens and where nobody has measured a single point.

**Architecture:** Three thin drivers in a new experiment directory, reusing existing reviewed machinery. Pretraining reuses EXP-052's `pretrain_sweep.pretrain_one` unchanged. The RL driver copies EXP-052's Phase-2 config field for field with one variable, epochs. The `S` measurement reuses EXP-054's `sequence_sensitivity` unchanged. The aggregator is the only piece with real new logic, and its shape gate is a condition rather than a convention.

**Tech Stack:** Python 3.10, PyTorch + snnTorch, pytest. No scipy: exact permutation tests over `2**12` sign flips.

**Spec:** `docs/superpowers/specs/2026-08-31-exp055-pretraining-left-edge-design.md`, pre-registered at `386851e`. Read section 2 before writing the aggregator: Claim 1 reports a BOUND rather than an equivalence, and Claim 3's shape gate exists because this rule has been broken twice.

## Global Constraints

- **Always run python via `.venv/bin/python`.** Never a bare `python`.
- **Always pass an explicit timeout on Bash calls. The tool default is 120 s**, and anything over it is auto-backgrounded by the harness, which strands agents. Pass 600000.
- **Never run the full test suite** (about 13 min, exceeds any single call). Run only the files each task names.
- **Never run an RL arm or the pretraining sweep.** Those are hours of laptop compute and dispatching them is the controller's call, not an implementer's. Dry-runs only.
- **No em-dashes** anywhere in code, comments, docstrings or commit messages. Use " - ".
- **Plain commit messages. No `Co-Authored-By`. No "Generated with".**
- **Never `git add -A`.** Stage explicit paths.
- **Never write an assertion that cannot fail.** Prefer a measured numeric threshold to a qualitative check.
- **Never weaken a pre-registered threshold.** If something appears to require it, stop and report.
- **Distance-to-solved is an instrument, never a model input.**
- **`N_ACTIONS` comes from `len(MOVES)`, never a literal.**
- Every arm runs 10,000 episodes with a **frozen** encoder (390 trainable), so there is no episode-budget confound.

---

### Task 1: The pretraining driver

**Files:**
- Create: `experiments/055_pretraining_left_edge/pretrain_left_edge.py`
- Create: `experiments/055_pretraining_left_edge/measure_s.py`
- Test: `tests/analysis/test_exp055_pretrain.py`

**Interfaces:**
- Consumes, already committed and working in `experiments/052_pretraining_optimum/pretrain_sweep.py`:
  - `pretrain_one(epochs: int, seed: int, out_dir: Path) -> dict` - trains ONE encoder from scratch, applies EXP-040's `rl_heldout_union` exclusions, asserts no held-out state leaked in as either endpoint, and saves the encoder. Do NOT reimplement any of that.
  - `encoder_path(out_dir, epochs, seed) -> Path` giving `exp052_encoder_e{epochs}_s{seed}.pt`. **EXP-055 needs its own name**, so define a local one rather than reusing this.
- Produces:
  - `EPOCH_ARMS = (1, 2, 3, 5)`
  - `SEEDS = tuple(range(12))`
  - `encoder_path(out_dir: Path, epochs: int, seed: int) -> Path` giving `exp055_encoder_e{epochs}_s{seed}.pt`
  - `measure_s.py` writing one record per encoder as `exp055_S_e{epochs}_s{seed}.json` with keys `epochs`, `seed`, `S`, `S_cross`, `level`, `sim`, `n_by_shell`

- [ ] **Step 1: Write the failing test**

Create `tests/analysis/test_exp055_pretrain.py`:

```python
"""EXP-055: the pretraining driver must reuse EXP-052's machinery, not reimplement it.

`pretrain_one` applies EXP-040's `rl_heldout_union` exclusions and asserts that no RL held-out
state leaked in as either endpoint of a training pair. Without those, an arm could win by
leakage rather than by epochs, and the whole epoch series would be measuring the wrong thing.
A reimplementation would be where that protection quietly goes missing.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

RUN_PATH = (Path(__file__).resolve().parents[2]
            / "experiments" / "055_pretraining_left_edge" / "pretrain_left_edge.py")


def _module():
    spec = importlib.util.spec_from_file_location("exp055_pretrain", RUN_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_four_pre_registered_epoch_arms():
    assert _module().EPOCH_ARMS == (1, 2, 3, 5)


def test_twelve_seeds():
    assert _module().SEEDS == tuple(range(12))


def test_encoder_names_are_exp055_and_do_not_collide_with_exp052():
    """EXP-052's encoders live in a different directory but share the seed and epoch fields.
    A name collision would make an EXP-055 arm silently load an EXP-052 encoder."""
    m = _module()
    name = m.encoder_path(Path("/tmp/x"), 1, 0).name
    assert name == "exp055_encoder_e1_s0.pt"
    assert "exp052" not in name


def test_encoder_names_are_unique_across_arms_and_seeds():
    m = _module()
    names = [m.encoder_path(Path("/tmp/x"), e, s).name
             for e in m.EPOCH_ARMS for s in m.SEEDS]
    assert len(set(names)) == len(names)


def test_it_reuses_exp052_pretrain_one_rather_than_reimplementing():
    """The leakage exclusions and their assertions live inside `pretrain_one`. If this module
    grew its own training loop, those protections would have to be re-derived and could
    silently differ."""
    m = _module()
    import inspect
    src = inspect.getsource(m)
    assert "pretrain_one" in src, "EXP-052's pretrain_one is not referenced at all"
    assert "build_pairs" not in src, (
        "this module appears to build its own training pairs; the leakage exclusions live in "
        "pretrain_one and must not be re-derived here"
    )
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
.venv/bin/python -m pytest tests/analysis/test_exp055_pretrain.py -v
```

Expected: FAIL, the module does not exist.

- [ ] **Step 3: Write the driver**

Create `experiments/055_pretraining_left_edge/pretrain_left_edge.py`:

```python
"""EXP-055 phase 1: pretrain encoders at 1, 2, 3 and 5 epochs, from scratch.

Two independent instruments say everything happens before 10 epochs and nothing after - the
policy goes 0.0000 to 0.2012 then flat, and EXP-054's `S` goes 0.0100 to 0.0242 then flat.
Nobody has measured a point inside that window. These are those points.

REUSES EXP-052's `pretrain_one` UNCHANGED. That function applies EXP-040's `rl_heldout_union`
exclusions and asserts no RL held-out state leaked in as either endpoint of a training pair.
Without them an arm could win by leakage rather than by epochs. Do not reimplement it.

Run (repo root):
    .venv/bin/python -u experiments/055_pretraining_left_edge/pretrain_left_edge.py --workers 10
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import torch

torch.set_num_threads(1)

HERE = Path(__file__).resolve().parent

# EXP-052's sweep is a script, not a package, so it is loaded by path exactly as it loads
# EXP-040's driver.
_SWEEP_PATH = HERE.parent / "052_pretraining_optimum" / "pretrain_sweep.py"
_spec = importlib.util.spec_from_file_location("exp052_sweep", _SWEEP_PATH)
exp052 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(exp052)

EPOCH_ARMS = (1, 2, 3, 5)
SEEDS = tuple(range(12))


def encoder_path(out_dir: Path, epochs: int, seed: int) -> Path:
    """EXP-055's own name. EXP-052's encoders share the epoch and seed fields, so reusing its
    naming would let an arm silently load the wrong encoder."""
    return Path(out_dir) / f"exp055_encoder_e{epochs}_s{seed}.pt"


def _pretrain(epochs: int, seed: int, out_dir: Path) -> dict:
    torch.set_num_threads(1)
    result = exp052.pretrain_one(epochs, seed, out_dir)
    # `pretrain_one` writes EXP-052's filename; rename to EXP-055's so the two sets cannot be
    # confused, and so `run.py` can find them by this experiment's convention.
    src = exp052.encoder_path(out_dir, epochs, seed)
    dst = encoder_path(out_dir, epochs, seed)
    src.replace(dst)
    return {**result, "encoder": dst.name}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, nargs="+", default=list(EPOCH_ARMS))
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--out-dir", type=Path, default=HERE / "outputs")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cells = [(e, s) for e in args.epochs for s in args.seeds]
    if args.skip_existing:
        cells = [(e, s) for e, s in cells if not encoder_path(args.out_dir, e, s).exists()]

    print(f"EXP-055 phase 1: epochs {tuple(args.epochs)}, {len(cells)} encoders, "
          f"{args.workers} workers")
    print("  FROM SCRATCH, not warm-started. EXP-040's rl_heldout_union exclusions apply.")
    print("  Pretraining is memory-bandwidth-bound: about 2.86 effective cores from 10 workers.\n",
          flush=True)
    if args.dry_run or not cells:
        print(f"  {len(cells)} cell(s) NOT started.")
        return

    records = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_pretrain, e, s, args.out_dir): (e, s) for e, s in cells}
        for i, fut in enumerate(as_completed(futures), 1):
            records.append(fut.result())
            print(f"  {i}/{len(cells)}", flush=True)

    (args.out_dir / "pretrain_left_edge.json").write_text(
        json.dumps(records), encoding="utf-8")
    print(f"\ndone. encoders in {args.out_dir}.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
.venv/bin/python -m pytest tests/analysis/test_exp055_pretrain.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Dry-run only**

```bash
.venv/bin/python experiments/055_pretraining_left_edge/pretrain_left_edge.py --dry-run
```

Expected: prints 48 cells and trains nothing. **Do not run it for real** - that is roughly half an hour of laptop compute and the controller dispatches it.

- [ ] **Step 6: Write the S driver**

Claim 4 needs EXP-054's statistic measured on THESE encoders. EXP-054's own driver cannot be
reused directly: its `ARMS` dict hardcodes paths to the EXP-040/052/050 encoders. The statistic
itself is reusable unchanged; only the iteration over encoders is new.

Create `experiments/055_pretraining_left_edge/measure_s.py`:

```python
"""EXP-055 phase 2: EXP-054's sequence-sensitivity, measured on the left-edge encoders.

FREE - 60 encoders took 8.8 s in EXP-054, because `concept_rates` is batched. Runs on the VPS
and must never touch the laptop.

The statistic is imported unchanged from EXP-054's module. Only the iteration is new: that
experiment's driver hardcodes its own arms' encoder paths and cannot be pointed here.

`S_cross` is reported beside `S` per EXP-054's amendment: `S` includes the within-shell term
and is therefore partly clustering, so a change in `S` that is not matched in `S_cross` is not
a change in graded distance structure. `level` is the collapse control.

Run (repo root), after phase 1:
    .venv/bin/python -u experiments/055_pretraining_left_edge/measure_s.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import torch

from neuromorphic.analysis.sequence_sensitivity import (
    DEPTHS,
    N_PER_SHELL,
    sensitivity_from_similarity,
    sequence_sensitivity,
)
from neuromorphic.envs.cube_distance import ExactBFSDistance
from neuromorphic.training.encoder_pretrain import load_encoder

torch.set_num_threads(1)

HERE = Path(__file__).resolve().parent

_PRETRAIN_PATH = HERE / "pretrain_left_edge.py"
_spec = importlib.util.spec_from_file_location("exp055_pretrain", _PRETRAIN_PATH)
_pre = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_pre)
encoder_path = _pre.encoder_path
EPOCH_ARMS = _pre.EPOCH_ARMS
SEEDS = _pre.SEEDS


def record_name(epochs: int, seed: int) -> str:
    return f"exp055_S_e{epochs}_s{seed}.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, nargs="+", default=list(EPOCH_ARMS))
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--out-dir", type=Path, default=HERE / "outputs")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cells = [(e, s) for e in args.epochs for s in args.seeds]
    missing = [str(encoder_path(args.out_dir, e, s)) for e, s in cells
               if not encoder_path(args.out_dir, e, s).exists()]
    if missing:
        raise SystemExit(
            f"missing {len(missing)} encoder(s), first {missing[:3]}. Run phase 1 first."
        )

    print(f"EXP-055 phase 2: S over {len(cells)} encoders, shells {DEPTHS}, "
          f"up to {N_PER_SHELL} states each")
    print("  the statistic trains NOTHING and this does not touch the laptop.\n", flush=True)
    if args.dry_run:
        print(f"  --dry-run: {len(cells)} cell(s) NOT measured.")
        return

    # One bounded build, reused. Depth 6 is 11,913 states, about 0.04 s.
    provider = ExactBFSDistance(max_depth=max(DEPTHS))

    for i, (e, s) in enumerate(cells, 1):
        sensory = load_encoder(encoder_path(args.out_dir, e, s), seed=s)
        result = sequence_sensitivity(sensory, provider, seed=s)
        sim = {tuple(int(x) for x in k.split("_")): v for k, v in result["sim"].items()}
        record = {
            "epochs": e,
            "seed": s,
            "S": result["S"],
            "S_cross": sensitivity_from_similarity(sim, min_separation=1),
            "level": sum(result["sim"].values()) / len(result["sim"]),
            "sim": result["sim"],
            "n_by_shell": result["n_by_shell"],
        }
        (args.out_dir / record_name(e, s)).write_text(json.dumps(record), encoding="utf-8")
        print(f"  {i}/{len(cells)}  e{e} s{s}  S={result['S']:+.4f}", flush=True)

    print(f"\ndone. S records in {args.out_dir}.")


if __name__ == "__main__":
    main()
```

**Check `sensitivity_from_similarity`'s real signature before writing this.** EXP-054's fix wave
added a keyword-only `min_separation` parameter defaulting to 0, and `sim_from_record` may already
exist to do the string-key parsing above. If it does, import and use it rather than re-parsing.
Report which you found.

- [ ] **Step 7: Dry-run the S driver**

```bash
.venv/bin/python experiments/055_pretraining_left_edge/measure_s.py --dry-run
```

Expected: it EXITS with "missing 48 encoder(s)... Run phase 1 first", because phase 1 has not
run. That is the pre-flight guard working. Confirm the message names a count and a path.

- [ ] **Step 8: Commit**

```bash
git add experiments/055_pretraining_left_edge/pretrain_left_edge.py experiments/055_pretraining_left_edge/measure_s.py tests/analysis/test_exp055_pretrain.py
git commit -m "EXP-055: pretrain encoders at 1, 2, 3 and 5 epochs, and measure S on them

Reuses EXP-052's pretrain_one unchanged rather than reimplementing it. That function
applies EXP-040's rl_heldout_union exclusions and asserts no RL held-out state leaked in
as either endpoint, so an arm cannot win by leakage rather than by epochs. A test asserts
this module never builds its own pairs.

Encoders take an exp055 name because EXP-052's share the epoch and seed fields, and a
collision would let an arm silently load the wrong encoder.

measure_s.py imports EXP-054's statistic unchanged and only re-does the iteration, because
that experiment's driver hardcodes its own arms' encoder paths. S_cross and level ride
along per its amendment, so a change in S that is not matched in S_cross can be seen as
clustering rather than graded distance structure."
```

---

### Task 2: The RL driver

**Files:**
- Create: `experiments/055_pretraining_left_edge/run.py`
- Test: `tests/analysis/test_exp055_arms.py`

**Interfaces:**
- Consumes `encoder_path` and `EPOCH_ARMS` from Task 1.
- Produces:
  - `tag_for(epochs: int) -> str` giving `exp055_e{epochs}_d6`
  - `sweep_configs(seeds, out_dir, epoch_arms) -> list[CubeConfig]`
  - `ANCHORS: dict[int, tuple[Path, str, float]]` mapping epoch level to (records dir, tag, published mean) for the anchors that already exist

- [ ] **Step 1: Write the failing test**

Create `tests/analysis/test_exp055_arms.py`:

```python
"""EXP-055: each arm must be EXP-052's Phase 2 with exactly one field changed.

EXP-052 copied EXP-043's depth-6 cell field for field and changed the encoder. EXP-055 does the
same. An arm carrying a second difference produces a paired delta that measures something
nobody chose, and a nine-point epoch curve would be silently wrong in one place.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

RUN_PATH = (Path(__file__).resolve().parents[2]
            / "experiments" / "055_pretraining_left_edge" / "run.py")


def _module():
    spec = importlib.util.spec_from_file_location("exp055_run", RUN_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_arms_match_exp052_phase_two_field_for_field(tmp_path):
    """Every field except the encoder path and the tag must equal EXP-052's."""
    m = _module()
    for c in m.sweep_configs((0, 1), tmp_path, (1,)):
        assert c.depth == 6
        assert c.episodes == 10_000
        assert c.curriculum == (1, 2, 3, 4, 5, 6)
        assert c.max_steps_by_depth == ((1, 2),)
        assert c.entropy_beta == 0.0
        assert c.normalize_advantages is False
        assert c.max_depth == 6
        assert c.arm == "regionalized"
        assert c.readout == "concept"


def test_every_arm_is_frozen(tmp_path):
    """390 trainable, not 27,206. A fine-tuned arm is a different architecture and must never
    be tabulated with these."""
    m = _module()
    for c in m.sweep_configs((0,), tmp_path, m.EPOCH_ARMS):
        assert c.encoder_lr is None, "an arm is fine-tuning; every EXP-055 arm is frozen"
        assert c.plasticity_gate is None
        assert c.critic_lr is None


def test_each_arm_loads_its_own_epoch_encoder(tmp_path):
    m = _module()
    for e in m.EPOCH_ARMS:
        c = m.sweep_configs((3,), tmp_path, (e,))[0]
        assert f"exp055_encoder_e{e}_s3.pt" in str(c.encoder_state_path)


def test_tags_are_distinct_per_epoch_arm():
    """`record_filename` does not encode the encoder, so without a per-epoch tag the four arms
    would silently overwrite each other into one set of files."""
    m = _module()
    tags = [m.tag_for(e) for e in m.EPOCH_ARMS]
    assert len(set(tags)) == len(tags)
    assert tags[0] == "exp055_e1_d6"


def test_record_filenames_do_not_collide(tmp_path):
    from neuromorphic.training.cube_baseline import record_filename
    m = _module()
    names = [record_filename(c)
             for c in m.sweep_configs(range(12), tmp_path, m.EPOCH_ARMS)]
    assert len(set(names)) == len(names)


def test_anchors_point_at_the_experiments_that_measured_them():
    """0, 10, 20, 40 and 80 all exist and are NOT re-run. A wrong anchor would corrupt every
    contrast that uses it."""
    m = _module()
    assert set(m.ANCHORS) == {10, 20, 40, 80}
    assert m.ANCHORS[10][1] == "exp052_e10_d6"
    assert m.ANCHORS[20][1] == "exp052_e20_d6"
    assert m.ANCHORS[40][1] == "exp043_capped_d6"
    assert m.ANCHORS[80][1] == "exp050_pre2_d6"
    assert m.ANCHORS[10][2] == pytest.approx(0.2012)
    assert m.ANCHORS[40][2] == pytest.approx(0.1800)
    assert m.ANCHORS[80][2] == pytest.approx(0.0887)
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
.venv/bin/python -m pytest tests/analysis/test_exp055_arms.py -v
```

Expected: FAIL, `run.py` does not exist.

- [ ] **Step 3: Write the driver**

Create `experiments/055_pretraining_left_edge/run.py`:

```python
"""EXP-055 phase 3: the RL arms at 1, 2, 3 and 5 pretraining epochs.

    epochs   depth-6 policy      S (EXP-054)
      0          0.0000            0.0100     EXP-036 policy, EXP-054 S
    1,2,3,5    UNMEASURED        UNMEASURED   <- this experiment
     10          0.2012            0.0242     EXP-052
     20          0.1850            0.0241     EXP-052
     40          0.1800            0.0246     EXP-043
     80          0.0887            0.0244     EXP-050

Both curves are flat from 10 onward. Both have one unexplained jump at the left edge.

PRE-REGISTERED CONTRACT: docs/superpowers/specs/2026-08-31-exp055-pretraining-left-edge-design.md

  1. PRIMARY: e10 minus e1. CONFIRMED at >= +0.05, p <= 0.05, meaning epochs 2 to 10 buy
     something real. IF NOT SIGNIFICANT THE OUTPUT IS A BOUND, NOT AN EQUIVALENCE.
  2. THE FLOOR: e1 against 0 epochs (EXP-036, 0.0000 on all twelve seeds). If e1 already lands
     near 0.20, pretraining's contribution is almost entirely escaping random init.
  3. SHAPE: adjacent contrasts, and a shape may be named ONLY where one is significant.
  4. Does S saturate at the same point as policy? A dissociation would separate "the encoder
     has the structure" from "the policy can use it".

Every arm is FROZEN (390 trainable) and runs 10,000 episodes, so there is no episode-budget
confound and EXP-046's curve does not apply.

Run (repo root), after phase 1:
    .venv/bin/python -u experiments/055_pretraining_left_edge/run.py --epochs 1 --workers 6
"""

from __future__ import annotations

import argparse
import importlib.util
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import torch

from neuromorphic.training.cube_baseline import (
    CubeConfig,
    record_filename,
    run_cube_baseline,
)

torch.set_num_threads(1)

HERE = Path(__file__).resolve().parent

_PRETRAIN_PATH = HERE / "pretrain_left_edge.py"
_spec = importlib.util.spec_from_file_location("exp055_pretrain", _PRETRAIN_PATH)
_pretrain_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_pretrain_mod)
encoder_path = _pretrain_mod.encoder_path
EPOCH_ARMS = _pretrain_mod.EPOCH_ARMS
SEEDS = _pretrain_mod.SEEDS

DEPTH = 6
EPISODES = 10_000
CAP = ((1, 2),)

# The anchors, none of which is re-run: (records dir, tag, published mean).
ANCHORS = {
    10: (Path("experiments/052_pretraining_optimum/outputs"), "exp052_e10_d6", 0.2012),
    20: (Path("experiments/052_pretraining_optimum/outputs"), "exp052_e20_d6", 0.1850),
    40: (Path("experiments/043_cap_at_depth_5_6/outputs"), "exp043_capped_d6", 0.1800),
    80: (Path("experiments/050_objective_vs_gradient/outputs"), "exp050_pre2_d6", 0.0887),
}
# 0 epochs is EXP-036, measured at exactly 0.0000 on all twelve seeds, so it has no variance
# and is handled descriptively rather than as a paired contrast. See spec Claim 2.
ZERO_EPOCH_MEAN = 0.0000

BAR = 0.05


def curriculum_for(depth: int) -> tuple[int, ...]:
    return tuple(range(1, depth + 1))


def tag_for(epochs: int) -> str:
    """`record_filename` does not encode the encoder, so the epoch count must live in the tag
    or the four arms would silently overwrite each other."""
    return f"exp055_e{epochs}_d{DEPTH}"


def sweep_configs(seeds, out_dir: Path, epoch_arms) -> list[CubeConfig]:
    """EXP-052's Phase 2 copied field for field. ONE variable: which encoder, hence epochs."""
    return [
        CubeConfig(
            arm="regionalized", readout="concept", tag=tag_for(e),
            depth=DEPTH, seed=seed, sigma=0.0, episodes=EPISODES,
            curriculum=curriculum_for(DEPTH), max_steps_by_depth=CAP,
            entropy_beta=0.0, normalize_advantages=False,
            encoder_state_path=str(encoder_path(out_dir, e, seed)),
            max_depth=DEPTH, out_dir=out_dir,
        )
        for e in epoch_arms
        for seed in seeds
    ]


def _run(cfg: CubeConfig) -> dict:
    torch.set_num_threads(1)
    return run_cube_baseline(cfg)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, nargs="+", default=list(EPOCH_ARMS))
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out-dir", type=Path, default=HERE / "outputs")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    missing = [str(encoder_path(args.out_dir, e, s))
               for e in args.epochs for s in args.seeds
               if not encoder_path(args.out_dir, e, s).exists()]
    if missing:
        raise SystemExit(
            f"missing {len(missing)} encoder(s), first {missing[:3]}. Run phase 1 first: "
            "pretrain_left_edge.py"
        )

    configs = sweep_configs(args.seeds, args.out_dir, args.epochs)
    if any(c.encoder_lr is not None for c in configs):
        raise SystemExit("EXP-055 arms are FROZEN: encoder_lr must stay None.")
    names = [record_filename(c) for c in configs]
    if len(set(names)) != len(names):
        raise SystemExit("record filename collision")
    if args.skip_existing:
        configs = [c for c in configs if not (args.out_dir / record_filename(c)).exists()]

    print(f"EXP-055: depth {DEPTH}, {EPISODES:,} episodes, epochs {tuple(args.epochs)}, "
          f"{len(configs)} runs, {args.workers} workers")
    print(f"  FROZEN encoder, 390 trainable. ONE VARIABLE vs EXP-052's phase 2: the epoch count.")
    print(f"  Anchors NOT re-run: 0 -> {ZERO_EPOCH_MEAN}, "
          + ", ".join(f"{e} -> {v[2]}" for e, v in sorted(ANCHORS.items())))
    print(f"  Claim 1 is e10 minus e1, CONFIRMED at >= +{BAR}, p <= 0.05. A NON-significant")
    print(f"  result is reported as a BOUND, never as an equivalence.\n", flush=True)

    if not configs:
        print("nothing to do.")
        return
    if args.dry_run:
        print(f"  --dry-run: {len(configs)} cell(s) NOT started.")
        return

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_run, c): c for c in configs}
        for i, fut in enumerate(as_completed(futures), 1):
            fut.result()
            print(f"  {i}/{len(configs)}", flush=True)

    print(f"\ndone. records in {args.out_dir}.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
.venv/bin/python -m pytest tests/analysis/test_exp055_arms.py -v
```

Expected: 6 passed.

- [ ] **Step 5: Dry-run only**

```bash
.venv/bin/python experiments/055_pretraining_left_edge/run.py --dry-run
```

Expected: it EXITS with "missing 48 encoder(s)... Run phase 1 first", because phase 1 has not been run. That is the pre-flight guard working, not a failure. Confirm the message names the count and a path, and say so in your report.

- [ ] **Step 6: Commit**

```bash
git add experiments/055_pretraining_left_edge/run.py tests/analysis/test_exp055_arms.py
git commit -m "EXP-055: RL arms at 1, 2, 3 and 5 pretraining epochs

EXP-052's phase 2 copied field for field with one variable, the epoch count. Tests pin
every copied field, assert every arm is frozen at 390 trainable rather than fine-tuning,
and assert each arm loads its own epoch's encoder.

Anchors at 10, 20, 40 and 80 are named with the experiment and tag that measured them and
are never re-run. A wrong anchor would corrupt every contrast that uses it."
```

---

### Task 3: The aggregator, with a shape gate that is a condition

**Files:**
- Create: `experiments/055_pretraining_left_edge/aggregate.py`
- Test: `tests/analysis/test_exp055_aggregate.py`

**Interfaces:**
- Consumes records written by Task 2 and the anchors from `run.py`.
- Produces:
  - `permutation_p(diffs) -> float`
  - `describe_contrast(delta: float, p: float, bar: float, alpha: float) -> str`
  - `shape_word(delta: float, p: float, alpha: float) -> str`
  - `load_s(directory: Path) -> dict` keyed by epochs then seed
  - a printed report covering Claims 1 to 4

- [ ] **Step 1: Write the failing test**

Create `tests/analysis/test_exp055_aggregate.py`:

```python
"""EXP-055: the aggregator's interpretive rules, tested before any number exists.

Two rules here are responses to specific failures in this project's own history, and both are
implemented as CONDITIONS rather than as conventions in the prose:

  - a non-significant contrast is reported as a BOUND, never as an equivalence
  - a shape word may not be emitted for a contrast that is not significant
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

AGG_PATH = (Path(__file__).resolve().parents[2]
            / "experiments" / "055_pretraining_left_edge" / "aggregate.py")


def _module():
    spec = importlib.util.spec_from_file_location("exp055_agg", AGG_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_permutation_p_on_a_known_case():
    """Twelve all-positive differences is the most extreme two-sided outcome: 2/4096."""
    assert _module().permutation_p([0.1] * 12) == pytest.approx(2 / 4096)


def test_a_non_significant_small_contrast_is_reported_as_a_bound():
    """THE CLAIM 1 RULE. A non-significant difference is not evidence of equality, and n=12 is
    exactly where that conversion is tempting."""
    out = _module().describe_contrast(delta=0.004, p=0.82, bar=0.05, alpha=0.05)
    assert "bound" in out.lower()
    assert "indistinguishable" in out.lower()
    for forbidden in ("as good as", "equal", "equivalent", "no different"):
        assert forbidden not in out.lower(), f"the wording claims equivalence: {out}"


def test_a_confirming_contrast_says_confirmed():
    out = _module().describe_contrast(delta=0.08, p=0.01, bar=0.05, alpha=0.05)
    assert "CONFIRMED" in out


def test_a_large_but_non_significant_contrast_is_not_a_bound_below_the_bar():
    """delta 0.09 at p 0.30 does NOT bound the effect below 0.05 - it is simply unresolved,
    and saying otherwise would invert the finding."""
    out = _module().describe_contrast(delta=0.09, p=0.30, bar=0.05, alpha=0.05)
    assert "unresolved" in out.lower()
    assert "bound" not in out.lower()


def test_shape_word_is_refused_when_a_contrast_is_not_significant():
    """THE CLAIM 3 GATE. EXP-052 named a monotone shape from indistinguishable means, and
    EXP-054's aggregator repeated it three days later. The gate is a condition, not a habit."""
    m = _module()
    assert m.shape_word(delta=0.04, p=0.40, alpha=0.05) == "indistinguishable"
    assert m.shape_word(delta=-0.04, p=0.40, alpha=0.05) == "indistinguishable"


def test_shape_word_is_allowed_when_significant():
    m = _module()
    assert m.shape_word(delta=0.06, p=0.01, alpha=0.05) == "rises"
    assert m.shape_word(delta=-0.06, p=0.01, alpha=0.05) == "falls"
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
.venv/bin/python -m pytest tests/analysis/test_exp055_aggregate.py -v
```

Expected: FAIL, `aggregate.py` does not exist.

- [ ] **Step 3: Write the aggregator**

Create `experiments/055_pretraining_left_edge/aggregate.py`:

```python
"""EXP-055 aggregator: apply the pre-registered rules to the records on disk.

Thresholds committed in the spec before any number existed.

> TWO RULES HERE ARE CONDITIONS, NOT CONVENTIONS, and both exist because this project broke
> them. EXP-052's aggregator named a monotone shape from four means, three of which were
> indistinguishable at p 0.49 to 0.84. EXP-054's aggregator then printed a verdict derived from
> a rank correlation over four means whose spread was 0.08x their own within-arm sd, three days
> after the rule against exactly that was adopted. Prose did not stop either one.

Usage:
    .venv/bin/python experiments/055_pretraining_left_edge/aggregate.py
"""

from __future__ import annotations

import importlib.util
import itertools
import json
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent

_RUN_PATH = HERE / "run.py"
_spec = importlib.util.spec_from_file_location("exp055_run", _RUN_PATH)
_run = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_run)

ALPHA = 0.05
BAR = 0.05
BONFERRONI = 0.0083          # six pre-registered comparisons
EPOCH_ARMS = _run.EPOCH_ARMS
ANCHORS = _run.ANCHORS
ZERO_EPOCH_MEAN = _run.ZERO_EPOCH_MEAN
ADJACENT = ((1, 2), (2, 3), (3, 5), (5, 10))


def permutation_p(diffs) -> float:
    """Exact two-sided paired permutation over all 2**n sign flips. No scipy in the venv."""
    n, obs = len(diffs), abs(sum(diffs))
    return sum(1 for s in itertools.product((1, -1), repeat=n)
               if abs(sum(x * y for x, y in zip(s, diffs))) >= obs - 1e-12) / 2 ** n


def describe_contrast(delta: float, p: float, bar: float = BAR, alpha: float = ALPHA) -> str:
    """The Claim 1 wording rule, as a function.

    A non-significant difference is NOT evidence of equality. When the contrast misses
    significance AND the delta is under the bar, the honest output is a BOUND on the effect at
    the power available. When it misses significance with a delta OVER the bar, nothing is
    bounded - it is simply unresolved, and calling that a bound would invert the finding.
    """
    if abs(delta) >= bar and p <= alpha:
        return (f"CONFIRMED: delta {delta:+.4f} at p {p:.4f}, clearing the +{bar} bar.")
    if p > alpha and abs(delta) < bar:
        return (f"indistinguishable at n=12: delta {delta:+.4f}, p {p:.4f}. This BOUNDS the "
                f"effect below +{bar} at this power. It is NOT evidence that the two are equal.")
    if p > alpha:
        return (f"unresolved: delta {delta:+.4f} exceeds the +{bar} bar but p {p:.4f} misses "
                f"significance. Nothing is bounded and nothing is confirmed; n=12 cannot "
                f"settle it.")
    return (f"significant but sub-bar: delta {delta:+.4f} at p {p:.4f}. Real, and smaller than "
            f"the +{bar} the claim required.")


def shape_word(delta: float, p: float, alpha: float = ALPHA) -> str:
    """The Claim 3 gate. Returns a direction word ONLY for a significant contrast.

    This is the condition that EXP-052 and EXP-054 each needed and neither had.
    """
    if p > alpha:
        return "indistinguishable"
    return "rises" if delta > 0 else "falls"


def load(directory: Path, tag: str) -> dict:
    out = {}
    for p in Path(directory).glob("*.json"):
        r = json.loads(p.read_text())
        if isinstance(r, dict) and r.get("tag") == tag and r.get("depth") == 6:
            out[int(r["seed"])] = r
    return out


def load_s(directory: Path) -> dict:
    """S records written by measure_s.py, keyed by epochs then seed.

    Only EXP-055's own encoders have these. The anchors' S values live in EXP-054's outputs and
    are reported in that experiment's RESULTS rather than recomputed here.
    """
    out = {}
    for p in Path(directory).glob("exp055_S_e*.json"):
        r = json.loads(p.read_text())
        out.setdefault(int(r["epochs"]), {})[int(r["seed"])] = r
    return out


def arm_records() -> dict:
    """Every epoch level that has records, keyed by epoch. Includes the anchors."""
    by_epoch = {}
    for e in EPOCH_ARMS:
        recs = load(HERE / "outputs", _run.tag_for(e))
        if recs:
            by_epoch[e] = recs
    for e, (directory, tag, _mean) in ANCHORS.items():
        recs = load(directory, tag)
        if recs:
            by_epoch[e] = {s: r for s, r in recs.items() if s < 12}
    return by_epoch


def paired(a: dict, b: dict, field: str = "success_rate"):
    seeds = sorted(set(a) & set(b))
    return seeds, [b[s][field] - a[s][field] for s in seeds]


def main() -> None:
    by_epoch = arm_records()
    if not by_epoch:
        print("no records; run pretrain_left_edge.py then run.py first")
        return

    print("EXP-055: the 0-to-10 pretraining window")
    print("Two rules here are conditions, not conventions: a non-significant contrast is a")
    print("BOUND and never an equivalence, and no shape word is emitted without significance.\n")

    print(f"{'epochs':>7} {'n':>3} {'policy':>9} {'sd':>8}")
    print(f"{0:>7} {12:>3} {ZERO_EPOCH_MEAN:>9.4f} {0.0:>8.4f}   (EXP-036, no variance)")
    for e in sorted(by_epoch):
        v = [by_epoch[e][s]["success_rate"] for s in sorted(by_epoch[e])]
        print(f"{e:>7} {len(v):>3} {st.mean(v):>9.4f} "
              f"{st.stdev(v) if len(v) > 1 else 0.0:>8.4f}")

    print("\nCLAIM 1 PRIMARY - is there a real ramp between 1 and 10 epochs?")
    if 1 in by_epoch and 10 in by_epoch:
        seeds, diffs = paired(by_epoch[1], by_epoch[10])
        p = permutation_p(diffs)
        print(f"  e10 minus e1, n {len(seeds)}")
        print(f"  {describe_contrast(st.mean(diffs), p)}")
    else:
        print("  requires both the e1 arm and the e10 anchor.")

    print("\nCLAIM 2 THE FLOOR - e1 against 0 epochs (EXP-036, 0.0000 on all twelve seeds).")
    if 1 in by_epoch:
        v = [by_epoch[1][s]["success_rate"] for s in sorted(by_epoch[1])]
        print(f"  e1 mean {st.mean(v):.4f} against {ZERO_EPOCH_MEAN:.4f}. Descriptive: the "
              "zero-epoch arm has no variance, so there is no paired test to run.")
        print("  If this lands near 0.20, pretraining's contribution is almost entirely "
              "escaping random init.")

    print("\nCLAIM 3 SHAPE - adjacent contrasts. A shape word appears ONLY where significant.")
    for a, b in ADJACENT:
        if a not in by_epoch or b not in by_epoch:
            continue
        seeds, diffs = paired(by_epoch[a], by_epoch[b])
        p = permutation_p(diffs)
        word = shape_word(st.mean(diffs), p)
        flag = "" if p <= BONFERRONI else f"   (above Bonferroni {BONFERRONI})"
        print(f"  e{a} -> e{b}: {word}   delta {st.mean(diffs):+.4f}   p {p:.4f}{flag}")

    print("\nCLAIM 4 - does S saturate at the same point as policy?")
    s_by_epoch = load_s(HERE / "outputs")
    if not s_by_epoch:
        print("  no S records; run measure_s.py. Claim 4 cannot be evaluated without them.")
    else:
        print(f"  {'epochs':>7} {'S':>9} {'S_cross':>9} {'level':>9}")
        for e in sorted(s_by_epoch):
            rs = s_by_epoch[e]
            print(f"  {e:>7} {st.mean([r['S'] for r in rs.values()]):>9.4f} "
                  f"{st.mean([r['S_cross'] for r in rs.values()]):>9.4f} "
                  f"{st.mean([r['level'] for r in rs.values()]):>9.4f}")
        print("  adjacent contrasts on S, under the SAME gate as Claim 3:")
        for a, b in ADJACENT:
            if a not in s_by_epoch or b not in s_by_epoch:
                continue
            seeds = sorted(set(s_by_epoch[a]) & set(s_by_epoch[b]))
            diffs = [s_by_epoch[b][x]["S"] - s_by_epoch[a][x]["S"] for x in seeds]
            p = permutation_p(diffs)
            print(f"    e{a} -> e{b}: {shape_word(st.mean(diffs), p)}   "
                  f"delta {st.mean(diffs):+.4f}   p {p:.4f}")
        print("  A DISSOCIATION - S turning over at a different epoch than policy - separates")
        print("  'the encoder has the structure' from 'the policy can use it'.")

    print(f"\nMULTIPLICITY: six pre-registered comparisons, Bonferroni {BONFERRONI}. Claim 1")
    print(f"keeps its {ALPHA} as the single primary; the rest are read against Bonferroni")
    print("whenever one is used to name a shape.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
.venv/bin/python -m pytest tests/analysis/test_exp055_aggregate.py -v
```

Expected: 6 passed.

- [ ] **Step 5: Run the aggregator against an empty outputs directory**

```bash
.venv/bin/python experiments/055_pretraining_left_edge/aggregate.py
```

Expected: it loads the ANCHORS that already exist on disk and prints their rows, then reports
that Claim 1 requires the e1 arm. It must NOT raise. If it raises, an anchor path or tag is
wrong and that is a real defect - report it rather than working around it.

- [ ] **Step 6: Commit**

```bash
git add experiments/055_pretraining_left_edge/aggregate.py tests/analysis/test_exp055_aggregate.py
git commit -m "EXP-055: aggregator, with the bound rule and the shape gate as conditions

describe_contrast refuses to convert a non-significant contrast into an equivalence, and
distinguishes a genuine bound (small delta, not significant) from an unresolved result
(large delta, not significant) - calling the second a bound would invert the finding.

shape_word returns a direction only for a significant contrast. EXP-052 named a monotone
shape from indistinguishable means and EXP-054's aggregator repeated it three days later,
so the gate is a condition rather than an intention in the prose."
```
