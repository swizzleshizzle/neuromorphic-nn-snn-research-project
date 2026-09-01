# EXP-055 results - the 0-to-10 pretraining window

> [!warning] PARTIAL. PHASES 1 AND 2 ONLY. **NO POLICY NUMBER EXISTS YET.**
> Phase 3's 48 RL cells are running on the laptop as this is written. **Claims 1, 2 and 3 are
> policy claims and are not adjudicated here. Claim 4 asks whether `S` saturates at the same
> point as policy and needs both halves, so it is not adjudicated either.** Nothing below may be
> combined with a policy number by hand: `aggregate.py` applies the pre-registered rules and is
> the authority. This file records Phase 2's numbers with provenance because that is a standing
> repo rule, not because the experiment has reported.

**Pre-registration:** `docs/superpowers/specs/2026-08-31-exp055-pretraining-left-edge-design.md`,
committed at `386851e` and amended at `9a35a2a`, both before any number existed.

## Provenance

| | |
|---|---|
| Phase 1 | `SwizzlesDuo` (Intel Ultra 9 185H, 22 cores), worktree `C:\Users\mlgbr\wt-exp053` at `da863ed`, 8 workers |
| Phase 2 | this VPS (`liquidweb-vps`, 2 cores) at `e0fe650`, single process |
| Dates | 2026-08-31 into 2026-09-01 |
| Seeds | 0 to 11, twelve per arm, all four arms |
| Encoders | 48, from scratch, EXP-040's `rl_heldout_union` exclusions applied and asserted |

`run.py` and `pretrain_left_edge.py` are byte-identical between `da863ed` and `e0fe650`, so the
encoders trained at the former are valid at the latter.

```bash
# phase 1, on the laptop, from the worktree (the launcher carries the PYTHONPATH override)
powershell -NoProfile -ExecutionPolicy Bypass -File C:\Users\mlgbr\launch055_wt.ps1 -Phase pretrain -Workers 8
# phase 2, on the VPS, after copying the encoders back
.venv/bin/python -u experiments/055_pretraining_left_edge/measure_s.py
```

## Phase 1 - the encoders

| arm | n | mean s | s per epoch | mean move accuracy |
|---|---|---|---|---|
| e1 | 12 | 88.0 | 88.0 | **0.2494** |
| e2 | 12 | 158.5 | 79.2 | 0.3107 |
| e3 | 12 | 269.0 | 89.7 | 0.3265 |
| e5 | 12 | 449.0 | 89.8 | 0.3493 |

**The objective is being learned from the first epoch.** Move accuracy at one epoch is 0.2494
against a 1/6 = 0.1667 chance floor, so `e1` is not a random encoder that failed to train. That
matters for reading the next section: whatever happens to `S` at one epoch happens while the
network is demonstrably learning the task it was given.

## Phase 2 - `S`, and it is NOT flat across this window

| arm | mean `S` | sd | mean `S_cross` | mean `level` |
|---|---|---|---|---|
| **E0** (anchor, EXP-054) | **+0.0100** | 0.0046 | | random init |
| **e1** | **+0.0061** | 0.0019 | +0.0082 | +0.0013 |
| e2 | +0.0113 | 0.0034 | +0.0120 | +0.0077 |
| e3 | +0.0156 | 0.0044 | +0.0154 | +0.0128 |
| e5 | +0.0214 | 0.0051 | +0.0205 | +0.0200 |
| **E10** (anchor, EXP-054) | **+0.0242** | | | |

**Every adjacent contrast is significant**, paired by seed, exact permutation over all `2**12`
sign flips. The shape gate that EXP-052 and EXP-054 each failed is passed here, so a shape word is
licensed rather than assumed:

| contrast | delta | p | |
|---|---|---|---|
| e2 - e1 | +0.0052 | **0.0005** | significant |
| e3 - e2 | +0.0043 | **0.0039** | significant |
| e5 - e3 | +0.0058 | **0.0005** | significant |

`S_cross` tracks `S` at every arm and the `level` control rises with it, so this is not the
within-shell clustering term moving on its own. That is EXP-054's amendment applied here.

### The finding: one epoch puts `S` BELOW its own initialisation

| contrast | delta | W-L-T | p |
|---|---|---|---|
| **e1 - E0** | **-0.0038** | **1-11-0** | **0.0010** |
| e2 - E0 | +0.0013 | 8-4-0 | 0.3408 |
| e3 - E0 | +0.0056 | 10-2-0 | 0.0059 |
| e5 - E0 | +0.0114 | **12-0-0** | 0.0005 |

**This is an exact before-and-after on the same network, not a comparison of two random draws.**
EXP-054 rebuilds E0 as `make_sensory(seed)`, and `pretrain_one` passes no `sensory=`, so
`train_inverse_model` builds `make_sensory(cfg.seed)` itself from the same seed. Verified rather
than reasoned: recomputing `S` from `make_sensory(seed)` under EXP-052's `PRETRAIN` config
reproduces EXP-054's stored E0 records to within 1e-9 on seeds 0, 1 and 2.

```
seed 0: recomputed +0.011564   EXP-054 stored +0.011564   MATCH
seed 1: recomputed +0.006799   EXP-054 stored +0.006799   MATCH
seed 2: recomputed +0.008882   EXP-054 stored +0.008882   MATCH
```

So the `S` curve across this window is **not monotone**. It falls below random init at one epoch,
is indistinguishable from random at two, and only exceeds random from three onward:

```
E0 0.0100  ->  e1 0.0061  ->  e2 0.0113  ->  e3 0.0156  ->  e5 0.0214  ->  E10 0.0242
      down, p 0.0010    up, every step significant
```

**This refines EXP-054 rather than contradicting it.** That experiment established that a random
encoder scores *lowest* of the arms it measured (E0 through E80) and concluded pretraining builds
distance structure in its first 10 epochs. Both statements survive. What was invisible at its
resolution is that the build is not monotone: the first epoch spends structure before the
objective starts paying it back. Every one of the four new arms is still significantly below E10
(p 0.0005, 0.0005, 0.0005, 0.0239), so the left edge really is where the movement is.

## What is NOT claimed here

- **Nothing about policy.** Phase 3 is running. The headline question, whether one epoch already
  reaches the policy plateau, has no data yet.
- **Claim 4 is not adjudicated.** It asks whether `S` and policy saturate at the same point. Half
  the comparison does not exist.
- **`S` is still not a validated policy predictor.** EXP-054 left it unevaluated, and its Claim 4
  now correctly prints UNEVALUATED rather than PASSED. This experiment moves `S` a lot, which is
  what makes it *capable* of evaluating `S` once policy lands. That is a property of the design,
  not a result.
- **Multiplicity.** The three adjacent contrasts are the pre-registered `S` family and the spec
  sets their Bonferroni threshold at 0.0125; all three clear it. **The E0 and E10 contrasts in this
  file are outside that family and are descriptive.** They are reported because the E0 comparison
  is the one the design was built to make possible, not because they were pre-registered as tests.

## Operational note, for the playbook

Pretraining at **8 workers ran at 86.7 s per epoch per seed**, against the 87.3 measured at 10 and
the 24.9 measured at 2 (EXP-050 phase 1). **8 and 10 are indistinguishable, so the contention knee
is below 8, not between 8 and 10.** The playbook warned against extrapolating that curve from two
points; this is the third. Total Phase 1 cost was 11,575 encoder-seconds, about 35 minutes of wall
clock.
