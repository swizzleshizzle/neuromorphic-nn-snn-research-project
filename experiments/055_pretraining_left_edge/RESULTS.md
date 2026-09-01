# EXP-055 results - the 0-to-10 pretraining window

> **COMPLETE.** All three phases ran. 48 encoders, 48 `S` measurements, 48 RL cells, 12 seeds per
> arm, no tracebacks and no broken pools.
>
> **HEADLINE: one epoch does NOT reach the plateau, so the framing this experiment was built to
> test is REFUTED.** `e1` lands at **0.0854**, 42.5% of the `e10` anchor. Escaping random
> initialisation is worth a lot and is not the whole story: epochs 2 to 10 buy a further
> **+0.1158 at p 0.0098**, which clears the pre-registered bar. The inverse-model objective is
> doing real work after the first epoch.
>
> **AND `S` AND POLICY COME APART AT THE LEFT EDGE, WITH BOTH SIDES RESOLVED.** At one epoch `S`
> falls **below its own initialisation** (11 of 12 seeds, p 0.0010) while policy rises from
> exactly 0.0000 to 0.0854 (12 of 12 seeds). The encoder's distance structure gets worse than
> random while the policy gets dramatically better.

**Pre-registration:** `docs/superpowers/specs/2026-08-31-exp055-pretraining-left-edge-design.md`,
committed at `386851e` and amended at `9a35a2a`, both before any number existed. Thresholds below
are the ones fixed there.

## Provenance

| | |
|---|---|
| Phase 1 (48 encoders) | `SwizzlesDuo` (Intel Ultra 9 185H, 22 cores), worktree `C:\Users\mlgbr\wt-exp053` at `da863ed`, 8 workers, ~35 min |
| Phase 2 (`S`) | this VPS (`liquidweb-vps`) at `e0fe650`, single process, seconds |
| Phase 3 (48 RL cells) | `SwizzlesDuo`, worktree at `e0fe650`, 8 workers, **20:11 to 14:12, 18.0 h** |
| Dates | 2026-08-31 into 2026-09-01 |
| Seeds | 0 to 11, twelve per arm, all four arms |
| Encoders | from scratch, EXP-040's `rl_heldout_union` exclusions applied and asserted |

`run.py` and `pretrain_left_edge.py` are byte-identical between `da863ed` and `e0fe650`, so the
encoders trained at the former are valid at the latter.

```bash
# phase 1 and phase 3, on the laptop, from the worktree (the launcher carries the PYTHONPATH
# override and refuses to start if the wrong library resolves)
powershell -NoProfile -ExecutionPolicy Bypass -File C:\Users\mlgbr\launch055_wt.ps1 -Phase pretrain -Workers 8
powershell -NoProfile -ExecutionPolicy Bypass -File C:\Users\mlgbr\launch055_wt.ps1 -Phase rl -AllArms -Workers 8
# phases 2 and 4, on the VPS, after copying the encoders and records back
.venv/bin/python -u experiments/055_pretraining_left_edge/measure_s.py
.venv/bin/python -u experiments/055_pretraining_left_edge/aggregate.py
```

## The nine-point curve

Four arms are new. Five are anchors and were **not** re-run.

| epochs | policy | sd | `S` | source |
|---|---|---|---|---|
| 0 | 0.0000 | 0.0000 | 0.0100 | EXP-036 policy, EXP-054 `S` |
| **1** | **0.0854** | 0.0480 | **0.0061** | **this experiment** |
| **2** | **0.1483** | 0.0602 | **0.0113** | **this experiment** |
| **3** | **0.1746** | 0.0441 | **0.0156** | **this experiment** |
| **5** | **0.1762** | 0.0869 | **0.0214** | **this experiment** |
| 10 | 0.2012 | 0.1026 | 0.0242 | EXP-052 policy, EXP-054 `S` |
| 20 | 0.1850 | 0.0939 | 0.0241 | EXP-052 |
| 40 | 0.1800 | 0.0985 | 0.0246 | EXP-043 policy, EXP-054 `S` |
| 80 | 0.0887 | 0.0879 | 0.0244 | EXP-050 policy, EXP-054 `S` |

## Claim 1, PRIMARY - CONFIRMED

**`e10` minus `e1` = +0.1158, p 0.0098**, paired by seed, exact permutation over all `2**12` sign
flips. Clears the pre-registered +0.05 bar and the 0.05 alpha.

**This is the claim the experiment existed to settle, and it settles it in the direction the spec
treated as the interesting one.** The spec's stake was: *"If one epoch already reaches the plateau,
pretraining's contribution is almost entirely 'stop being randomly initialised', and the objective
itself is doing far less work than four experiments of narrative imply."* One epoch does not reach
the plateau. **The EXP-039/040 framing does not need the rewrite this experiment was prepared to
force.**

## Claim 2, THE FLOOR - descriptive, and it did not land where the spec flagged

`e1` mean **0.0854** against EXP-036's zero-epoch arm at exactly **0.0000 on all twelve seeds**.
No paired test exists, because the zero arm has no variance. Every one of the twelve `e1` seeds is
strictly above zero (min 0.0050, max 0.1750).

The spec pre-registered a reading for one specific outcome: *"If this lands near 0.20, pretraining's
contribution is almost entirely escaping random init."* **It landed at 0.0854, not near 0.20**, so
that reading does not apply. The split is roughly **42.5% from escaping random init, 57.5% from the
objective**:

```
0 -> 1 epoch   buys +0.0854
1 -> 10 epochs buys +0.1158
```

## Claim 3, SHAPE - one resolved step, and the rest are UNRESOLVED rather than flat

A shape word is emitted only where a contrast is significant. Read against the policy family's
Bonferroni threshold of 0.01 (0.05 / 5).

| contrast | delta | p | verdict |
|---|---|---|---|
| e1 -> e2 | **+0.0629** | **0.0068** | **rises** |
| e2 -> e3 | +0.0262 | 0.1123 | indistinguishable |
| e3 -> e5 | +0.0017 | 0.9478 | indistinguishable |
| e5 -> e10 | +0.0250 | 0.3994 | indistinguishable |

> [!warning] THREE "INDISTINGUISHABLE" VERDICTS ARE THE PREDICTED OUTPUT AND ARE NOT EVIDENCE OF
> FLATNESS.
> The spec pre-registered this as **badly powered on purpose-of-record**: about 28% power for
> Claim 1's own effect size and about 10% at Bonferroni for an adjacent step that size. The
> aggregator prints that paragraph above the numbers so a null cannot be read as a plateau. **This
> was stated before the numbers existed precisely so it could not be invented afterwards.**
>
> `e1 -> e2` is the single largest resolved step anywhere in the 0-to-80 curve after the jump off
> zero, at +0.0629. Whether the curve keeps climbing after that is **not resolved by n=12**.

## Claim 4 - `S` rises through the window and saturates by 10

`S` is measured far more precisely than policy: within-arm sd is 0.0019 to 0.0057, against policy's
0.0441 to 0.1026. A small `S` delta can therefore be significant while a much larger policy delta
is not, and that is a difference in instrument resolution, not a difference in the world.

Own family, Bonferroni 0.0125 (0.05 / 4):

| contrast | delta | p | verdict |
|---|---|---|---|
| e1 -> e2 | +0.0052 | **0.0005** | rises |
| e2 -> e3 | +0.0043 | **0.0039** | rises |
| e3 -> e5 | +0.0058 | **0.0005** | rises |
| e5 -> e10 | +0.0028 | 0.0239 | indistinguishable (above 0.0125) |

`S_cross` tracks `S` at every arm and the `level` control rises with it, so this is not the
within-shell clustering term moving alone. That is EXP-054's amendment applied here.

**NO DISSOCIATION IS LICENSED BY COMPARING THESE TWO TABLES.** `S` resolves three rising steps and
policy resolves one, but policy's three nulls are underpowered rather than measured-flat. Calling
that a dissociation would compare a significant result against an underpowered null, which is the
error this project has made repeatedly. Both instruments are consistent with saturation somewhere
around 5 to 10 epochs, and neither resolves exactly where.

## The dissociation that IS resolved, at the 0-to-1 transition

**Descriptive, and outside the pre-registered Claim 4 family.** It is reported because it is the
comparison the design made possible for the first time, not because it was pre-registered.

| instrument | at 0 epochs | at 1 epoch | change | |
|---|---|---|---|---|
| `S` | 0.0100 | 0.0061 | **-0.0038** | **11/12 seeds DOWN, p 0.0010** |
| policy | 0.0000 | 0.0854 | **+0.0854** | **12/12 seeds UP from an exact zero** |

**Both sides are resolved, which is what makes this different from the comparison above.** The `S`
side is a significant paired contrast; the policy side moves off a floor with literally zero
variance where every seed improves.

**This is an exact before-and-after on the same network, not two random draws.** EXP-054 rebuilds
E0 as `make_sensory(seed)`, and `pretrain_one` passes no `sensory=`, so `train_inverse_model` builds
`make_sensory(cfg.seed)` itself from the same seed. Verified rather than reasoned: recomputing `S`
from `make_sensory(seed)` under EXP-052's `PRETRAIN` config reproduces EXP-054's stored E0 records
to within 1e-9 on seeds 0, 1 and 2.

```
seed 0: recomputed +0.011564   EXP-054 stored +0.011564   MATCH
seed 1: recomputed +0.006799   EXP-054 stored +0.006799   MATCH
seed 2: recomputed +0.008882   EXP-054 stored +0.008882   MATCH
```

Move accuracy at one epoch is **0.2494 against a 1/6 = 0.1667 chance floor**, so `e1` is not an
encoder that failed to train. The network is demonstrably learning the task it was given, its
distance structure gets *worse than random* while it does so, and the policy built on it improves
enormously.

**So `S` moves opposite to policy across the 0-to-1 transition.** EXP-054 established that `S` does
not track policy across 10 to 80. This is stronger and at higher resolution: over the one interval
where policy makes its largest move, `S` goes the other way. **`S` should not be used as a policy
predictor**, and this is now a measurement rather than the "unevaluated" status EXP-054 left it in.

## What this changes

1. **The left edge is where the action is, and it is not a single jump.** Two resolved steps live
   inside 0 to 2 epochs: +0.0854 off the floor, then +0.0629 more. Everything from 3 to 40 is
   unresolved at n=12, and 80 is a collapse.
2. **Pretraining's value is not mostly "stop being randomly initialised".** 42.5% of it is. The
   objective supplies the rest.
3. **`S` is refuted as a policy predictor**, not merely unevaluated. Successor specs must not cite
   it as a diagnostic, and must not cite EXP-054's Claim 4 as a clearance either: that verdict now
   correctly prints UNEVALUATED.
4. **The cheapest useful encoder in this project is 1 to 2 epochs, not 10 and certainly not 40.**
   `e2` reaches 0.1483, 74% of `e10`, for a fifth of the pretraining cost. EXP-052 cut 40 to 10;
   this cuts the interesting range further, subject to the power caveat on Claim 3.

## What is NOT claimed

- **Not that the curve is flat from 3 to 10.** Three indistinguishable contrasts at ~10% power are
  unresolved, and the spec said so in advance.
- **Not a dissociation from the Claim 4 tables.** Only the 0-to-1 transition supports one.
- **Not that `e2` is optimal.** `e2 -> e3` and `e3 -> e5` are unresolved; a bigger n could separate
  them. What is established is that the range worth searching is 1 to 5, not 10 to 80.
- **Nothing about depth 7 or about compounding.** The encoder line does not compound (EXP-049) and
  its advantage erodes with depth (EXP-051). This is a depth-6 result.

## Compute

Phase 3 ran **18.0 h wall clock** for 48 cells at 8 workers, about **2h45m per cell**.

**8 workers bought almost nothing over the spec's 6.** The pre-registration estimated 17.2 h at 6
workers; 8 delivered 18.0 h. Per-cell time rises with contention about as fast as the extra workers
add throughput, which is the same effect the playbook documents in "fewer workers is faster per
cell". **Choose the worker count from memory headroom and stop expecting wall-clock returns.** The
choice of 8 here was correct for a different reason: RL workers measured 1.05 GB private each, so
10 would have put system commit at 34.0 GB against the playbook's 32.7 GB half-limit line.

Pretraining at 8 workers ran at **86.7 s per epoch per seed** against 87.3 at 10 and 24.9 at 2, so
the pretraining contention knee is **below 8**, not between 8 and 10.
