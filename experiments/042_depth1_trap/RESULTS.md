# EXP-042 Results - repricing depth 1 fixes the failures AND helps every seed

> **Why this file exists:** the standing habit from the 2026-07-13 Phase-2 audit. The
> interpretation contract was committed at `820680c` **before the data existed**, and
> `aggregate.py` at `d831c30` **while the run was at 34/36 and no arm was complete**. No
> threshold was edited while filling this in; one broken diagnostic was fixed and is disclosed
> below.
>
> **Provenance:** 36 records (3 arms x 12 seeds), depth 4, on EXP-040's pretrained encoders,
> 10,000 episodes, `entropy_beta=0.0`, `normalize_advantages=False`. Run 2026-08-12 18:55 to
> 2026-08-13 03:21 on the laptop `SwizzlesDuo`, `--workers 10`, **8 h 26 m**, zero tracebacks.
> The laptop slept mid-run (accidental power button); Windows suspends rather than kills, so the
> run rode through it - worker start times and CPU accumulation confirm no restart.

## Headline

**Capping the depth-1 training budget at 2 steps is the largest single improvement the cube
line has produced.**

| arm | mean | sd | min | seeds at 0.000 | final-stage entropy |
|---|---|---|---|---|---|
| baseline | 0.3471 | 0.2242 | 0.000 | **2/12** | 0.3034 |
| **capped** | **0.5351** | **0.1012** | **0.241** | **0/12** | **0.4367** |
| skipped | 0.3565 | 0.1958 | 0.000 | 1/12 | 0.3197 |

**Depth 4 now scores 0.5351** - higher than depth 3's best-ever 0.500 (EXP-035 at 30,000
episodes, three times the budget). The **variance more than halved** (sd 0.2242 -> 0.1012) and
the **worst seed went from 0.000 to 0.241**.

## Claim 1 (PRIMARY) - entropy entering the final stage

Pre-registered: **>= +0.05** at **p <= 0.05**, paired against baseline.

| arm | delta | exact p | seeds moved | share from seeds 2,4 | mean **excluding** 2,4 | verdict |
|---|---|---|---|---|---|---|
| **capped** | **+0.1332** | **0.0054** | 12/12 | 46% | **+0.0754** | **CONFIRMED** |
| skipped | +0.0163 | 0.7954 | 12/12 | 44% | **-0.0496** | **REFUTED** |

**The `excluding` column is what makes this readable.** Capping raises entropy by +0.0754 even
after removing the two seeds it rescued - it is a broad effect, not a two-seed repair. Skipping
*lowers* it by 0.0496 on the other ten: it helps the failures slightly and **hurts everyone
else**.

## The mechanism, confirmed seed by seed

EXP-041 predicted the failure comes from depth 1 paying a constant-action policy 1/3. Capping
the budget at 2 admits the inverse and not the order-4 cycle, dropping that to 1/6 - **below** a
random policy's 0.2208.

Stage-1 exit entropy and depth-1 training success:

| seed | baseline entropy (solved) | capped entropy (solved) |
|---|---|---|
| **2** | **0.0053** (0.331) | **0.3498** (0.537) |
| **4** | **0.0019** (0.323) | **0.4419** (0.558) |
| 0 | 0.2219 (0.554) | 0.3030 (0.724) |
| 1 | 0.2445 (0.766) | 0.2542 (0.711) |

The baseline's failing seeds sit at **0.331 and 0.323** - the 1/3 signature of a constant action,
to three decimals. Under the cap they reach **0.537 and 0.558**, and their entropy survives by
two orders of magnitude.

**No capped seed is anywhere near the degenerate rate.** Depth-1 training success across all
twelve: `0.724 0.711 0.537 0.692 0.558 0.760 0.447 0.663 0.724 0.640 0.763 0.813` - every one
far above the 0.167 a constant policy could now achieve. **The attractor is gone, not avoided.**

## Claim 2 (SECONDARY, descriptive - no p-value by design)

| arm | seeds at 0.000 |
|---|---|
| baseline | 2/12 |
| **capped** | **0/12** |
| skipped | 1/12 |

> [!warning] This is suggestive and no more, exactly as pre-registered
> A 2/12 -> 0/12 drop **cannot** be shown significant at n=12: Fisher's exact gives p ~ 0.48.
> Confirming the failure rate is actually zero needs roughly 40+ seeds per arm. **The reason to
> believe the fix is Claim 1 and the mechanism, not this count.**

Note the skipped arm **created a new failure**: seed 3 went 0.256 -> 0.000, while rescuing seeds
2 and 4. Its 1/12 is a different seed failing, not the same two partially fixed.

## Claim 3 (COST) - among the ten seeds that never failed

| arm | delta | verdict |
|---|---|---|
| **capped** | **+0.1195** | no cost; it **helps** the healthy seeds too |
| skipped | **-0.0744** | **COST FLAGGED** |

**This is the most important secondary result.** Capping was expected, at best, to remove a
failure mode. It instead improved the seeds that were already working by +0.12, which says the
depth-1 trap was **degrading every run**, not only the two that collapsed visibly.

Skipping depth 1 pays the cost EXP-037 predicted: the bootstrap stages do real work, and
deleting one is worse than repricing it.

## Claim 4 (mechanism check) - deterministic, enumerated

| arm | depth-1 constant-action reward | vs random 0.2208 |
|---|---|---|
| baseline | 0.3333 | degeneracy **wins** |
| **capped** | **0.1667** | degeneracy **loses** |
| skipped | n/a | no depth-1 stage |

Pinned in `tests/training/test_encoder_seam.py`, which fails if the cap is ever loosened to 3 -
the value that silently restores the exploit.

## A broken diagnostic in the aggregator, disclosed

> [!bug] The first run printed "354% of the effect", which is impossible
> The share-of-effect diagnostic divided by `abs(sum(diffs))`. When an arm's net effect is near
> zero because seeds cancel, that denominator is tiny and the ratio is unbounded - so it
> exceeded 100% and, worse, **tripped the `share > 0.8` branch and labelled the skipped arm
> UNDERPOWERED BY CONSTRUCTION when it should have read REFUTED.**
>
> Fixed to divide by total **absolute** movement, which is bounded by 1, and to also report the
> mean excluding the two failed seeds - the directly interpretable quantity, and the one that
> turned out to carry the real story for both arms.
>
> **No threshold moved.** This was a broken instrument producing a wrong label from correct
> arithmetic, which is the same shape as EXP-037's "INTERIOR OPTIMUM". The impossible number is
> what exposed it; a share of 0.9 would have passed unnoticed.

## Limitations

- **Depth 4 only.** Depths 5 and 6, where the same seeds also failed, are untested under the cap
  and are the obvious next run.
- **One cap value (2).** A budget of 1 is untested and would also remove the exploit.
- **Pretrained encoders only.** The trap predates pretraining but only fires when learning is
  fast enough to reach the attractor; a frozen-encoder arm is untested.
- **The failure rate is not shown to be zero** - see Claim 2.
- **The bootstrap stages remain unexamined at depths 2-3.** Their constant-action rates (0.037
  and 0.000) are below random, so no trap exists there, but their budgets are equally unchosen.

## Lead for the next experiment

1. **Re-run depths 5 and 6 with the cap.** EXP-040 left depth 5 at 0.2304 and depth 6 at 0.1037
   *with the trap still in place*. If capping is worth +0.19 at depth 4, the break point may move
   again.
2. **Revisit whether the 2d+3 rule is right anywhere.** It was never chosen; it was a generous
   default, and at depth 1 it was actively harmful. Depths 2-3 are safe by enumeration, but
   "safe" is not "optimal".
3. **The variance collapse deserves its own look.** sd 0.2242 -> 0.1012 means the pretrained
   encoder's unreliability (EXP-040's headline caveat) was **largely the trap**, not the encoder.

**Still refuted and CLOSED:** width (EXP-033), volume alone (EXP-034), curriculum stage weighting
(EXP-037), starvation at depth 6 (EXP-037), trainer stabilizers (EXP-038), and now **removing the
depth-1 stage** (this experiment - reprice it, do not delete it).

## Regenerate

```bash
.venv/bin/python -u experiments/042_depth1_trap/run.py \
    --seeds 0 1 2 3 4 5 6 7 8 9 10 11 --workers 10 --skip-existing
.venv/bin/python experiments/042_depth1_trap/aggregate.py
```
