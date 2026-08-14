# EXP-044 design - where is the break point now? Depth 7.

**Status: pre-registration. Written 2026-08-14, before any EXP-044 number exists.**
Thresholds are fixed. If one is edited after data arrives, that edit is the finding.

## 1. Why

**For the first time since EXP-036, the location of the break point is unknown.** It was depth 5
from EXP-036 until 2026-08-13, unmoved by EXP-037 (curriculum weighting) and EXP-038 (trainer
stabilizers). Two levers moved it, and they compound: the self-supervised encoder (EXP-039/040)
and capping the depth-1 training budget (EXP-041/042/043).

| depth | frozen (EXP-036) | + trained encoder (EXP-040) | + depth-1 cap |
|---|---|---|---|
| 4 | 0.1591 | 0.3471 | **0.5351** (EXP-042) |
| 5 | 0.0396 broken | 0.2304 | **0.3412** (EXP-043) |
| 6 | 0.0000 broken | 0.1037 at the bar | **0.1800 WORKING** (EXP-043) |

Depths 3 through 6 all clear the working bar. **Depth 7 has never been attempted.**

This experiment is not a comparison. There is no prior depth-7 arm to pair against, so unlike
EXP-042 and EXP-043 the primary claim is **absolute** - does depth 7 clear the pre-registered
working rule - and **no p-value is attached to it**. The uncertainty is carried by the margin and
seed-count conditions instead. Saying so in advance matters: a paired p appearing in the write-up
would mean a baseline was invented after the fact.

## 2. What the pre-flight already settled

Measured 2026-08-14 on the VPS, so none of it needs designing around:

| | |
|---|---|
| depth-7 shell | **33,058 states** (an earlier handoff guessed ~58,000; it was wrong) |
| `ExactBFSDistance(max_depth=8)` | **0.96 s**, 95 MB resident |
| `heldout_cap` | 200 already binds at depth 6, and equally at depth 7 |
| eval cost | **constant in shell size**; grows only with episode length, 17/15 of depth 6 |
| env steps per run | 105,740 at 10,000 episodes, **1.113x** depth 6 |

`provider = ExactBFSDistance(max_depth=max(cfg.max_depth, cfg.depth, *cfg.curriculum))`, so the
table extends itself to 7 and no config change is needed. The provider is used **only** for shell
enumeration, never for a per-step lookup, so a state beyond the table cannot produce a `None`
mid-episode.

## 3. THE CONFOUND, and why it does not cost 44,000 episodes to handle

The training side grows 3.7x while the episode budget stays at 10,000. Per-state coverage at the
deepest curriculum stage:

| depth | shell | train side | episodes at that stage | **episodes per train state** |
|---|---|---|---|---|
| 5 | 2,256 | 2,056 | 2,000 | **0.973** |
| 6 | 8,969 | 8,769 | 1,670 | **0.190** |
| 7 | 33,058 | 32,858 | 1,432 | **0.044** |

So a failure at depth 7 would confound *"the task is harder"* with *"each state was seen a
quarter as often"*. Matching depth 6's coverage would need **44,000 episodes**, about **52 hours**
of wall clock against arm A's ~12. That is the expensive way to buy the answer.

> [!important] The confound only bites on a NEGATIVE result
> If depth 7 clears the bar at 10,000 episodes, coverage is irrelevant: a success at low coverage
> is still a success, and in fact a **stronger** generalisation result than depth 6's, because at
> 0.044 episodes per state most training states are never visited at all.
>
> The confound therefore does not need paying for up front. It needs a **pre-registered trigger**,
> which is what Claim 2 is.

There is also prior evidence that coverage is not the binding constraint: **depth 5 to depth 6
already dropped coverage 5.1x** (0.973 -> 0.190) and depth 6 works anyway.

## 4. Arms

| arm | what | n | cost |
|---|---|---|---|
| **A (primary)** | depth 7, 10,000 episodes, cap `((1,2),)`, curriculum 1..7 | 12 seeds | ~12 h at 10 workers |
| **F (floor)** | depth 7, `arm="random"` | 12 seeds | **minutes** |
| **B (conditional)** | depth 7, **44,000** episodes, otherwise identical to A | 12 seeds | ~52 h |

Arm A runs on **EXP-040's pretrained encoders**, the same ones EXP-042 and EXP-043 used, so the
only thing that differs from the depth-6 cell is the depth.

**Arm B is NOT dispatched with A.** It runs only if Claim 1 refutes.

Arm F exists because of the standing rule to **measure the chance floor, never assume it** - at
depth 1 it is 21%, not 1/6, because a `2d+3` budget lets a random walk stumble home. `arm="random"`
short-circuits training entirely and only evaluates, so 12 seeds cost minutes.

## 5. The contract

### Claim 1 (PRIMARY) - is depth 7 working?

EXP-036's rule, with the two conditions EXP-043 added after 0.1037 cleared the bare rule on noise:

> **WORKING** if all three hold:
> 1. mean success **>= BAR**, where `BAR = max(0.10, 2 x arm F's measured floor)`
> 2. margin **>= 1.0 SE** above BAR
> 3. **>= 8 of 12** seeds individually above BAR

BAR is written as a formula, not a number, because arm F has not run yet. The 0.10 term is
expected to bind: the floor is 0.0008 at depth 6 and cannot rise with depth.

### Claim 2 - the pre-registered escalation

> If **Claim 1 CONFIRMS**: arm B is **not run**. The frontier is past depth 7 and still unmeasured,
> and the next experiment is depth 8. The write-up must **not** claim the cube is solved: depths
> 1-7 are 0.9% of the state space and a random scramble sits at depth 11.
>
> If **Claim 1 REFUTES**: arm B runs, and its reading is fixed here.
>
> | A | B | conclusion |
> |---|---|---|
> | fails | **works** | the depth-7 failure was **starvation, not difficulty**. The break point is not found; what is found is a budget scaling law, and every depth's number becomes a function of coverage. |
> | fails | fails | **the break point is depth 7 for this recipe.** That is the frontier result this experiment set out to locate. |

### Claim 3 - failure counts, DESCRIPTIVE, no p-value

Seeds at exactly 0.000, reported as a count. **n=12 cannot show a count went to zero**: Fisher's
exact on 2/12 against 0/12 is ~0.48, and a paired permutation where two seeds carry the difference
gives p ~ 0.5 by construction. No test is run on this.

### Claim 4 - variance, DESCRIPTIVE

sd at depth 7 against EXP-043's 0.1277 (depth 5) and 0.0985 (depth 6), continuing that series.
Descriptive only - it is one number per depth and there is nothing to pair.

### Claim 5 - the pre-committed null

If Claim 1 refutes **and** arm B also refutes, that is **not a failed experiment**. It is the
first located break point since EXP-036 and it makes depth 7 the target for the next lever, the
obvious candidate being encoder fine-tuning during RL, which is still untested.

## 6. What would make this wrong

- **A recipe difference other than depth.** Arm A must differ from EXP-043's depth-6 cell in
  `depth` and `curriculum` only. Everything else - encoders, cap, episodes, entropy_beta,
  normalize_advantages, heldout_cap - is copied.
- **Tag collision.** `record_filename` covers tag/arm/depth/seed/sigma and **not** `episodes`, so
  arm B's records would overwrite arm A's if they shared a tag. A carries `exp044_d7_e10000`, B
  carries `exp044_d7_e44000`, and `run.py` fails loudly on any duplicate filename.
- **Reading the records before `aggregate.py` exists.** It is written from this document before
  the run finishes, as in EXP-042 and EXP-043.

## 7. Regenerate

```bash
.venv/bin/python -u experiments/044_depth7_frontier/run.py \
    --seeds 0 1 2 3 4 5 6 7 8 9 10 11 --workers 10 --skip-existing
.venv/bin/python experiments/044_depth7_frontier/aggregate.py
```

EXP-040's `exp040_encoder_s*.pt` must be present; they are gitignored and live on the laptop.
