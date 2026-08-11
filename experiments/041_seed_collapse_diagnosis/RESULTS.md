# EXP-041 - why 2 of 12 seeds collapse: depth 1 pays 33% for the worst possible policy

> **Why this file exists:** the standing habit from the 2026-07-13 Phase-2 audit. This is a
> **diagnosis**, not a pre-registered experiment - there is no interpretation contract, because
> the question was "what causes an observed failure", not "does an intervention work". Every
> number below is measured; the mechanism is verified by exhaustive enumeration.
>
> **Provenance:** four depth-4 cells (seeds 2, 4, 0, 1) at EXP-040's exact configuration, run
> 2026-08-10/11 on the **VPS** at 1 worker, plus exhaustive enumeration over the depth-1..4
> shells. Records in `diag_summary.json`. Uses the `stage_trace` telemetry added at `8a640b1`.

## The question

EXP-040 tripled mean success but left **2 of 12 seeds at exactly 0.000**, where EXP-036's
frozen-encoder baseline had none. Their pretraining looked entirely normal (move-naming accuracy
0.434 and 0.433, inside the 0.430-0.437 band every seed produced).

## What was eliminated first

| hypothesis | verdict | evidence |
|---|---|---|
| the encoder is degenerate | **NO** | seeds 2/4 sit mid-pack on mean rate, dead units and weight scale, and **seed 2 has the HIGHEST across-state discrimination of all twelve** (0.207) |
| pretraining saturates the logits | **NO** | before training, every seed sits at **96-99%** of the log-6 entropy ceiling |
| it is a generalisation failure | **NO** | train-side success is also **0.000** |

Encoder quality does not even correlate with policy success: seed 8, the **best** performer at
0.669, has the **lowest** across-state sd.

## Where the collapse happens

Per-curriculum-stage entropy, from `stage_trace`:

| seed | | stage 1 (d1) | stage 2 (d2) | stage 3 (d3) | stage 4 (d4) |
|---|---|---|---|---|---|
| **2** | entropy start -> end | 0.616 -> **0.0054** | 0.0009 | 0.0010 | 0.0002 |
| | train solved | **0.331** | 0.033 | 0.000 | 0.000 |
| **4** | entropy start -> end | 0.751 -> **0.0064** | 0.0027 | 0.0018 | 0.0012 |
| | train solved | **0.322** | 0.032 | 0.000 | 0.000 |
| 0 | entropy start -> end | 0.993 -> 0.0895 | **0.399** | 0.365 | 0.412 |
| | train solved | 0.610 | 0.663 | 0.562 | 0.396 |
| 1 | entropy start -> end | 1.191 -> 0.2275 | **0.394** | 0.367 | 0.489 |
| | train solved | 0.740 | 0.726 | 0.662 | 0.478 |

**The policy dies in curriculum stage 1 and never recovers.** Working seeds' entropy *rebounds*
in stage 2 (0.09 -> 0.399); failing seeds' does not (0.005 -> 0.0009).

## The mechanism, verified by enumeration

The failing seeds solve **0.331** and **0.322** of depth-1 training episodes. That is 1/3, and
it is exactly what a constant-action policy scores:

```
depth-1 shell: 6 states, step budget 2d+3 = 5
always move 0: 2/6     always move 1: 2/6     always move 2: 2/6
always move 3: 2/6     always move 4: 2/6     always move 5: 2/6
mean over all six constant policies: 0.3333
```

**Why:** a face move has **order 4** on a 2x2. From a state scrambled by move `m`, repeating any
move `a` either inverts it (`a = m'`, 1 step) or cycles back to solved (`a = m`, 3 more steps,
since `m**4 = identity`). Distances after repeating one move from a depth-1 state:
**`[2, 1, 0, 1, 2]`**. The `2d+3 = 5` budget is generous enough for both paths.

### And depth 1 is the ONLY stage where degeneracy beats exploring

| depth | budget | constant-action | uniform-random | degenerate policy is... |
|---|---|---|---|---|
| **1** | 5 | **0.3333** | 0.2208 | **BETTER (1.5x)** |
| 2 | 7 | 0.0370 | 0.0509 | worse |
| 3 | 9 | 0.0000 | 0.0104 | strictly zero |
| 4 | 11 | 0.0000 | 0.0024 | strictly zero |

**Curriculum stage 1 positively selects for the worst possible policy.** It pays a constant
action 33% - half again what exploration earns - and that reward is enough to reinforce the
collapse before any later stage can punish it. By stage 2 the entropy is 0.001 and there is no
exploration left to recover with.

## Why this only surfaced now

The trap has been in every curriculum run since EXP-034; the curriculum has always started at
depth 1. It did not fire with a **frozen random** encoder because learning there is slow enough
that entropy survives stage 1 (EXP-036 depth 4: 0 seeds at zero).

**The pretrained encoder sharpens the gradient, so stage 1 converges faster - toward whichever
attractor it reaches first.** That is the same mechanism behind both EXP-040 tails: the mean
tripled *and* 2 of 12 seeds fell into the degenerate attractor. It is not a defect of the
encoder; it is a pre-existing reward-shaping trap that a stronger learner finds sooner.

## Two process notes

> [!bug] The reproduction check was mis-designed, and got lucky
> It compared VPS results against EXP-040's **laptop** results. `CLAUDE.md` states plainly that
> **seeded runs are not reproducible across platforms**; only byte-identity within a machine
> holds. So seeds 0 and 1 "DIVERGED" (0.4962 vs 0.188 and 0.526) by construction, not by defect.
>
> It accidentally produced **stronger** evidence than it was designed for: the failing seeds
> still scored exactly 0.000 on a different platform with an entirely different random stream.
> The failure is robust to the whole RNG changing, which exact reproduction would not have shown.
>
> Seeds 0 and 1 both landing on 0.4962 is 66/133 twice - a genuine coincidence at this variance
> (EXP-040's depth-4 spread is 0.000-0.669), but worth re-checking if it ever recurs.

> [!note] The hypothesis going in was wrong, and the trace refuted it
> The pre-stated guess was that the pretrained encoder made depths 1-2 **too easy**, causing
> premature convergence. The opposite: failing seeds solved **fewer** depth-1 episodes (0.33)
> than working ones (0.61-0.74). They did not master depth 1 - they collapsed at it.

## Candidate fixes, none applied

Root cause only; no fix has been implemented or tested.

1. **Tighten the depth-1 step budget.** `max_steps_for(d) = 2d+3` gives 5 where optimal is 1. A
   budget of 2 admits the inverse (1 step) but not the cycle (3 steps), dropping a constant
   policy to 1/6 = 0.167, **below** random's 0.221. Cleanest incentive flip, but
   `max_steps_for` is used by every cube experiment and changing it breaks comparability with
   all of them.
2. **Start the curriculum at depth 2.** No trainer change and no neutrality problem, but loses
   the depth-1 bootstrap, and EXP-037 showed the early stages do more work than expected.
3. **An entropy floor during stage 1 only.** Targeted at the exact failure. Note EXP-038 closed
   entropy bonuses as a lever *at depth 6, for a different purpose*; this is a different claim.
4. **Detect and restart** a collapsed stage 1. Pragmatic, ugly, and does not fix the trap.

**Whichever is chosen needs a pre-registered arm against EXP-040's 2/12 failure rate**, because
the effect being fixed shows up in 17% of seeds and n=12 is thin for that.

## Regenerate

```bash
# 1 WORKER on the VPS. ~2 h per seed; four seeds sequential.
.venv/bin/python -u experiments/041_seed_collapse_diagnosis/run.py --workers 1 --out-dir <dir>
```
