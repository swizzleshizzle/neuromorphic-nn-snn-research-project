# The visual story: from 2.2% to a moving break point

A three-act arc for explaining this project, built **only** from measured results. Every number
below is in a committed `RESULTS.md`. Nothing is illustrative.

The through-line: **a spiking network that started barely above chance now solves cubes four
moves deep, and the thing that finally moved it was not more compute.**

---

## Act 1 — "It barely works" (EXP-029)

The origin point. Five brain regions, a `Linear(64 -> 6)` head, **390 trainable parameters**, and
a frozen randomly-initialised encoder.

| depth | regionalized | random floor | verdict |
|---|---|---|---|
| 1 | 87.5% | 20.8% | works |
| 2 | 38.0% | 4.3% | works |
| 3 | **2.2%** | 1.4% | barely above chance |
| 4 | **0.0%** | 0.3% | nothing |

**The hook:** depth 3 is 120 states out of 3,674,160. A random scramble sits at **depth 11**.

### Scene 1 — `ScaleOfTheCube`
Shells expanding outward from solved. Depths 1-3 are **153 states, 0.004%** of the cube; depths
10-12 hold **3,063,976 — 97.5%**. Land the line the vault already makes: *we are not a quarter of
the way to a random cube, we are at the shallow edge.*

**Why it opens the video:** it sets the difficulty honestly, so every later gain reads as real
rather than as a demo.

---

## Act 2 — "One thing works, and four things do not"

### Scene 2 — `TheCurriculumUnlock`
Depth 3, one line climbing, four points, all measured:

| | success |
|---|---|
| EXP-029 baseline | 0.022 |
| EXP-034 curriculum, 600 ep | 0.097 |
| EXP-034 curriculum, 3,000 ep | 0.256 |
| EXP-035, 10,000 ep | 0.397 |
| EXP-035, 30,000 ep | **0.500** |

**The control that makes it honest:** direct training at 3,000 episodes scores **0.019** — *worse
than the 600-episode baseline*. Same compute, no curriculum, no gain. Show both lines. The
curriculum is the lever, not the compute.

### Scene 3 — `PolicyCollapse`
The most visceral shot available, and it needs no chart.

Two cubes, same scramble, side by side. Left: a collapsed policy playing **one move, over and
over** (modal action fraction **1.000** against a uniform floor of 0.354; EXP-031 found 7 of 12
seeds doing exactly this). Right: a working policy solving it.

**The twist that makes it a real finding:** EXP-032 *fixed* the collapse and success got
**worse** (0.380 -> 0.117 at depth 2). EXP-038 repeated that at depth 6 where the diagnosis was
strongest: modal fell 0.975 -> 0.631 and bought **nothing**. **Collapse is a symptom, not the
cause** — a genuinely counter-intuitive beat.

### Scene 4 — `TheWall`
Why it gets hard. A linear probe for "which move reduces distance-to-solved", read off the raw
observation:

| depth | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| best linear probe | 1.000 | 1.000 | 0.956 | 0.766 | 0.598 |

Extrapolating, it reaches chance (0.19) around depth 8-9. **A linear head cannot solve deep
cubes regardless of how good the encoder is** — not "will not", cannot.

And the frozen concept the policy actually reads is *worse* than the raw pixels: **0.459 at depth
4 against 0.766**. Widening it 8x reaches only 0.638 — refuted as a route.

**Act 2 closes on four refutations:** width, volume alone, curriculum weighting, trainer
stabilizers. Everything cheap inside the architecture, measured and closed.

---

## Act 3 — "Train the encoder" (EXP-039, EXP-040)

### Scene 5 — `TheEncoderLearns`
Self-supervised pretraining: predict which move was played between two cube states. No labels,
no oracle. Then re-measure the probe:

| depth | frozen | **trained** | raw-facelet ceiling |
|---|---|---|---|
| 3 | 0.614 | 0.908 | 0.906 |
| 4 | 0.447 | **0.786** | 0.742 |
| 5 | 0.406 | **0.660** | 0.618 |
| 6 | 0.344 | **0.575** | 0.488 |

All at p 0.0005, every seed. **The trained encoder beats the ceiling on the raw observation** —
the first time the spiking network does work rather than acting as a fixed random projection.

**The detail worth animating:** at depth 3 it merely *matches* the ceiling (+0.003). The margin
grows with depth: +0.044, +0.042, **+0.087**. It helps most exactly where the observation fails.

### Scene 6 — `TheBreakPointMoves`
The payoff. Same 390-parameter head, one thing changed:

| depth | before (EXP-036) | **after (EXP-040)** |
|---|---|---|
| 4 | 0.1591 | **0.3471** |
| 5 | 0.0396 (broken) | **0.2304** (working) |
| 6 | 0.0000 | **0.1037** |

Depth 5 had been broken since it was first measured, and neither curriculum tuning nor trainer
stabilizers moved it. **Depth 6 was 0.0000 on all twelve seeds.**

**Keep it honest on screen:** depth 6 clears its "working" bar by **0.11 standard errors** with
5 of 12 seeds above it. Say *off the floor*, not *working*.

### Scene 7 — `WhereWeStarted`
Close the loop. EXP-029's opening table dissolving into EXP-040's:

> **depth 4: 0.0% -> 34.7%. depth 3: 2.2% -> 50%.**
> Same 390 trainable parameters. The encoder changed, not the head.

---

## Honest framing notes for narration

These keep the video credible, and they are all in the write-ups:

- **Still 0.02% of the cube.** Depths 3-4 out of 14. A random scramble is at depth 11 and remains
  out of reach. Do not imply the cube is solved.
- **The wins came from refutations.** Four things were closed before the one that worked was
  found. That is the actual method and it is the more interesting story.
- **Two of twelve seeds fail completely** with the pretrained encoder (0.000 where the baseline
  managed 0.278). The lever is powerful and unreliable, and the pretraining metric does not
  predict which seeds fail. Currently under diagnosis.
- **Pre-registration did real work.** EXP-036's gap came back *inconclusive* against a bar set in
  advance, and was not upgraded. EXP-037's aggregator printed a true-but-misleading verdict and
  was corrected without moving a threshold.

---

## Render plan

| scene | needs | data source | priority |
|---|---|---|---|
| `PolicyCollapse` | cube renderer, no LaTeX | EXP-031/032/038 records | **first** |
| `TheBreakPointMoves` | bars | EXP-036 + EXP-040 records (local) | **first** |
| `ScaleOfTheCube` | shells | BFS table, computed live | second |
| `TheCurriculumUnlock` | line chart | EXP-034/035 records (local) | second |
| `TheWall` | line chart | EXP-033 records (local) | third |
| `TheEncoderLearns` | grouped bars | EXP-039 records (local) | third |
| `WhereWeStarted` | table morph | EXP-029 published + EXP-040 | last |

**Hardware:** iterate at `-ql` (480p) anywhere; final renders at `-qh`/`-qk` want the laptop's 22
cores or the desktop. **Not this VPS** — 2 cores, 4 GB, and one worker is the standing limit.
