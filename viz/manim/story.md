# The visual story: from 2.2% to a moving break point

A three-act arc for explaining this project, built **only** from measured results. Every number
below is in a committed `RESULTS.md`. Nothing is illustrative.

The through-line: **a spiking network that started barely above chance now solves cubes six
moves deep, and neither thing that moved it was more compute.**

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

Two cube nets, same scramble, side by side. **Left: EXP-036 seed 3 plays R fifteen times and
never stops** (modal action fraction **1.000**, success **0.0000**; four of its twelve seeds do
exactly this, and EXP-031 found 7 of 12 at depth 2). **Right: EXP-043 seed 11 solves it in
eight** - `U U U R U' R' U F'`.

**The pairing crosses experiments, and the scene says so on screen.** EXP-036 has no working seed
at depth 6 to pair against - all twelve score 0.0000 - so a same-experiment version of this shot
does not exist at this depth, and depth 3 has no fully collapsed seed (modal tops out at 0.793).

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

## Act 3 — "Two levers, and they compound" (EXP-039/040, EXP-041/042/043)

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

### Scene 5b — `TheDepth1Trap`
The second lever, and the one that needs explaining rather than showing. `max_steps_for(d) = 2d+3`
gave depth 1 **five** steps where optimal is **one**. A cube face has **order 4**, so a repeated
move either inverts a one-move scramble (1 step) or cycles all the way back to solved (3 steps).

**A constant-action policy therefore scored 0.3333 at depth 1, against a random policy's 0.2208.**
Curriculum stage 1 was paying more for the worst possible policy than for exploring. Capping the
depth-1 **training** budget at 2 steps flips it to 0.1667, below random. Nothing else changed.

**The refutation that came with it:** EXP-042 also tried *deleting* the depth-1 stage. That scored
0.3565 against the cap's 0.5351. **Reprice it, do not remove it.**

### Scene 6 — `TheBreakPointMoves`
The payoff. Same 390-parameter head throughout, two levers, and they compound:

| depth | frozen (EXP-036) | + trained encoder (EXP-040) | **+ depth-1 cap** |
|---|---|---|---|
| 4 | 0.1591 | 0.3471 | **0.5351** (EXP-042) |
| 5 | 0.0396 (broken) | 0.2304 | **0.3412** (EXP-043) |
| 6 | 0.0000 (broken) | 0.1037 (at the bar) | **0.1800 working** (EXP-043) |

Depth 5 had been broken since it was first measured, and neither curriculum tuning nor trainer
stabilizers moved it. **Depth 6 was 0.0000 on all twelve seeds.** The break point is now **past
depth 6 and unmeasured** - the first time in the project its location is genuinely unknown.

**Neither lever alone reaches depth 6**, which is why this shot has three bars and not two.

**Keep it honest on screen, both ways:**
- Depth 6 is *working* by EXP-036's own pre-registered rule, and this time with room:
  **+2.81 SE** of margin with **10 of 12** seeds above the bar, against EXP-040's +0.11 SE and
  5 of 12. That rule's extra conditions were written precisely because 0.1037 cleared the bare
  `>= 0.10` on noise.
- Depth 5's **+0.1108 is REFUTED** - `p = 0.0815` against a pre-registered 0.05, with four seeds
  regressing. It is the larger effect and it still misses. Say *"large but not established at
  n=12"*, and do **not** say the cap fails at depth 5; it is a p-value miss, not a demonstrated
  absence.

### Scene 7 — `WhereWeStarted`
Close the loop. EXP-029's opening table dissolving into today's:

> **depth 4: 0.0% -> 53.5%. depth 6: did not exist -> 18.0%.**
> Same 390 trainable parameters. The encoder changed and the reward budget changed, not the head.

---

## Honest framing notes for narration

These keep the video credible, and they are all in the write-ups:

- **Still 0.32% of the cube.** Depths 3-6 out of 14. A random scramble is at depth 11, where 97.5%
  of the state space lives, and remains out of reach. Do not imply the cube is solved.
- **The wins came from refutations.** *Six* things were closed before the *two* that worked: width,
  volume alone, curriculum stage weighting, starvation at depth 6, trainer stabilizers, and
  deleting the depth-1 stage. That is the actual method and it is the more interesting story.
- **The "powerful but unreliable" caveat has mostly, not entirely, gone.** EXP-040's two dead seeds
  at depth 5 were largely the depth-1 trap: zeros went 2/12 -> 0/12 at depth 5 and 3/12 -> 1/12 at
  depth 6, and the spread narrowed at every depth. But the halving of variance seen at depth 4
  (sd 0.2242 -> 0.1012) does **not** repeat at 5 and 6 (0.80, 0.82), so some genuine seed
  sensitivity remains deeper. Report the counts descriptively: **n=12 cannot show a count went to
  zero**, and Fisher's exact on 2/12 against 0/12 is ~0.48.
- **Pre-registration did real work.** EXP-036's gap came back *inconclusive* against a bar set in
  advance, and was not upgraded. EXP-037's aggregator printed a true-but-misleading verdict and
  was corrected without moving a threshold.

---

## Render plan

| scene | form | data source | built? |
|---|---|---|---|
| `TheBreakPointMoves` | grouped bars | EXP-036 + 040 + 042/043 records | **rendered** |
| `ScaleOfTheCube` | one stacked bar | `STATE_CENSUS`, published | **rendered** |
| `TheCurriculumUnlock` | line chart, log x | EXP-034/035 records | **rendered** |
| `TheWall` | table | EXP-033, published | **rendered** |
| `TheEncoderLearns` | line chart | EXP-039 records | **rendered** |
| `CollapseIsASymptom` | table | EXP-036 + 038 records | **rendered** |
| `WhereWeStarted` | table morph | EXP-029 published + today | **rendered** |
| `PolicyCollapse` | two cube nets, animated | **recorded rollouts** + EXP-038 | **rendered** |
| `FullStory` | all of the above, in order | - | **rendered** |

Three of the six are charts and three are tables, which is deliberate: the two levers and the
curriculum climb are **trends**, so they are lines and bars, while the probe ceiling and the
collapse sweep are **comparisons at a glance**, so they are tables. Two bar charts back to back
read as one slide.

**Hardware:** everything renders on the laptop, at any quality. **Not this VPS at all**: manim
cannot import there. `scripts/laptop/render_story.ps1` drives it and pulls stills; `README.md` has
the setup, which is already done.

**Status 2026-08-15: every scene in this document is built and rendered at 1080p**, and
`FullStory` assembles them with act cards and fades - see `README.md` for how that composes and
what still belongs in an editor.

`PolicyCollapse` landed last and replaces `CollapseIsASymptom` in the assembled cut: the two land
the same punchline from the same records, one as two cubes and one as a table, and back to back
the second is the first repeated. The table stays a first-class scene for anywhere the numbers
matter more than the picture.

**Its frames are replayed from real rollouts, not drawn.** `record_traces.py` loads the two head
checkpoints and rolls them out from one shared held-out scramble; `traces.json` was verified
frame by frame against `apply_move`, and its solved flags against `is_solved`, before being
committed.

**Records do not travel with the repo.** They are gitignored, so each machine holds only what it
ran, and a scene whose experiment is missing now fails by name rather than as a `NoneType` deep
in a coordinate function. The laptop needed EXP-039 copied to it; the VPS still has no EXP-029 or
EXP-030, which **the laptop does** (265 and 144 records) if `WhereWeStarted` ever wants per-seed
detail.
