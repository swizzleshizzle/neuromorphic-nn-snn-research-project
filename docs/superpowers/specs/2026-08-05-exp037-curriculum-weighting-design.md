# EXP-037 Design: does the curriculum's stage weighting matter?

Written 2026-08-05, week 18 session 2, before any number exists.

## Why this experiment exists

`curriculum_schedule` splits the episode budget **equally** across stages, and has since
EXP-034 introduced it. Nobody chose that; it was the obvious default. EXP-036 made the
consequence explicit: at depth 6 the equal split leaves only **1,666 episodes at depth 6
itself**, and EXP-036's own limitations section flagged that as possibly the whole story of its
collapse.

The lever is free relative to buying budget. EXP-035 showed another 3x on episodes costs 5.5 h
per run and the curve had not saturated, so more compute remains available but expensive.
Reallocating compute we already spend costs nothing extra in episodes.

## The axis

One parameter, with an interpretation that survives being read six months later: **the share of
the episode budget spent at the evaluated depth**. The remainder is split equally among the
bootstrap stages.

Exact schedules, computed rather than estimated:

| share | weights | depth-4 schedule (10,000 episodes) | env steps |
|---|---|---|---|
| 12.5% | `(7,7,7,3)` | 2916 / 2916 / 2916 / **1252** | 75,008 |
| **25%** | `(1,1,1,1)` | 2500 / 2500 / 2500 / **2500** | 80,000 |
| 50% | `(1,1,1,3)` | 1666 / 1666 / 1666 / **5002** | 90,008 |
| 75% | `(1,1,1,9)` | 833 / 833 / 833 / **7501** | 100,004 |
| 100% | - | direct training, no curriculum | - |

**25% is already measured: EXP-036's depth-4 cell, 0.1591.** It is not re-run.

**100% is already refuted: EXP-034** showed direct training loses to the curriculum at a matched
budget. That endpoint is what makes this a dose-response with a *predicted interior optimum*
rather than a fishing expedition.

## Pre-registered interpretation contract

Committed before any number exists. Each claim is marked confirmed or refuted in `RESULTS.md`.

All comparisons are **paired per seed against EXP-036's 25% arm**, same twelve seeds, same
machine, with an **exact permutation test over all 2^12 = 4096 sign flips**. No scipy in the
venv and n = 12, so this is exhaustive rather than approximated.

**Claim 1 - is weighting a lever at all?**

- **50% beats 25% by >= 0.03 with p <= 0.05** -> stage weighting is established as a lever, and
  the curriculum's equal split was leaving performance on the table.
- **50% beats 25% by < 0.03, or p > 0.05** -> weighting is refuted as a lever at this budget and
  the equal default stands. **Do not then go hunting for a different share that works.**

The 0.03 bar is set against measured spread, not chosen for convenience: EXP-036's depth-4 arm
had **sd 0.0739** across twelve seeds, so 0.03 is about 0.4 sd, and on a base of 0.1591 it is a
19% relative gain. Anything smaller is not worth reallocating a curriculum for.

**Claim 2 - is there an interior optimum?**

- **50% > 75%** -> interior optimum confirmed, consistent with EXP-034's refuted 100% endpoint.
  The curve turns over somewhere between 50% and 100%.
- **75% >= 50%** -> no turnover in range. More back-loading remains a live lever and the next
  experiment is 85/95%, not more seeds here.

**Claim 3 - the control. 12.5% must be WORSE than 25%.**

This is the arm that stops a positive Claim 1 from being unattributable. If **12.5% >= 25%**,
then performance is not tracking the share at the evaluated depth, and any 50% win needs a
different explanation than the starvation story. **Report it as such rather than keeping the
headline.** The repo's EXP-030 lesson applies directly: ask what a control holds fixed besides
the thing you named.

**Claim 4 - depth 6.** One arm at 50% share (5,000 episodes at depth 6 instead of 1,666) against
EXP-036's 0.0000 on all twelve seeds.

- **Any seed above 0.000** is worth reporting as a signal that depth 6 was partly starved.
- **Clearing the EXP-036 break bar** (twice the measured floor AND 0.10 absolute) would mean
  depth 6 is reachable and the break point moves.
- **Still 0.0000 on all twelve** -> depth 6's failure is not starvation. It is the collapse the
  instruments already showed (modal fraction 0.975), and the next lever there is EXP-031/032
  territory, not the curriculum.

**Claim 5 - instruments.** Report `greedy_modal_action_frac` and `mean_train_entropy` per arm.
EXP-036 found modal fraction climbing monotonically with depth (0.630 / 0.685 / 0.779 / 0.975)
while entropy stayed flat. **If back-loading helps, does it help by reducing collapse?** A gain
that arrives with modal fraction falling is a different and more trustworthy mechanism than one
that arrives with modal fraction unchanged.

**Claim 6 - the confound, disclosed rather than hidden.** See below. Report total environment
steps per arm alongside every success number.

## The episodes-versus-steps confound

**Holding episodes fixed does NOT hold compute fixed.** An episode at depth `d` runs up to
`2d+3` steps, so back-loaded arms spend more environment steps at the same episode budget:

| arm | env steps | vs the 25% arm |
|---|---|---|
| 12.5% | 75,008 | -6.2% |
| 25% | 80,000 | - |
| 50% | 90,008 | +12.5% |
| 75% | 100,004 | +25.0% |

**Episodes are held fixed, deliberately**, because that is the fairness criterion EXP-034
established when it introduced the curriculum and the one EXP-035 and EXP-036 inherited.
Switching currencies mid-line would make this experiment incomparable with the three before it.

**But a 75% arm that wins by 25% more steps has not cleanly won.** The honest reading:

- If **50% wins**, the +12.5% step asymmetry is the first thing to rule out, and the 12.5% arm
  helps: it spends **fewer** steps than the 25% arm, so if it also scores worse, success is
  tracking share rather than steps in at least one direction.
- If **75% wins by a margin under 25%**, that is not separable from its extra compute on this
  design alone. Say so.
- A follow-up matched-steps arm is the clean resolution. **It is deliberately not in this
  experiment**, to keep one axis.

## Design

| | |
|---|---|
| primary depth | 4 (the frontier: 0.1591, learning and capped, no gap, no collapse) |
| arms at depth 4 | 12.5%, 50%, 75% share (25% reused from EXP-036) |
| add-on | depth 6 at 50% share |
| seeds | 0-11 (n = 12, the standing minimum) |
| episodes | 10,000, matching EXP-036 exactly |
| tag | `exp037_s{share}_d{depth}`, unique per cell |
| runs | 48 (4 cells x 12 seeds) |

`record_filename` encodes tag, arm, depth, seed and sigma and **not** `curriculum` or
`curriculum_weights`, so the tag must carry the share. The driver's collision guard tests the
real naming function.

**No random arm.** EXP-036 measured the floors at depths 4 and 6 (0.0031 and 0.0008) on the same
machine with the same seeds. Re-measuring an unchanged quantity would cost 24 runs for nothing.

## Why depth 4 is the primary and depth 6 only an add-on

Depth 4 is the only depth where an intervention is cleanly readable. It is learning (51x its
floor), it is capped, and EXP-036 showed it has **no generalisation gap and no collapse** - so a
change there moves one thing. Depth 6 sits at exactly 0.0000, which is a floor effect: any gain
is hard to size and any null is ambiguous between "starvation is not the problem" and "the
intervention was too small."

Depth 5 is skipped. It is broken but not floored, which sounds ideal, but it would double the
cost of the primary axis for a second reading of the same question.

## Code change

One change, in `src/neuromorphic/training/cube_baseline.py`:

- **`curriculum_schedule(stages, episodes, weights=None)`** - integer weights, proportional
  split, remainder to the final stage exactly as now. `weights=None` means uniform and must
  produce **the identical list** the current implementation produces.
- **`CubeConfig.curriculum_weights: tuple[int, ...] = ()`** - empty means uniform. Empty rather
  than `None` is safe here because an empty weight tuple has no meaningful non-default reading,
  unlike the `encoder_seed=0` case that forced `None` sentinels in `12bbbf8`.
- Validation: `len(weights)` must equal `len(curriculum)`, and every weight must be a positive
  integer. A zero weight would silently drop a stage.

**What must NOT change:** the per-stage environment construction. Each stage builds a fresh
`ShellCubeEnv` with `random.Random(train_seed)`.

> **This is why weighting needed a code change at all.** Weights can be faked today by repeating
> a depth in the curriculum tuple - `(1,2,3,4,4,4)` gives depth 4 half the budget. **That would
> have been a trap.** Each stage re-seeds its pool RNG from `train_seed`, so three consecutive
> depth-4 stages replay the *identical start-state sequence* three times. It is not 5,000 fresh
> episodes at depth 4; it is the same 1,666-episode sequence looped, with head and optimizer
> state carrying over. A win under that scheme would be unattributable. Real weights give one
> continuous stage with one continuous RNG stream.

## Testing

Governed by the repo's test-strength rule: **an assertion that cannot fail is not a test.**

- **Uniform weights reproduce the current schedule exactly**, for every stage count from 1 to 6
  and for the exact budgets used here. Equality on the whole list, not on its length.
- **Weights actually shift the split**, asserted on the exact expected counts from the table
  above. A test asserting only "the last stage is larger" would pass against an implementation
  that added one episode.
- **Total is conserved** under every weighting. This is the property that makes arms comparable
  at all, and it is the one a proportional-split bug would break first.
- **Invalid weights are rejected**: wrong length, zero, and negative each raise. A zero weight
  silently dropping a stage is the specific failure worth pinning.
- **Byte-identity on the default path**: depth 1, seed 0, 600 episodes, `tag=exp030_concept`
  must still give `success_rate 0.6666666666666666`, `revisit_rate 0.16633266533066132`,
  `eval_revisit_rate 0.25`, `greedy_modal_action_frac 1.0`,
  `mean_train_entropy 0.5422023858095053`.
- **End-to-end neutrality at depth 4.** A short depth-4 curriculum run (small budget, so it is
  affordable on the VPS) must produce an identical record with `curriculum_weights=()` and with
  explicit uniform weights. This is what licenses reusing EXP-036's depth-4 records as the 25%
  comparator; the depth-1 reference values alone do not exercise a multi-stage curriculum.

## Cost and dispatch

Using EXP-036's **measured** throughput, not the 90 ms rule of thumb and not the 153 ms
wall-equivalent that was itself measured at the wrong concurrency:

| cell | runs | steps each | total |
|---|---|---|---|
| d4 @ 12.5% | 12 | 75,008 | 900,096 |
| d4 @ 50% | 12 | 90,008 | 1,080,096 |
| d4 @ 75% | 12 | 100,004 | 1,200,048 |
| d6 @ 50% | 12 | 120,000 | 1,440,000 |
| **total** | **48** | | **4,620,240** |

EXP-036 sustained roughly 4.08 M trained steps in about 21 h wall, so this is **about 23-24 h**
at the same settings.

**Run with 10 workers, not 16.** EXP-036 ran 16 workers at **920 MB private each**, driving
system commit to 48.6 of 50.4 GB and holding utilisation at a measured **43.1%** - the workers
spent most of their time paging, not computing. Ten workers is 9.2 GB instead of 14.7 GB.
Effective throughput is roughly a wash on paper (10 x ~70% vs 16 x 43%), so this is close to
free, it leaves the laptop usable, and **it is a testable prediction**: record utilisation early
and compare against 43.1%.

Dispatch per `docs/playbooks/remote-experiment-runs.md`, using **`ssh laptop`**, never the
fully-qualified hostname.

## Out of scope

- **Adaptive advancement** (advance a stage when a success criterion is met). A different
  mechanism; folding it in would confound the weighting result. Its own experiment.
- **A matched-steps arm.** The clean resolution of the confound above, deliberately deferred to
  keep one axis.
- **Depth 5**, and anything past depth 6.
- **Re-measuring the random floors.** EXP-036 has them for depths 4 and 6.
- **Which depths are in the curriculum at all.** Always `(1..d)` here; only the weights move.

## Related

- `experiments/036_generalisation_gap/RESULTS.md` - the 25% arm, the floors, and the limitation
  that motivated this
- `experiments/034_learning_signal/RESULTS.md` - the refuted 100% endpoint, and the
  matched-budget fairness criterion
- `experiments/035_budget_scaling/RESULTS.md` - why buying more episodes is the expensive
  alternative
- `docs/handoffs/SESSION-HANDOFF-2026-08-03.md` section 1.6 - the throughput and memory numbers
  behind the cost model
