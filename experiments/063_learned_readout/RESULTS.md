# EXP-063 results - can ANY readout use the stored content?

> **PENDING. Dispatched 2026-09-17, no numbers yet.** This file exists before the run so the
> provenance and the regeneration command are recorded in git rather than in a session, which is
> the standing habit since the 2026-07-13 audit found EXP-027's numbers living only in a
> gitignored `outputs/`.

**Pre-registration:** `docs/superpowers/specs/2026-09-17-exp063-learned-readout-design.md`.
**The aggregator was written before any EXP-063 number existed**, together with 17 tests and 8
mutations, all caught. Confirm completion from **counts** - 48 records, zero python processes -
**not** from the run log, which prints one success rate per line and would contaminate the
reading.

## The question

EXP-059: memory HURTS (`M - A` = -0.0954, p 0.0056). EXP-061: the recall block is
indistinguishable from matched-magnitude noise (`M - N` = +0.0210, p 0.4989). Neither can say
whether the episodic information is **unusable** or merely **unused**, and EXP-059's own gate
says the content is there (`recall_content_cos` **0.8514**, well below 1.0).

Every cube experiment to date has measured ONE readout: a raw hippocampal read concatenated onto
the concept. Nothing has learned **what to attend to**.

## Provenance

| | |
|---|---|
| Run | `SwizzlesDuo`, worktree `C:\Users\mlgbr\wt-exp053`, 6 workers |
| Depth / budget | 5, 10,000 episodes, curriculum 1..5, `max_steps_by_depth=((1, 2),)` |
| Seeds | 0-23, frozen EXP-040 E0 encoders |
| Arms T, U | **NEW**: `exp063_attn_d5`, `exp063_attnnoise_d5` |
| Arms A, M | **REUSED from EXP-059**, 24 seeds, not re-run |
| Arm N | **REUSED from EXP-061**, 24 seeds, not re-run |
| Estimated cost | 48 cells x 3.05 h / 6 workers = **~24.4 h** compute |

```bash
powershell -File C:\Users\mlgbr\launch063_wt.ps1 -Phase rl -Workers 6 -SkipExisting
.venv/bin/python -u experiments/063_learned_readout/aggregate.py
```

> [!important] **REUSING EXP-059's AND EXP-061's ARMS REQUIRED PROVING THE CODE CHANGE WAS INERT,
> NOT ASSUMING IT.** All five pre-existing readouts were run at depth 1, seed 3 against the
> pre-change code in a clean worktree and against the post-change code, on nine recorded
> quantities: **identical to full float repr**. The attention module and the noise generator are
> constructed only for the two new modes, so no other mode's torch RNG stream shifts.
> `test_preexisting_modes_are_numerically_inert` locks it in, and mutation M8 - building the
> attention for every non-concept mode - fails it.

## Gate calibration, before the spec was committed

| gate | reading | measured (seeds 0-3, both arms) | floor | margin |
|---|---|---|---|---|
| 1 | `recall_concept_norm_ratio` | **0.7836 - 0.7972** | 0.05 | **15.7x** |
| 2 | `attn_choice_steps / train_steps` | **0.7601 - 0.7672** | 0.15 | 5.1x |

Gate 2's floor is set from the **reasoned trained-regime minimum (0.375)**, not from the
measurement: the calibration policy is untrained, its episodes run to the cap, and it therefore
reads near the CEILING. Setting a floor just under a measurement taken in the wrong regime is
exactly what killed EXP-057's and EXP-058's gates.

## Results

*(pending)*
