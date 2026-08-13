# Prompt for a fresh session

Paste the block below into a new session. It is deliberately short - the handoff carries the
detail, and the point is to get the next session reading the right file rather than guessing.

---

```
Picking up the neuromorphic cube project. EXP-043 is running on the laptop.

Read docs/handoffs/SESSION-HANDOFF-2026-08-13.md first, then CLAUDE.md.

Context in one paragraph: a spiking network learns to solve a 2x2 Rubik's cube. Two
levers were found this week. First, training the sensory encoder self-supervised
(predict the move between two states) took depth 4 from 0.1591 to 0.3471. Second,
EXP-041 found that curriculum stage 1 paid a constant-action policy 0.3333 against a
random policy's 0.2208, because a cube face has order 4 and the 2d+3 step budget let a
repeated move cycle back to solved. Capping the depth-1 TRAINING budget at 2 (EXP-042)
took depth 4 to 0.5351, seeds-at-zero from 2/12 to 0/12, and sd from 0.2242 to 0.1012 -
and it helped the ten seeds that never failed by +0.1195, so the trap was degrading
every run.

EXP-043 applies that same cap at depths 5 and 6, paired against EXP-040 (0.2304 and
0.1037, both measured with the trap still in place). 24 runs, dispatched 2026-08-13
13:13, ETA about 02:15 Friday.

First tasks, in order:

1. Probe the run:
   ssh -n laptop 'powershell -NoProfile -ExecutionPolicy Bypass -File C:\Users\mlgbr\probe_run.ps1 -OutDir "C:\Users\mlgbr\Desktop\Projects\neuromorphic-nn-snn-research-project\experiments\043_cap_at_depth_5_6\outputs"'

2. WRITE experiments/043_cap_at_depth_5_6/aggregate.py BEFORE reading any record.
   Rules on disk before numbers is the standing habit here. The contract is in
   docs/superpowers/specs/2026-08-13-exp043-cap-at-depth-5-6-design.md and
   experiments/042_depth1_trap/aggregate.py is the template.

3. When it lands: fetch the records (outputs/ is gitignored, the laptop is the only
   copy), aggregate, and write RESULTS.md marking each claim confirmed or refuted.

Two things to know before interpreting anything:

- The EXP-040 baseline has NO stage_trace telemetry, so the primary claim is on
  SUCCESS, not entropy. Do not try to pair the mechanism.

- n=12 CANNOT prove a 2-of-12 failure rate went to zero. A paired permutation test
  gives p about 0.5 by construction, and Fisher's exact gives about 0.48. Report
  failure counts descriptively with no p-value, and put claims on quantities that move
  on every seed. EXP-042's aggregate.py shows the pattern.

Also open: the manim visual-story scenes have never been rendered and Content Day is
Saturday Aug 16. manimpango will not build on the VPS (three attempts recorded in
viz/manim/README.md) - render on the laptop once EXP-043 clears.
```

---

## Why this prompt is shaped this way

- **It names the file to read rather than restating it.** The handoff is the source of truth;
  duplicating it here would create two versions that drift.
- **It leads with the two levers**, because every current number depends on both and neither is
  guessable from the code.
- **It puts "write the aggregator first" as task 2, not task 3.** Once the records are readable
  the temptation is to look, and looking first is how thresholds get chosen to fit.
- **It states the statistical limit up front**, because the most likely mistake with this data is
  reporting a 2/12 -> 0/12 drop as if a p-value applied to it.
- **It flags the Saturday deadline**, which is the only item with an external date.
