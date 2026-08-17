# Prompt for a fresh session

Paste the block below into a new session. It is deliberately short - the handoff carries the
detail, and duplicating it here would create two versions that drift.

---

```
Picking up the neuromorphic cube project. Nothing is running; both machines are idle.

Read docs/handoffs/SESSION-HANDOFF-2026-08-17.md first, then CLAUDE.md.

Where we are: a spiking network learns to solve a 2x2 Rubik's cube with a 390-parameter
linear head on a frozen encoder. Depths 3 through 7 all clear the working bar. EXP-044
just finished and its result reframes the series: depth 7 scored 0.0621 at the standard
10,000 episodes and REFUTED, then 0.1971 at 44,000 and CONFIRMED with 12 of 12 seeds
above the bar. Depth 6 at 10,000 and depth 7 at 44,000 sit at the same coverage - 0.190
against 0.191 episodes per training state - and score 0.1800 against 0.1971, inside
noise. So the depth-7 deficit was STARVATION, not difficulty, and the break point is
not found.

Do NOT generalise that to "every depth just needs more episodes". Depth 3 has 37x depth
4's coverage and scores lower, so coverage cannot explain the whole series. Only the
6-to-7 step has a matched-coverage comparison.

YOUR JOB THIS SESSION: test the scaling law where it is falsifiable.

Re-run depth 5 or 6 at raised coverage. If depth 6 at ~0.97 coverage (matching depth 5's)
lands near depth 5's 0.3412, the law holds and the whole depth series has to be restated
as a budget series. If it does not move, depth 7 was a special case.

Depth 8 is the tempting alternative and it is the wrong call for now: matched coverage
there needs ~174,000 episodes, about 100 h, because the matched budget grows ~4x per
depth. Test the law first; it is cheaper and it decides something.

Pre-register the contract before the numbers exist, n >= 12 seeds, measure the floor
rather than assuming it, and put the primary claim on a quantity that moves on every
seed - n=12 cannot show a failure count went to zero.

Other live options: fine-tune the encoder during RL (untested), or re-run depth 5 with
24+ seeds to settle EXP-043's Claim 1 (+0.1108 at p 0.0815, four regressing seeds).
```

---

## Why this prompt is shaped this way

- **It names the file to read rather than restating it**, so the two cannot drift apart.
- **It leads with the reframing, not the number.** "Depth 7 works" is the shallow reading;
  "the deficit was coverage, and the break point moved again" is the finding.
- **It carries the refusal with the result.** The overreach this invites is a single sentence
  away from the truth, and depth 3's coverage is the one-line disproof.
- **It argues against the obvious next experiment.** Depth 8 is what a reader would reach for, so
  the prompt says why the cheaper test comes first.
