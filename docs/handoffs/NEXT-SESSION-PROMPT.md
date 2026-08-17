# Prompt for a fresh session

Paste the block below into a new session. It is deliberately short - the handoff carries the
detail, and duplicating it here would create two versions that drift.

---

```
Picking up the neuromorphic cube project. Nothing is running; both machines are idle.

Read docs/handoffs/SESSION-HANDOFF-2026-08-17.md first, then CLAUDE.md.

Where we are: a spiking network learns to solve a 2x2 Rubik's cube with a 390-parameter
linear head on a frozen encoder. Depths 3-7 all clear the working bar. Two experiments
just landed and together they change how the whole series should be read.

EXP-044: depth 7 scored 0.0621 at the standard 10,000 episodes (REFUTED) and 0.1971 at
44,000 (CONFIRMED, 12 of 12 seeds). So depth 7 was under-trained, not out of reach.

EXP-045: which quantity did that? It gave stage 7 the same episode count inside the SAME
10,000 budget, by weighting the curriculum. Result: 0.0142 against arm A's 0.0621, paired
delta -0.0479, 0-11-1, p 0.0010, 9 of 12 seeds at exactly zero. So the operative variable
is TOTAL BUDGET, and moving budget toward the deep end is actively harmful. Entropy inside
that long deep stage fell to 2.7e-06 while only 2.2% of training episodes solved - the
policy collapsed during the stage meant to help it. EXP-037's depth-4 decline generalises.

The lesson worth carrying: the shallow curriculum stages are not warm-up, they are what
keeps the policy from collapsing. Present and cheap, not absent and not dominant.

YOUR JOB THIS SESSION: re-run depth 6 at raised TOTAL budget.

About 51,000 episodes for depth-5-like exposure, ~25 h on the laptop. If depth 6 moves the
way depth 7 did, the whole depth series is a budget series and every number in this repo
needs restating as "at 10,000 episodes". If it does not move, depth 7 was the special case.

Do NOT reach for depth 8: matched exposure there needs ~174,000 episodes, about 100 h,
because the budget grows ~4x per depth. And do not conclude "every depth just needs more
episodes" - only depth 7 has ever been run above 10,000, and depth 3 scores LOWER than
depth 4 despite a far smaller shell.

Pre-register the contract before the numbers exist, n >= 12 seeds, measure the floor rather
than assuming it, and put the primary claim on a quantity that moves on every seed.

Other live options: fine-tune the encoder during RL (the only untried lever that is not
about budget), or re-run depth 5 with 24+ seeds to settle EXP-043's Claim 1.
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
