# Prompt for a fresh session

Paste the block below into a new session. It is deliberately short - the handoff carries the
detail, and duplicating it here would create two versions that drift.

---

```
Picking up the neuromorphic cube project. Nothing is running; both machines are idle.

Read docs/handoffs/SESSION-HANDOFF-2026-08-20.md first, then CLAUDE.md.

Where we are: a spiking network solves 2x2 cubes with a 390-parameter linear head on a
FROZEN encoder. Depths 3-7 work. Four experiments this week established that the depth
series is really a BUDGET series: depth 7 needed 4.4x the episodes to work (EXP-044),
that gain came from total budget and not from deep-end exposure (EXP-045, back-loading a
fixed budget made it WORSE at p 0.0010), depth 6 responds identically to the same 4.4x
(EXP-046, 12-0-0, p 0.0005), and the curve between 10k/25k/44k is LOG-LINEAR with no
knee - about 0.22 success per log10 of spend.

So budget is solved and unattractive: 4.4x buys about one depth, depth 8 would be
~194,000 episodes for ~0.18, and there is no cheap fraction of a log-linear curve.

YOUR JOB THIS SESSION: design and pre-register encoder fine-tuning during RL.

It is the only untried lever that could change the exchange rate rather than pay it.
Unlike the last four experiments it needs real design work, so budget the session for
that rather than for a dispatch.

Three things to get right before writing any code:
- The encoder is frozen BY CONSTRUCTION. make_agent builds the brain and nothing
  unfreezes it; the 390-parameter head is the entire trainable surface. This is a
  trainer change, not a config flag.
- "The same 390 trainable parameters" is load-bearing in every write-up and the whole
  visual story. A fine-tuned arm is a DIFFERENT ARCHITECTURE and must be reported as
  one, not folded into the depth series.
- Pre-register the confound: fine-tuning adds trainable parameters AND compute per step.
  Decide what the control holds fixed before the numbers exist - a budget-matched frozen
  arm is the obvious one, because EXP-044/046 make budget the leading alternative
  explanation for any gain.

Pre-register the contract before the numbers exist, n >= 12 seeds, measure the floor
rather than assuming it, and put the primary claim on a quantity that moves on every
seed.

Cheaper alternatives if you want a dispatch instead: re-run depth 5 with 24+ seeds to
settle EXP-043's Claim 1 (+0.1108 at p 0.0815), or bring the vault's
road-to-a-solved-cube and progress-tracker current - both are still at EXP-037.
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
