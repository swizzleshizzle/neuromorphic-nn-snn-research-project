# Prompt for a fresh session

Paste the block below into a new session. It is deliberately short - the handoff carries the
detail, and duplicating it here would create two versions that drift.

---

```
Picking up the neuromorphic cube project. Nothing is running; both machines are idle.

Read docs/handoffs/SESSION-HANDOFF-2026-08-20.md first, then CLAUDE.md. It has a decision
rule for tonight in section 0 - follow it.

Where we are: a spiking network solves 2x2 cubes with a 390-parameter linear head on a
FROZEN encoder. Depths 3-7 work. Four experiments this week established that the depth
series is really a BUDGET series: depth 7 needed 4.4x the episodes to work (EXP-044),
that gain came from total budget and not deep-end exposure (EXP-045: back-loading a fixed
budget made it WORSE, p 0.0010), depth 6 responds identically to the same 4.4x (EXP-046,
12-0-0, p 0.0005), and the curve across 10k/25k/44k is LOG-LINEAR with no knee - about
0.22 success per log10 of spend.

So budget is solved and unattractive: 4.4x buys about one depth, depth 8 would be
~194,000 episodes for ~0.18, and a log-linear curve has no cheap fraction.

TIMING MATTERS TONIGHT: this is session 3, run tonight, and Michael is out tomorrow
night. Aim to have something dispatched before the session ends so the laptop is not idle
all day. But do NOT rush a pre-registration to manufacture a dispatch - there is a
fallback for exactly that reason.

YOUR JOB THIS SESSION: design, pre-register and (if ready) dispatch encoder fine-tuning
during RL.

It is the only untried lever that could change the exchange rate rather than pay it.
Three things to get right before writing code:
- The encoder is frozen BY CONSTRUCTION. make_agent builds the brain and nothing unfreezes
  it; the 390-parameter head is the entire trainable surface. This is a trainer change.
- "The same 390 trainable parameters" is load-bearing in every write-up and the whole
  visual story. A fine-tuned arm is a DIFFERENT ARCHITECTURE and must be reported as one.
- Pre-register the confound: fine-tuning adds trainable parameters AND compute per step.
  Matching episodes gives it more compute; matching compute gives the frozen control more
  episodes, which EXP-046 prices at ~+0.22 per log10 and could swamp the effect. Pick one,
  and report what it cannot rule out.

Depth 6 at 10,000 episodes is the natural test bed: baseline 0.1800, ~5 h for 12 seeds,
paired against EXP-043. Consider re-probing the fine-tuned encoder as well as scoring it -
that says whether RL improved the representation or just fitted the head to it.

IF IT IS NOT SMOKE-TESTED AND READY when the session winds down, dispatch the fallback
instead and finish the design unhurried: re-run depth 5 with 24 seeds to settle EXP-043's
Claim 1 (+0.1108 at p 0.0815). It needs 12 more pretrained encoders first (~20 min at 12
workers, only seeds 0-11 exist) and then ~4.6 h. Total ~5 h.

Pre-register the contract before the numbers exist, n >= 12 seeds, measure the floor
rather than assuming it, and put the primary claim on a quantity that moves on every seed.
```

---

## Why this prompt is shaped this way

- **It names the file to read rather than restating it**, so the two cannot drift apart.
- **It leads with the reframing, not the number.** "Depth 7 works" is the shallow reading;
  "the deficit was coverage, and the break point moved again" is the finding.
- **It carries the refusal with the result.** The overreach this invites is a single sentence
  away from the truth, and depth 3's coverage is the one-line disproof.
- **It carries the schedule, because the schedule changes the plan.** A design-only session is
  right in general and wrong the night before a day nobody is around to dispatch.
- **It names the fallback in the prompt itself**, so "dispatch something" never competes with
  "pre-register properly". The fallback exists to keep that trade off the table.
