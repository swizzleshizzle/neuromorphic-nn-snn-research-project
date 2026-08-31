# Prompt for a fresh session

Paste the block below into a new session. It is deliberately short - the handoff carries the
detail, and duplicating it here would create two versions that drift.

---

```
Picking up the neuromorphic cube project, Week 22 session 1. Nothing is running, the laptop
is idle, and main is clean at 0e4ddfe.

Read docs/handoffs/SESSION-HANDOFF-2026-08-31.md first, then CLAUDE.md. IGNORE the 08-27
handoff - it presents a Stage 3 decision that was settled four days and three experiments
ago.

Where we are: a spiking network solves 2x2 cubes at depth 6 around 0.30, and Stage 3 is
finished. Four facts shape everything:

1. STAGE 3 IS DONE AND HALF OF IT REFUTED. A learned critic CONFIRMED at depth 7 (+0.0533,
   p 0.0498) - but by a hair, and its mechanism is absent: critic explained variance is
   0.0021, so V(s) is no better than a constant. Gating encoder plasticity on the
   neuromodulatory bus did NOT clear its bar, and the pre-registered rule RETIRES the
   neuromorphic claim rather than deferring it. The bus is load-bearing in code now and
   bought nothing. Do not write that up as a second neuromorphic change; week 20's encoder
   training was the one.

2. SEQUENCE-BLINDNESS IS REFUTED. S is flat across 10, 20, 40 and 80 epochs - the four arms
   differ by 0.08x their own sd while the policy halves. Four experiments leaned on that
   hypothesis. A random encoder scores LOWEST, refuting the spec's own pre-registered
   Johnson-Lindenstrauss prediction. Pretraining BUILDS structure, all of it before 10
   epochs.

3. FOUR INSTRUMENTS NOW MOVE AGAINST POLICY QUALITY: the EXP-033 probe, pretraining
   move-accuracy, the entropy trace, and S is UNEVALUATED rather than cleared - its Claim 4
   "PASSED" came from a correlation over four indistinguishable means, which is EXP-052's
   own process failure recurring inside the aggregator built to prevent it. Use
   revisit_rate and optimality. Do not put a probe number, an entropy number, or "S was
   cleared" in a new spec.

4. THE BASH TOOL DEFAULT TIMEOUT IS 120s AND 600s IS A HARD CEILING. Exceeding either
   auto-backgrounds the command and strands any subagent waiting on it. Six stalls came
   from this. Always pass an explicit timeout and say the number; split suites so no call
   approaches 600s.

YOUR JOB THIS SESSION: EXP-055 is pre-registered, built, reviewed and READY TO RUN on
branch exp-055-left-edge (unmerged, no PR). It measures the 0-to-10 pretraining window,
where both instruments say all the movement is and nobody has measured a point. Four arms
at 1, 2, 3, 5 epochs. Phase 1 pretraining is ~0.5h and phase 2 is seconds; the four RL arms
are ~17h.

Read section 3 of the handoff BEFORE dispatching. The laptop setup changed: work happens in
a worktree at C:\Users\mlgbr\wt-exp053, that worktree has NO .venv, and without PYTHONPATH
pointing at its own src the scripts silently import the main checkout's OLD library.
Start-Process over ssh dies when the session ends - run the launcher in the ssh FOREGROUND
and background it on the controller side.

If Michael wants something cheaper instead, the best-value alternative is asking WHY the
critic works: its explained variance is 0.0021, so replace the EMA baseline with a
per-episode batch-mean constant and no critic. If that matches +0.0533, the
state-dependence is irrelevant. One arm, ~6.3h, and it attacks the mechanism of the only
claim that confirmed.

Pre-register before the numbers exist, n >= 12 seeds, measure the floor, and price any
compute confound against EXP-046's budget curve. Two process failures in section 0 and the
Claim 4 warning are worth reading: both happened with every threshold obeyed.
```

---

## Why this prompt is shaped this way

- **It leads by disowning the previous handoff**, because that document still presents a settled
  decision as open and a fresh session would otherwise act on it.
- **It carries the four standing facts, not the newest result.** A session that knows only EXP-055
  will mis-plan; one that knows Stage 3 is finished, sequence-blindness is refuted, four instruments
  are inverted and the timeout ceiling exists will not.
- **It names S as UNEVALUATED rather than cleared.** That distinction is one aggregator line away
  from being cited as a clearance in a later spec, which is exactly how the probe spread through
  EXP-033/039/047.
- **It puts the laptop worktree trap above the dispatch instruction**, because importing the old
  library would produce a complete, plausible, wrong result rather than an error.
- **It names a cheaper alternative** so "do something tonight" never competes with the 18 h run.
