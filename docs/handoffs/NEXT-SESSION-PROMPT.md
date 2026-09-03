# Prompt for a fresh session

Paste the block below into a new session. It is deliberately short - the handoff carries the
detail, and duplicating it here would create two versions that drift.

---

```
Picking up the neuromorphic cube project, start of Week 23. Nothing is running, the laptop
is idle, and main is clean at 08c60bd with no branches and no open PRs.

Read docs/handoffs/SESSION-HANDOFF-2026-09-03.md first, then CLAUDE.md. IGNORE the 08-31
handoff. It is three experiments out of date AND its item 2 proposes an arm that the data
has since disqualified - following it would spend six hours building a control that cannot
answer its own question.

Where we are: a spiking network solves 2x2 cubes at depth 6 around 0.30. Five facts shape
everything:

1. THE CRITIC'S BENEFIT IS ITS WITHIN-EPISODE STATE-DEPENDENCE, and this reverses what
   EXP-053 wrote down. That experiment reported "the mechanism is measurably absent" from a
   critic explained variance of 0.0021 - but that number is the FINAL STAGE ONLY. By stage
   it is +0.4702 at depth 1 and +0.2054 at depth 2, every seed positive at depths 1-3.
   EXP-056 then flattened V(s_t) to its own episode mean and lost 0.0646 at p 0.0234,
   collapsing the arm to the EMA baseline it was meant to beat. Do NOT build the
   batch-mean-baseline arm the 08-31 handoff proposes: it forms G_t - mean(G), which is
   exactly zero on a one-step episode, and depth 1 averages 1.22 steps per episode.

2. FIVE INSTRUMENTS NOW MOVE AGAINST POLICY QUALITY: the EXP-033 probe, pretraining
   move-accuracy, the entropy trace, S (refuted outright by EXP-055, not merely
   unevaluated), and critic_ev. Use revisit_rate and optimality. critic_ev MUST NOT gate
   critic work - EXP-056's worse-performing arm had the BETTER-fitting critic at every
   stage. Do not put a probe number, an entropy number, an S number, or a critic_ev number
   in a new spec.

3. THE PRETRAINING LEFT EDGE IS REAL AND THE EXP-039/040 FRAMING SURVIVES. One epoch
   reaches 0.0854, only 42.5% of the plateau, and e10 - e1 is +0.1158 at p 0.0098. Roughly
   42.5% of the benefit is escaping random init and 57.5% is the objective. Only e1 -> e2
   resolves (+0.0629); the other adjacent steps are UNRESOLVED at ~10% power, not flat.

4. THE LAPTOP IS NOT READY TO DISPATCH. Its worktree C:\Users\mlgbr\wt-exp053 is on
   exp-056-flattened-critic at e7e0d73, one commit behind a branch that no longer exists on
   origin. Sync it to main with sync_repo.ps1 (-Repo and -Branch are parameters), never a
   bare checkout. Expect untracked files to move to an attic; 114 already did. THE WORKTREE
   HAS NO .venv, so the only interpreter imports the MAIN checkout's src unless PYTHONPATH
   overrides it - that failure is silent and produces a complete, plausible, wrong result.
   Both launch055_wt.ps1 and launch056_wt.ps1 refuse to start unless neuromorphic.__file__
   resolves under the worktree. Copy that gate.

5. THE BASH TOOL DEFAULT TIMEOUT IS 120s AND 600s IS A HARD CEILING. Always pass an
   explicit timeout. Split any suite so no single call approaches 600s; the tests/training
   remainder is ~837s and must be backgrounded. Never put "run the whole suite" in a
   task-scoped brief. And NEVER pass comma-separated arguments over ssh: cmd.exe eats the
   commas, and the resulting failure EXITS ZERO. Verify a launch by probing for records and
   worker processes, never by the ssh exit code.

Highest-value open item: the CONSTANT-CRITIC arm, already pre-registered in EXP-056's spec
section 5. A single learned scalar with no state input, fitted by the same MSE loss at the
same rate. It differs from arm B in exactly one way and from the EMA baseline in exactly one
way, and it has no one-step degeneracy. It separates calibration from between-episode
state-dependence, which EXP-056 deliberately could not. About 4-6 h on the idle laptop.

Also open: repeating EXP-056 at higher n (p 0.0234 against a 0.025 threshold is thin for a
result this load-bearing), and the memory re-ask, which needs the full three-arm design at
depth 6, roughly 15 h, and its own pre-registration.

Both are yours to schedule; nothing decays if they wait.
```

---

## Why it is shaped that way

It opens by disowning the 08-31 handoff, because that document does not merely go stale, it
carries an **actively wrong instruction**. Its item 2 proposes the per-episode batch-mean arm,
which EXP-053's corrected by-stage numbers disqualify: the arm loses gradient exactly where the
critic predicts best, so the confound is aligned with the hypothesis.

Three things most likely to be lost otherwise:

**`critic_ev` is retired, and it is the fifth instrument to go.** The pattern is now consistent
enough that a new spec citing any single-number instrument as evidence about policy quality should
be treated as suspect by default. EXP-056's own pilot predicted the wrong answer from `critic_ev`,
and running the pre-registered arm anyway is what caught it.

**The laptop's worktree is stale in a way that does not error.** It sits on a deleted branch, one
commit behind, with 114 files in an attic. A bare checkout will either fail or, worse, run the
wrong library and produce a plausible number.

**"Unanimous at p 0.0005" is not evidence of behavioural relevance.** The probe re-analysis found
the probe unanimous at every depth and still unable to rank the seeds better than chance. That
phrase appears in several older `RESULTS.md` files and reads far stronger than it is.
