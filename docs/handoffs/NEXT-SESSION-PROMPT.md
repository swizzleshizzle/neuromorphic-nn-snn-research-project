# Prompt for a fresh session

Paste the block below into a new session. It is deliberately short - the handoff carries the
detail, and duplicating it here would create two versions that drift.

---

```
Picking up the neuromorphic cube project. Nothing is running; both machines are idle.

Read docs/handoffs/SESSION-HANDOFF-2026-08-14b.md first, then CLAUDE.md. The earlier
SESSION-HANDOFF-2026-08-14.md is still accurate on the science and is the fuller account.

Where we are: a spiking network learns to solve a 2x2 Rubik's cube. Depths 3 through 6
now all work (0.3972 / 0.5351 / 0.3412 / 0.1800), and the break point - stuck at depth 5
since EXP-036 - is now past depth 6 and unmeasured. Two levers got us here and they
compound: training the sensory encoder self-supervised, and capping the depth-1 training
step budget at 2. That second one closed a trap where curriculum stage 1 paid a
constant-action policy 0.3333 against a random policy's 0.2208, because a cube face has
order 4 and the 2d+3 budget let a repeated move cycle back to solved.

The visual story is DONE and no longer the job: four scenes render at 1080p on the laptop
and were checked against real frames on 2026-08-14.

YOUR JOB THIS SESSION: find the new break point at depth 7.

The pre-flight is already measured, so do not redo it:
- The depth-7 shell is 33,058 states, NOT the ~58,000 an earlier handoff guessed.
- ExactBFSDistance(max_depth=8) builds in 0.96 s at 95 MB. It is a non-issue.
- heldout_cap=200 already binds at depth 6, so evaluation cost does NOT grow with the
  shell. It grows only with episode length.

The one thing that DOES need designing: the training side goes 8,769 -> 32,858 states,
3.7x, while the episode budget stays at 10,000. If depth 7 fails, "harder task" and
"each state seen a quarter as often" are not separable unless the pre-registration says
so. Decide that before dispatching - either hold episodes-per-state roughly fixed, or
run a budget arm the way EXP-035 did at depth 3.

Pre-register the contract before the numbers exist, n >= 12 seeds, and put the primary
claim on a quantity that moves on every seed - n=12 cannot show a failure count went to
zero.

If depth 7 is not the right call, the other live options are re-running depth 5 at 24+
seeds to settle EXP-043's Claim 1 (p 0.0815, four regressing seeds, a p-value miss and
not a demonstrated absence), or building the three story-only scenes.
```

---

## Why this prompt is shaped this way

- **It names the file to read rather than restating it**, so the two cannot drift apart.
- **It leads with the two levers**, because every current number depends on both and neither is
  guessable from the code.
- **It spends its length on the confound, not the compute.** The pre-flight that the last prompt
  asked for is done, and it came back "no problem" - so the useful thing to carry forward is the
  one that is still a real decision, which is coverage against episode budget.
- **It says the render is finished**, because a prompt that still asks for it would get it redone.
