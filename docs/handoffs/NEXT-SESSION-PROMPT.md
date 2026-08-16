# Prompt for a fresh session

Paste the block below into a new session. It is deliberately short - the handoff carries the
detail, and duplicating it here would create two versions that drift.

---

```
Picking up the neuromorphic cube project. Nothing is running. EXP-044 landed:
depth 7 scores 0.0621 and does NOT clear the 0.10 bar, so Claim 1 is REFUTED - but the
break point is NOT located, because the pre-registered escalation (arm B, 44,000
episodes, ~32 h) is triggered and unrun.

Read docs/handoffs/SESSION-HANDOFF-2026-08-14b.md first, then CLAUDE.md. The earlier
SESSION-HANDOFF-2026-08-14.md is still accurate on the science and is the fuller account.

Where we are: a spiking network learns to solve a 2x2 Rubik's cube. Depths 3 through 6
now all work (0.3972 / 0.5351 / 0.3412 / 0.1800), and the break point - stuck at depth 5
since EXP-036 - is now past depth 6 and unmeasured. Two levers got us here and they
compound: training the sensory encoder self-supervised, and capping the depth-1 training
step budget at 2. That second one closed a trap where curriculum stage 1 paid a
constant-action policy 0.3333 against a random policy's 0.2208, because a cube face has
order 4 and the 2d+3 budget let a repeated move cycle back to solved.

The visual story is DONE and no longer the job: seven scenes render at 1080p on the laptop,
FullStory assembles them into one 2:21 cut with act cards, and all of it was checked
against real frames on 2026-08-14. The only unbuilt scene is PolicyCollapse, which needs a
cube renderer.

YOUR JOB THIS SESSION: decide on arm B, and if yes, dispatch and land it.

Arm B is the only thing that turns 'depth 7 does not clear the bar' into a located
frontier. Its reading is already fixed in the spec: B works -> the failure was
STARVATION and the break point is not found; B also fails -> the break point IS depth 7
for this recipe. Do not edit a threshold, and do not compute a p-value for Claim 1 - it
is absolute by design, and computing one means you invented a baseline.

Read experiments/044_depth7_frontier/RESULTS.md first. Facts already measured:
- The depth-7 shell is 33,058 states, NOT the ~58,000 an earlier handoff guessed.
- ExactBFSDistance(max_depth=8) builds in 0.96 s at 95 MB. It is a non-issue.
- heldout_cap=200 already binds at depth 6, so evaluation cost does NOT grow with the
  shell. It grows only with episode length.

- Arm A took at most 7.3 h for 12 seeds at 12 workers. Arm B is 4.40x the steps, so
  about 32 h. Nothing competes for the laptop: Content Day is defunct.
- The floor is measured at exactly 0.0000, so BAR = 0.10 binds.
- The 12 depth-7 head checkpoints are still only on the laptop. They are tracked; copy
  and commit them.

If depth 7 is not the right call, the other live options are re-running depth 5 at 24+
seeds to settle EXP-043's Claim 1 (p 0.0815, four regressing seeds, a p-value miss and
not a demonstrated absence), or building PolicyCollapse, which needs a 2x2 cube renderer
and is the most visceral shot in the deck.

One thing to know before touching the visual story: experiment records are gitignored, so
each machine holds only what it ran. The laptop had no EXP-039 and a scene died on it; the
VPS still has no EXP-029/030 and the laptop has both. Copy records, never transcribe them.
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
