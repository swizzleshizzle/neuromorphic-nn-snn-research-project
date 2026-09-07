# Prompt for a fresh session

Paste the block below into a new session. It is deliberately short - the handoff carries the
detail, and duplicating it here would create two versions that drift.

---

```
Picking up the neuromorphic cube project, Week 23. Nothing is running, the laptop is idle, and
main is clean at 99dcfdd with no branches and no open PRs.

Read docs/handoffs/SESSION-HANDOFF-2026-09-07.md first, then CLAUDE.md. IGNORE the 09-03
handoff (out of date) and DO NOT ACT ON the 08-31 one at all: its item 2 proposes an arm that
is disqualified, and building it would waste six hours on a control whose confound is aligned
with its own hypothesis.

Where we are: a spiking network solves 2x2 cubes at depth 6 around 0.36. Five facts shape
everything:

1. READ "THE GATE-CALIBRATION RULE" IN CLAUDE.md BEFORE WRITING A VALIDITY GATE. Two of the
   three experiments that used one got it wrong this month, both the same shape: a threshold
   chosen in one regime and applied in another. EXP-058 is VOID because its gate required
   mean_n_stored > 10, a quantity bounded by episode length, in a setting where episodes
   average 7.76 steps. Unsatisfiable by construction. Compute a gate's maximum attainable
   value before committing it.

2. THE CRITIC QUESTION IS CLOSED. Its benefit is within-episode state-dependence, not
   calibration. EXP-056 showed flattening V(s_t) to its episode mean costs -0.0646; EXP-057
   showed a state-blind fitted constant beats the lagging EMA by only +0.0088 at p 0.7822.
   Everything without within-episode dependence sits 0.1358-0.1558; the full critic sits at
   0.2004. That is a SHAPE across two experiments, not a resolved decomposition.

3. FIVE INSTRUMENTS MOVE AGAINST POLICY QUALITY: the EXP-033 probe, pretraining move-accuracy,
   the entropy trace, S, and critic_ev. Use revisit_rate and optimality. Do not put any of the
   five in a new spec. And remember unanimity at p 0.0005 measures an instrument's consistency,
   not its link to the outcome - the probe was unanimous at every depth and still ranked seeds
   no better than chance.

4. THE LAPTOP IS NOT READY TO DISPATCH. Its worktree C:\Users\mlgbr\wt-exp053 is on
   exp-058-memory-reask at 4a96137, two commits behind a branch that no longer exists on
   origin. Sync it to main with sync_repo.ps1 (-Repo and -Branch are parameters), never a bare
   checkout. THE WORKTREE HAS NO .venv, so the only interpreter imports the MAIN checkout's src
   unless PYTHONPATH overrides it, silently and plausibly. Copy the gate in any launch0NN_wt.ps1.

5. OPERATIONAL: the Bash default timeout is 120 s and 600 s is a hard ceiling, so always pass
   an explicit timeout and split suites. NEVER pass comma-separated arguments over ssh - cmd.exe
   eats the commas and the failure EXITS ZERO. Verify a launch by probing for records and worker
   processes, never by an exit code. Long background commands get killed here around 2-3 h with
   empty output; foreground chunks under 600 s are reliable.

Highest-value open item, and it needs a design decision BEFORE any dispatch: a successor to
EXP-058. Its claim thresholds were never contaminated and can be reused, but THE SEEDS ARE
BURNED - runs here are byte-identical, so re-running seeds 0-11 under a corrected gate
reproduces exactly the void records, which is laundering rather than replication. An
uncontaminated re-test needs new seeds (expensive: E2 encoders exist only for seeds 0-11, so it
means re-running the EXP-047 and EXP-049 chains first) or a different measurement, such as the
same question at another depth on a base config whose encoders already exist more widely. The
second is cheaper and is genuinely new rather than a repeat.

Its prior: the effects to detect are about -0.03 for memory vs amnesic and -0.045 for shuffled
vs amnesic, both under the +0.05 bar EXP-058 set. Raise n or lower the bar deliberately and say
which first. Gate on RECALL differing between arms, not on storing.

Also open: repeating EXP-056 at higher n (~25 h, not the ~6 h an earlier note claimed, because
14 h of it is re-manufacturing encoders), EXP-055's two leads, and a standing note on the five
retired instruments.

Both are yours to schedule; nothing decays if they wait.
```

---

## Why it is shaped that way

It opens on the gate-calibration rule rather than on a result, because that is the failure mode
actively costing experiments: **EXP-058 spent 20.2 hours and produced nothing reportable**, and the
cause was one arithmetic check nobody did before committing the spec.

Three things most likely to be lost otherwise:

**The seeds are burned, and that is not obvious.** Byte-identical seeded runs are normally an asset
here, used as a correctness check. After a void experiment they become a liability: the obvious
"fix the gate and re-run" produces the same records and licenses nothing. A fresh session will
reach for it immediately.

**EXP-058's void run still reproduced EXP-030's trap.** Memory beat the shuffle-null and did not
beat the amnesic control, exactly as in 2026-07 on a policy 15x worse. The same two-arm design
would have reported a win both times. That is the reason the successor keeps three arms.

**The EXP-056 repeat is ~25 h, not ~6 h.** An earlier handoff priced it by costing the arm and
forgetting it has neither controls nor encoders at new seeds. Anyone re-deriving that estimate
should check what exists at which seed before quoting a number.
