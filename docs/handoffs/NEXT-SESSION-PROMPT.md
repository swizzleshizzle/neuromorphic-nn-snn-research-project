# Prompt for a fresh session

Paste the block below into a new session. It is deliberately short - the handoff carries the
detail, and duplicating it here would create two versions that drift.

---

```
Picking up the neuromorphic cube project, Week 23. A RUN IS IN FLIGHT: EXP-059 was dispatched
to the laptop and takes about 30 hours, so it likely spans a laptop sleep.

Read docs/handoffs/SESSION-HANDOFF-2026-09-09.md FIRST, especially its section 0, then
CLAUDE.md. Ignore every earlier handoff; the 08-31 one in particular proposes an arm that is
disqualified.

FIRST ACTION: check whether EXP-059 finished, with probe_run.ps1 and a record count against
C:\Users\mlgbr\wt-exp053\experiments\059_memory_depth5\outputs. 72 records means done.
JUDGE PROGRESS BY CPU-HOURS PER WORKER, NEVER WALL CLOCK - EXP-058 slept 26 of its 39.7 hours
in transit and finished correctly. If it stopped early, --skip-existing resumes it losslessly
because seeded runs here are byte-identical.

There is deliberately NO aggregator for EXP-059 yet, so it can be written from the spec rather
than from the numbers. When the run lands: write aggregate.py against
docs/superpowers/specs/2026-09-09-exp059-memory-depth5-design.md, produce RESULTS.md, run the
suite in chunks, merge --no-ff, delete the branch, add the vault row.

Five facts shape everything:

1. READ "THE GATE-CALIBRATION RULE" IN CLAUDE.md BEFORE WRITING OR READING A VALIDITY GATE.
   EXP-058 was VOID because its gate required mean_n_stored > 10, a quantity bounded by episode
   length where episodes average 7.76 steps. Unsatisfiable by construction. EXP-059's gate was
   calibrated across its full range BEFORE its spec was written: empty attractor 1.000000,
   random loaded 0.9437, real depth-5 run 0.8128, threshold 0.95.

2. EXP-059's primary is M vs A and it is NOT directional. EXP-030's trap is that memory beats
   the shuffle-null while losing to the amnesic control; EXP-058 reproduced it exactly on a
   policy 15x better. A significant -0.05 is a real finding that memory HURTS, not a failed
   confirmation. n=24 is still only ~30-40% powered at a 0.03 effect, which is about what
   EXP-058 saw, so revisit_rate is the better-powered instrument and is a claim, not a footnote.

3. THE CRITIC QUESTION IS CLOSED. Its benefit is within-episode state-dependence, not
   calibration. Everything without it sits 0.1358-0.1558; the full critic sits at 0.2004.

4. FIVE INSTRUMENTS MOVE AGAINST POLICY QUALITY: the EXP-033 probe, pretraining move-accuracy,
   the entropy trace, S, and critic_ev. Use revisit_rate and optimality, and put none of the
   five in a new spec. Unanimity at p 0.0005 measures an instrument's consistency, not its link
   to the outcome.

5. OPERATIONAL: the Bash default timeout is 120 s and 600 s is a hard ceiling. Long background
   commands are killed here around 2-3 h with empty output - the remote run survives, only the
   notification is lost, and `| tail` buffers until exit so a killed piped command tells you
   nothing. NEVER pass comma-separated arguments over ssh; that failure EXITS ZERO. Verify a
   launch by probing for records and worker processes, never by an exit code. Sync the laptop
   worktree with sync_repo.ps1, never a bare checkout, and note exp040_encoder_s*.pt are NOT
   tracked in git.

After EXP-059: repeating EXP-056 at higher n (~25 h, 14 of it re-manufacturing encoders),
EXP-055's two leads, and a standing note on the five retired instruments. Nothing decays if
they wait.
```

---

## Why it is shaped that way

It opens with the in-flight run and its resume procedure, because a session that starts by
re-deriving state will burn an hour and may re-dispatch 30 hours of work that is already running.

Three things most likely to be lost otherwise:

**There is deliberately no aggregator yet.** That looks like an omission and is not: writing it
after seeing the numbers is how a contract gets bent. The spec is complete enough to write it from.

**Wall clock lies on this laptop.** EXP-058 slept 26 of its 39.7 hours in transit and still
finished correctly, and its records looked stalled for a full day. CPU-hours per worker is the only
honest progress signal. The laptop is on **EDT, UTC-4** - an earlier draft called that "a day
behind", which is wrong and would corrupt any ETA arithmetic built on it.

**EXP-058's seeds are burned and EXP-059 exists because of it.** The reflex of "fix the gate and
re-run" reproduces byte-identical records and licenses nothing. Changing venue to depth 5 is what
made a real re-test affordable.
