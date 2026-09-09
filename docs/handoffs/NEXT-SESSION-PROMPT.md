# Prompt for a fresh session

Paste the block below into a new session. It is deliberately short - the handoff carries the
detail, and duplicating it here would create two versions that drift.

---

```
Picking up the neuromorphic cube project, Week 23. NOTHING IS RUNNING.

EXP-059 was dispatched and then DESTROYED by a Windows Update reboot on the laptop at
2026-09-09 07:35 UTC, 3.75 hours in. No cell had completed, records are written only on
completion, so the outputs directory is empty and about 19 CPU-hours were lost. No numbers
exist, so nothing is contaminated - the spec, the calibrated gate and the pre-registered
claims are intact and re-dispatchable as they stand.

Read docs/handoffs/SESSION-HANDOFF-2026-09-09.md FIRST, sections 0 and 0b, then CLAUDE.md.
Ignore every earlier handoff.

THERE IS A DECISION WAITING FOR MICHAEL, in section 0b: re-dispatch unchanged (~45 h),
re-dispatch at 12 seeds (~22 h, but n=24 was chosen deliberately to halve the standard
error), or leave it. DO NOT RE-DISPATCH WITHOUT DEFERRING WINDOWS UPDATE on the laptop -
that combination is exactly what destroyed the first attempt, and a second TrustedInstaller
reboot is likely.

Five facts shape everything:

1. AN UNREACHABLE TAILSCALE PEER IS AN UNKNOWN, NOT A PAUSED JOB. "offline, last seen 1h
   ago" cannot distinguish a sleep from a reboot. The playbook used to say it meant "slept,
   the job is paused, not dead"; that was false and EXP-059 was reported as healthy for 13
   hours while its workers no longer existed. The corrected three-reading test - uptime,
   python process count, record count - is in the playbook.

2. NOTHING IS DURABLE UNTIL A CELL COMPLETES. Records are written per cell at
   cube_baseline.py:1058, so a run carries a rolling exposure of workers x per-cell-hours,
   about 20 CPU-h at 6 workers. Before wave 1 lands that exposure is the entire run.

3. READ "THE GATE-CALIBRATION RULE" IN CLAUDE.md BEFORE WRITING OR READING A VALIDITY GATE.
   EXP-058 was VOID because its gate required mean_n_stored > 10, bounded by episode length
   where episodes average 7.76 steps. EXP-059's gate was calibrated across its full range
   BEFORE its spec: empty attractor 1.000000, random loaded 0.9437, real depth-5 run 0.8128,
   threshold 0.95.

4. EXP-059's primary is M vs A and it is NOT directional. EXP-030's trap is that memory beats
   the shuffle-null while losing to the amnesic control; EXP-058 reproduced it exactly on a
   policy 15x better. A significant -0.05 is a real finding that memory HURTS.

5. THE CRITIC QUESTION IS CLOSED - the benefit is within-episode state-dependence, not
   calibration. FIVE INSTRUMENTS MOVE AGAINST POLICY QUALITY: the EXP-033 probe, pretraining
   move-accuracy, the entropy trace, S, and critic_ev. Use revisit_rate and optimality.

Cost note: EXP-059 was priced at 2.5 h/cell and measured above 3.21 h, because the estimate
multiplied the step-budget ratio by the stage-count ratio. A curriculum SPLITS a fixed
episode count across stages; the playbook's steps() helper gives 0.90, not 0.72. A shallower
depth is not proportionally cheaper when the episode count is fixed.
```

---

## Why it is shaped that way

It opens with the loss and the pending decision, because the single worst outcome would be a fresh
session cheerfully re-dispatching 45 hours into an un-deferred Windows Update.

Three things most likely to be lost otherwise:

**The misdiagnosis, not just the failure.** The run died at 07:35 and was reported as "asleep,
nothing lost" five times over 13 hours, because the playbook's own table said an unreachable peer
meant a paused job. Recording only "Windows Update killed it" would leave the reasoning error in
place to be repeated.

**Nothing is contaminated.** No cell completed, so no number exists. That is the one piece of good
news and it is easy to miss under the loss: the pre-registration survives completely intact.

**The exposure window is structural.** Records land only on cell completion, so any re-dispatch
carries the same rolling ~20 CPU-h risk, and the whole-run risk until wave 1 lands.
