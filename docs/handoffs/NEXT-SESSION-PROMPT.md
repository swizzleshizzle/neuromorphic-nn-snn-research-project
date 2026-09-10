# Prompt for a fresh session

Paste the block below into a new session. It is deliberately short - the handoff carries the
detail, and duplicating it here would create two versions that drift.

---

```
Picking up the neuromorphic cube project, Week 23. EXP-059 IS RUNNING on the laptop, ETA
around 2026-09-11 16:30 UTC. Everything that does not need the laptop is already done.

Read docs/handoffs/SESSION-HANDOFF-2026-09-09.md FIRST - sections 0c, 0d and 0e - then
CLAUDE.md. Ignore every earlier handoff.

FIRST ACTION: check the run with the THREE-READING TEST in
docs/playbooks/remote-experiment-runs.md, not by ssh exit code. Run age is measured from
2026-09-09 23:14 UTC. If uptime is LESS than the run age, or python processes are 0 while the
machine answers, it was rebooted; count records to price the loss. If ssh fails entirely the
state is UNKNOWN, not "paused" - a sleep and a reboot are indistinguishable from outside, and
misreading that as "nothing is lost" is how the first EXP-059 attempt was reported healthy for
13 hours while its workers no longer existed.

DO NOT WRITE aggregate.py. It already exists, with 22 tests, written from the spec while zero
records existed and verified by probe. Rewriting it from the numbers is exactly what the
pre-registration discipline exists to prevent. When 72 records land: scp them and the *_head.pt
back, RUN the existing aggregator, produce RESULTS.md, run the suite in chunks, merge --no-ff,
delete the branch, add the vault row.

MICHAEL NEEDS TO DEFER WINDOWS UPDATE on the laptop (Settings > Windows Update > Pause
updates). A TrustedInstaller reboot destroyed the first attempt 3.75 h in, costing ~19
CPU-hours, and the registry write is blocked by the permission classifier here. Still not done.

Five facts shape everything:

1. AN UNREACHABLE TAILSCALE PEER IS AN UNKNOWN, NOT A PAUSED JOB. The playbook used to say
   "offline, last seen 1h ago" meant the machine slept and the job was paused. That was false
   and it cost an experiment. The corrected three-reading test is uptime, python process
   count, record count - and nothing is durable until a cell COMPLETES, so a run always
   carries a rolling exposure of workers x per-cell-hours.

2. READ docs/retired-instruments.md BEFORE PUTTING ANY INSTRUMENT IN A SPEC. Five are retired:
   the EXP-033 probe, pretraining move-accuracy, the entropy trace, S, and critic_ev. Every one
   works as a THRESHOLD and fails as a GRADIENT. Use revisit_rate and optimality.

3. READ "THE GATE-CALIBRATION RULE" IN CLAUDE.md BEFORE WRITING OR READING A VALIDITY GATE.
   EXP-058 was VOID because its gate required mean_n_stored > 10, bounded by episode length
   where episodes average 7.76 steps. EXP-059's gate was calibrated across its full range
   before its spec: empty attractor 1.000000, real depth-5 run 0.8128, threshold 0.95.

4. EXP-059's primary is M vs A and it is NOT directional. A significant -0.05 is a real finding
   that memory HURTS, not a failed confirmation. EXP-030's trap is that memory beats the
   shuffle-null while losing to the amnesic control, and EXP-058 reproduced it exactly.

5. DO NOT SCALE A MEASURED PER-CELL COST TO A NEW DEPTH BY HAND - use the playbook's steps()
   helper. EXP-059 was priced at 2.5 h/cell and measured 3.05, because the estimate multiplied
   the step-budget ratio by the stage-count ratio and a curriculum SPLITS a fixed episode count
   across stages. steps() predicted 3.03 against the measured 3.05.

NEXT AFTER EXP-059: dispatch EXP-060, the independent replication of EXP-056. Spec is written
and pre-registered; ~14 h. Its primary is seeds 14-23 alone, NOT the pooled n=22, because
extending an experiment because its p-value was marginal and then pooling is optional stopping.
```

---

## Why it is shaped that way

It opens with the run and the three-reading test, because the expensive failure this month was not
the Windows Update reboot - it was reporting a dead run as healthy for 13 hours on the strength of a
playbook table that could not tell a sleep from a reboot.

Three things most likely to be lost otherwise:

**The aggregator already exists.** A fresh session that writes one from the records would destroy
the single property that makes it trustworthy, and it would look like diligence while doing it.

**Nothing is durable until a cell completes.** Records land only on completion, so "the run has been
going for hours" says nothing about how much survives an interruption. Before wave 1, the answer is
nothing.

**Cost estimates here are wrong in a consistent direction.** Two were corrected this week: EXP-059
was under-priced by hand-scaling a measured cell, and EXP-060 was over-priced by ~11 h because
nobody checked which encoders already existed. Check the inventory and use `steps()`.
