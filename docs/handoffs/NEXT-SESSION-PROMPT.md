# Prompt for a fresh session

Paste the block below into a new session. It is deliberately short - the handoff carries the
detail, and duplicating it here would create two versions that drift.

---

```
Picking up the neuromorphic cube project, Week 23/24. NOTHING IS RUNNING and the laptop is
FREE. main is at 054db85, clean, no branches. Suite is 591 passing under -m "not slow".

Read docs/handoffs/SESSION-HANDOFF-2026-09-11.md first, then CLAUDE.md. Ignore every earlier
handoff.

EXP-059 IS DONE AND MERGED, and the answer is that episodic memory HURTS a policy that works:
M minus A is -0.0954 at p 0.0056, clearing the pre-registered bar downward and clearing
Bonferroni. That reading was fixed in the spec beforehand as a real finding rather than a
failed confirmation. Arms: amnesic 0.3138, memory 0.2183, shuffled 0.1979.

The sharpest part is that correct memory is INDISTINGUISHABLE from wrong memory: M minus S is
+0.0204 at p 0.4268. Reading the attractor at all costs 0.095; the content being correct buys
nothing measurable. Arm A is not "no recall" - 65% of its recall block is a memory-free
transform of the current concept - so adding stored content is precisely what costs.

NEXT UP, and a decision for Michael:

1. EXP-060 is specced and ready (docs/superpowers/specs/2026-09-10-exp060-flattened-critic-
   replication-design.md, ~14 h), an independent replication of EXP-056 whose primary is seeds
   14-23 ALONE, not the pooled n=22, because extending an experiment because its p-value was
   marginal and then pooling is optional stopping. DO NOT DISPATCH UNTIL WINDOWS UPDATE IS
   DEFERRED on the laptop (Settings > Windows Update > Pause updates) - a TrustedInstaller
   reboot destroyed EXP-059's first attempt 3.75 h in, costing ~19 CPU-hours, and the registry
   write is blocked by the permission classifier so it needs Michael.

2. The live scientific thread is WHY the read hurts. EXP-059's gate ruled out an EMPTY
   attractor, which is far narrower than ruling out a badly scaled or uninformative recall
   code. This is more interesting than EXP-055's two leads.

Five facts shape everything:

1. READ docs/retired-instruments.md BEFORE PUTTING ANY INSTRUMENT IN A SPEC. Five are retired;
   every one works as a THRESHOLD and fails as a GRADIENT. Use revisit_rate and optimality.

2. READ "THE GATE-CALIBRATION RULE" IN CLAUDE.md BEFORE WRITING A GATE. EXP-059's is the fourth
   gate and the first right by construction: calibrated across its full attainable range before
   the spec existed, able to both pass and fail, and a bounded scale-free cosine.

3. A TWO-ARM MEMORY DESIGN WOULD HAVE PUBLISHED A WIN FOR THE THIRD TIME. Memory versus
   shuffle-null measures +0.0204 here and reads as a win; only the amnesic arm shows both arms
   losing by ~0.10. Three arms is the minimum that answers the question, not thoroughness.

4. THE CRITIC QUESTION IS CLOSED - the benefit is within-episode state-dependence, not
   calibration. Everything without it sits 0.1358-0.1558; the full critic sits at 0.2004.

5. OPERATIONAL: an unreachable tailscale peer is an UNKNOWN, not a paused job - that misread
   let a dead run be reported healthy for 13 hours. Nothing is durable until a cell COMPLETES.
   The Bash default timeout is 120 s and 600 s is a hard ceiling. Never pass comma-separated
   arguments over ssh; that failure EXITS ZERO. Do not scale a per-cell cost by hand - use the
   playbook's steps() helper.

Vault 4f13 ("re-ask the EXP-030 memory question") IS EXP-059 and can be ticked, but that is
Michael's call rather than something to tick because the work looks finished.
```

---

## Why it is shaped that way

It leads with the finding rather than the status, because EXP-059's result inverts the assumption
the whole memory line was built on and a session that skims will carry the old one forward.

Three things most likely to be lost otherwise:

**"Memory hurts" is a pre-registered reading, not a disappointed null.** The spec fixed that wording
before any number existed precisely so it could not be softened into "memory did not help".

**M versus S is the sharp result, and it is the contrast most often misread.** Correct and incorrect
memory are indistinguishable; the harm is in the read. EXP-030's headline came from this same
contrast and was misread for months.

**The laptop being free is not permission to dispatch.** Windows Update is still not deferred, and
that is what destroyed EXP-059's first attempt.
