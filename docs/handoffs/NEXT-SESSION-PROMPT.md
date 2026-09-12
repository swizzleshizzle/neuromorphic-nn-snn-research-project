# Prompt for a fresh session

Paste the block below into a new session. It is deliberately short - the handoff carries the
detail, and duplicating it here would create two versions that drift.

---

```
Picking up the neuromorphic cube project, Week 23/24. NOTHING IS RUNNING, the laptop is FREE,
main is clean at 1c82a33, suite 591 passing under -m "not slow".

Read docs/handoffs/SESSION-HANDOFF-2026-09-12.md first, then CLAUDE.md. Ignore earlier handoffs.

TWO EXPERIMENTS LANDED AND THEY POINT OPPOSITE WAYS.

EXP-059: episodic memory HURTS a working policy. M-A = -0.0954 at p 0.0056. And correct memory
is INDISTINGUISHABLE from wrong memory (M-S = +0.0204, p 0.4268), so the harm is in READING the
attractor, not in its content being wrong. Arm A is not "no recall" - 65% of its recall block
is a memory-free transform of the current concept - so adding stored content is the harm.

EXP-060: EXP-056 REPLICATES on fresh seeds. F-B = -0.0925 at p 0.0156, larger than the original
-0.0646. The critic conclusion no longer rests on one knife-edge contrast.

DO NOT QUOTE EXP-060's POOLED n=22 FIGURE (-0.0773, p 0.0005) WITHOUT BOTH CAVEATS: it inherits
optional stopping, AND it pools across a level shift of 0.05-0.08 in both arms between seed
blocks. The contrast replicates because it is paired within seed. Quote Claim 1 instead.

THE MOST INTERESTING OPEN QUESTION: why does the memory read hurt? EXP-059's gate ruled out an
EMPTY attractor, which is far narrower than ruling out a badly scaled or uninformative recall
code. That is the live scientific thread, ahead of EXP-055's two leads.

Five facts shape everything:

1. READ docs/retired-instruments.md BEFORE PUTTING AN INSTRUMENT IN A SPEC. All five work as a
   THRESHOLD and fail as a GRADIENT. Use revisit_rate and optimality.

2. READ "THE GATE-CALIBRATION RULE" IN CLAUDE.md BEFORE WRITING A GATE. Five used; two were
   wrong. A gate's pre-registered wording BINDS even when you find a looser precedent -
   EXP-060's spec said every-stage where EXP-056's code used any-stage, and every-stage was
   implemented.

3. AN UNREACHABLE TAILSCALE PEER IS AN UNKNOWN, NOT A PAUSED JOB. That misread let a dead run be
   reported healthy for 13 h after Windows Update destroyed EXP-059's first attempt, ~19
   CPU-hours. Use the three-reading test: uptime, python process count, artifact count.

4. CROSS-WORKER-COUNT COST ESTIMATES UNDER-PRICE - four instances. Treat them as a floor, add
   20-40%, and read wave 1 before trusting a total. Check the artifact INVENTORY before pricing:
   EXP-060 needed an EXP-043 depth-6 baseline nobody had costed, found at dispatch.

5. A TWO-ARM MEMORY DESIGN WOULD HAVE PUBLISHED A WIN FOR THE THIRD TIME. Three arms is the
   minimum that answers the question, not thoroughness.
```

---

## Why it is shaped that way

It leads with both findings because they pull in opposite directions - one mechanism refuted, one
confirmed - and a session that skims will flatten that into a single mood.

Three things most likely to be lost otherwise:

**EXP-059's "memory hurts" is a pre-registered reading, not a disappointed null**, and M-versus-S
is the sharp part: correct and incorrect memory are indistinguishable.

**EXP-060's pooled figure is the most quotable number in the repo and the least trustworthy.** It
has two independent problems, and the tempting p 0.0005 is exactly why both caveats travel with it.

**The open question is now mechanistic, not evaluative.** "Does memory help" is answered. "Why does
the read hurt" is not, and it is cheaper to probe than either experiment that produced it.
