# Prompt for a fresh session

Paste the block below into a new session. It is deliberately short - the handoff carries the
detail, and duplicating it here would create two versions that drift.

---

```
Picking up the neuromorphic cube project, Week 24. NOTHING IS RUNNING, the laptop is FREE,
main is clean at c52fe18, suite 601 passing under -m "not slow" (623 total).

Read docs/handoffs/SESSION-HANDOFF-2026-09-13.md first, then CLAUDE.md. Ignore earlier ones.

WEEK 23 LANDED THREE EXPERIMENTS AND CLOSED THE MEMORY LINE WITH A MECHANISM.

EXP-059: episodic memory HURTS a working policy. M-A = -0.0954, p 0.0056.
EXP-061: and the reason is that the recall is NOISE. Matched-magnitude noise is
indistinguishable from real memory (M-N = +0.0210, p 0.4989) and both cost ~0.1 against the
amnesic arm (N-A = -0.1165, p 0.0001). Four arms: A 0.3138 || M 0.2183, S 0.1979, N 0.1973 -
the three uninformative arms within 0.021 of each other, the memory-free transform 0.10-0.12
above all three. So the recall block is valuable for what it says about the CURRENT state, and
stored content destroys that as thoroughly as noise.
EXP-060: EXP-056 replicates on fresh seeds, F-B = -0.0925 at p 0.0156, larger than the
original -0.0646.

H1 IS SUPPORTED, NOT CONFIRMED - EXP-061's primary is a null and a null is a bound. What it
rules against is H2 (that stored content is actively misleading). N-S = -0.0006 is descriptive
with NO p-value: unregistered, and a post-hoc p would inflate a fixed multiplicity.

DO NOT QUOTE EXP-060's POOLED n=22 without both caveats: optional stopping, AND it pools across
a level shift of 0.05-0.08 between seed blocks.

THE NEXT QUESTION IS THE READOUT, NOT THE HIPPOCAMPUS. EXP-059's gate showed stored content
genuinely changes the recall (recall_content_cos 0.8514, well below 1.0), so the content IS
there and this readout cannot use it. MemoryReadout concatenates a raw hippocampal read; a
readout that learned what to attend to is untested and is the obvious follow-up.

Five facts shape everything:

1. AN INSTRUMENT THAT CANNOT DETECT THE DEFECT IT EXISTS FOR is the gate-that-cannot-fail trap
   in disguise, and only MUTATION TESTING finds it. EXP-061's leak detector was blind to a 50%
   leak because it compared two pre-substitution values. Break the implementation deliberately
   and check each test fails against the bug it names.

2. REUSING ANOTHER EXPERIMENT'S ARMS MUST BE PROVEN INERT, NOT ASSUMED. EXP-061 reused
   EXP-059's three arms only after verifying the added instruments left them identical to full
   float repr, with a regression test.

3. CONFIRM COMPLETION FROM COUNTS, NOT THE LOG. The log prints one success rate per line, and
   reading it contaminates the aggregator you have not written yet. Counts plus zero processes
   is the cheap fix.

4. READ docs/retired-instruments.md BEFORE PUTTING AN INSTRUMENT IN A SPEC, and "THE
   GATE-CALIBRATION RULE" in CLAUDE.md before writing a gate. Six gates used, two were wrong.

5. OPERATIONAL: an unreachable tailscale peer is an UNKNOWN, not a paused job. Nothing is
   durable until a cell COMPLETES. Price a run from a SAME-worker-count measurement; four
   cross-worker-count scalings came in low and the one same-count estimate held exactly.
   WINDOWS UPDATE IS STILL NOT VERIFIABLY PAUSED on the laptop - check before a long dispatch.
```

---

## Why it is shaped that way

It leads with EXP-059 and EXP-061 as a pair, because the second supplies the mechanism for the
first and either one alone invites the wrong summary: "memory hurts" sounds like a performance
quirk until you know the recall is literally noise.

Three things most likely to be lost otherwise:

**H1 is supported, not confirmed**, and the distinction is load-bearing. The primary is a null, and
the spec said before dispatch that a null is a bound - so the finding is that H2 is ruled against,
not that H1 is proven.

**The next question moved.** It is no longer "does memory help" - that is answered - but "can any
readout use content the hippocampus demonstrably stores". That is a different experiment.

**The mutation-testing lesson generalises past this project.** An instrument blind to the defect it
exists for passes every test, and no amount of reading the test list reveals it.
