# Prompt for a fresh session

Paste the block below into a new session. It is deliberately short - the handoff carries the
detail, and duplicating it here would create two versions that drift.

---

```
Picking up the neuromorphic cube project, Week 24/25. NOTHING IS RUNNING, the laptop is FREE,
main is clean at 5f79a62, suite 601 passing under -m "not slow" (623 total).

Read docs/handoffs/SESSION-HANDOFF-2026-09-17.md first, then CLAUDE.md. Ignore earlier ones.

WEEK 25 IS THE PHASE 3 CHECKPOINT and it is the largest remaining work: 62 numbered
experiments need one coherent narrative. It is laptop-free. Start from
docs/retired-instruments.md, the three closed lines below, and weekly notes 16-24.

PHASE 3'S HONEST STORY IS THAT THREE LINES CLOSED AND TWO CLOSED NEGATIVE:
- The CRITIC is closed and REPLICATED: the benefit is within-episode state-dependence
  (EXP-056/057), replicated on fresh seeds at -0.0925, p 0.0156 (EXP-060).
- EPISODIC MEMORY is closed NEGATIVE with a mechanism: it HURTS (-0.0954, p 0.0056) and the
  recall is NOISE - matched-magnitude noise performs the same (EXP-059/061).
- The NEUROMORPHIC CLAIM for Stage 3 is REFUTED: the neuromod bus bought nothing (EXP-053).
- STAGE 4, the stated deliverable, is PRICED OUT at ~33 days/seed (EXP-062).
- The FRONTIER is depth 8 at 0.0783, against a chance floor of exactly 0.0000.

Week 20's spiking-encoder training remains the ONE genuinely neuromorphic change that pays.
The critic that did work is ordinary RL. Say that plainly in the checkpoint.

MICHAEL NEEDS TO PAUSE WINDOWS UPDATE. PAUSE_VALUES still reads NONE, checked 2026-09-16.
He paused on 09-11 and TWO runs have died since, both at ~03:31 laptop-local (ActiveHours end
at 03:00 and Windows takes the first slot ~30 min later). Before any dispatch still running at
03:31 local / 07:31 UTC, confirm Settings shows "Updates paused until <date>".

Five facts shape everything:

1. THE CENSORING TRAP, from EXP-062 and the most transferable thing this week produced. Success
   is bounded below at zero, so an arm nearing the floor MUST show a smaller absolute decline.
   Averaging a floor-compressed step into an exchange rate under-prices badly: it would have
   put Stage 4 at 9.4 days/seed instead of 33. Use steps whose endpoints are both clear of zero.

2. CONFIRM COMPLETION FROM COUNTS, NEVER THE LOG. The log prints one success rate per line, and
   reading it to check a run finished contaminates the aggregator you have not written yet.
   EXP-060 had to disclose that; EXP-061 and EXP-062 avoided it.

3. AN INSTRUMENT THAT CANNOT DETECT THE DEFECT IT EXISTS FOR is the gate-that-cannot-fail trap
   in disguise, and only MUTATION TESTING finds it. Break the implementation deliberately and
   check each test fails against the bug it names.

4. READ docs/retired-instruments.md BEFORE PUTTING AN INSTRUMENT IN A SPEC, and "THE
   GATE-CALIBRATION RULE" in CLAUDE.md before writing a gate. Seven gates used, two were wrong.

5. OPERATIONAL: an unreachable tailscale peer is an UNKNOWN, not a paused job. Nothing is
   durable until a cell COMPLETES. Price from a SAME-worker-count measurement - cross-count
   scalings came in low five times. Pick a worker count that DIVIDES the cell count; collapsing
   two waves into one is what saved EXP-062's re-run.

THE LIVE SCIENTIFIC THREAD, deliberately not started: can any readout use the stored content?
EXP-059's gate showed the content IS there (recall_content_cos 0.8514) and this readout cannot
use it. MemoryReadout concatenates a RAW hippocampal read; a readout that learned what to
attend to is untested. It is new science that will not resolve before the checkpoint.
```

---

## Why it is shaped that way

It leads with the checkpoint rather than the last experiment, because week 25's deliverable is a
synthesis and the risk is a session spending its time on one more arm instead.

Three things most likely to be lost otherwise:

**Two of the three closed lines closed NEGATIVE**, and the checkpoint is more credible for saying
so directly - memory hurts, the neuromorphic claim is refuted, the deliverable is priced out. The
temptation is to lead with the critic because it worked.

**The censoring trap is a general statistical result, not a cube fact.** It will recur anywhere a
bounded metric is extrapolated, and it changed Stage 4's price by 3.5x.

**Windows Update has killed two runs since it was supposedly paused.** Any plan that dispatches
overnight without checking is planning to lose a wave.
