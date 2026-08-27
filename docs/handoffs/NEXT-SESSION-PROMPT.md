# Prompt for a fresh session

Paste the block below into a new session. It is deliberately short - the handoff carries the
detail, and duplicating it here would create two versions that drift.

---

```
Picking up the neuromorphic cube project. Nothing is running; the laptop is idle and the
repo is clean at 54f2510.

Read docs/handoffs/SESSION-HANDOFF-2026-08-27.md first, then CLAUDE.md. There is a
decision waiting in section 0 - do not start building until it is settled.

Where we are: a spiking network solves 2x2 cubes at depth 6 with 0.3525 held-out, and
depths 3-7 all work given budget. Three facts shape everything:

1. The depth series was a BUDGET series. Success is linear in the LOGARITHM of spend,
   ~0.22 per log10, no knee. Every depth number written before EXP-046 means "at 10,000
   episodes".

2. The encoder can be trained, and it is bounded. Fine-tuning during RL works (EXP-047),
   the gain is in the encoder not its head (EXP-048), it is RL's objective and not merely
   more gradient (EXP-050) - but it does NOT compound (EXP-049, constant ~+0.05 per round)
   and its advantage ERODES with depth (EXP-051: 1.69x cheaper than budget at depth 6,
   1.09x at depth 7).

3. THREE INSTRUMENTS NOW MOVE AGAINST POLICY QUALITY. The EXP-033 probe fell 0-12 at
   p 0.0005 while policy rose, rose 12-0 at p 0.0005 while policy halved, and pretraining
   move-accuracy climbs monotonically while policy collapses. USE revisit_rate and
   optimality instead - they are in every record since EXP-029. Do not put a probe number
   in a new spec.

YOUR JOB THIS SESSION: settle the Stage 3 direction, then design it.

NeuromodBus is a 46-line stub whose learning_enabled property nothing reads. The fork:
- 3a: a learned critic, TD error as the advantage. Targets the MEASURED failure (EXP-045's
  depth-7 entropy collapse, 0.591 -> 0.098). Safer. But it is ACTOR-CRITIC, and routing
  delta through a bus object does not make it neuromorphic.
- 3b: gate encoder plasticity on learning_enabled. This is the increment where the bus
  becomes load-bearing and the neuromorphic claim becomes honest.

CLAUDE.md and road-to-a-solved-cube both say nothing neuromorphic participates in the
learning yet. Week 20 changed that once. Do not let 3a be written up as if it changed it
twice. Ask Michael which way before building; the previous session deliberately did not
choose.

If he wants something cheaper instead, the best-value alternative is measuring the
pretraining plateau's LEFT edge: between 0 epochs (0.0000) and 10 (0.2012) nobody has
looked, and if 3 epochs also gives ~0.20 then pretraining is doing far less than the
EXP-039/040 story implies. ~4.5 h.

Pre-register the contract before the numbers exist, n >= 12 seeds, measure the floor
rather than assuming it, and price any compute confound against EXP-046's budget curve.
Two process failures last week are worth reading in section 5 of the handoff: both
happened with every threshold obeyed.
```

---

## Why this prompt is shaped this way

- **It names the file to read rather than restating it**, so the two cannot drift apart.
- **It leads with the three standing facts, not the newest result.** A fresh session that knows
  only EXP-052 will mis-plan; a session that knows budget is log-linear, the encoder line is
  bounded, and the instruments are inverted will not.
- **It carries the retired instrument loudly.** Three experiments now show it moving the wrong
  way, and it is the single easiest mistake for a fresh session to make, because the probe is what
  EXP-033/039/047 all used and the vault is full of its numbers.
- **It refuses to make the Stage 3 choice.** The previous session declined twice on purpose. A
  prompt that quietly picked one would erase that, and 3a is the option that is easy to ship and
  easy to over-claim.
- **It names a cheaper fallback**, so "do something tonight" never competes with "settle the
  direction properly".
