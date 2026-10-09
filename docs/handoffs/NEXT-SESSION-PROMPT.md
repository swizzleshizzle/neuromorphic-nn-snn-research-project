# Prompt for a fresh session

Paste the block below into a new session. It is deliberately short - the handoff carries the
detail, and duplicating it here would create two versions that drift.

---

```
Picking up the neuromorphic cube project. NOTHING IS RUNNING, the laptop is FREE.
The goal: an SNN that solves a 2x2 cube, via look-ahead plus a learned judgement. EXP-070 to 072:
the policy head and critic are not judges; policy look-ahead plus no-revisit (P3V) moved depth 9
to 0.063. EXP-074 (superseding 073): a judge trained by value iteration from its own look-ahead,
reading all 192 sensory neurons, CONFIRMED at depth 9: 0.193 vs P3V 0.063, and 0.101 vs 0.031 at
depth 11 (exploratory).

Read docs/handoffs/SESSION-HANDOFF-2026-10-06.md first (it covers 2026-10-06 to 09), then
CLAUDE.md. Ignore earlier ones.

Section 2b of the handoff lists what could come next (wider sensory region, deeper search,
judge-trained policy, learned world model). Ask me which before starting anything.
```
