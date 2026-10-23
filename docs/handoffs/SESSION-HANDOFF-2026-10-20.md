# Session Handoff - 2026-10-20 - WEEK 29 closed, WEEK 30 open. EXP-080 DONE: ENCODER-LIMITED.

> **2026-10-23: week 29 closed.** EXP-080 is written up everywhere: RESULTS.md `f9307f0`, README and
> roadmap `ea98027`, report sections 1, 8 and 9 `11de006`. Vault: weekly note
> `week-29-the-encoder-is-the-limit`, tracker row 29 done and row 30 opened, weekly index
> brought up to week 29. **Next stage is Michael's call on neuromorphic#27.** The laptop is idle.
>
> **This supersedes `SESSION-HANDOFF-2026-10-12.md`** (weeks 27 and 28, EXP-076 to 079). Read that
> one, and its predecessor `SESSION-HANDOFF-2026-10-06.md` section 3 (the gotchas), for history;
> both still hold.

## 0. Week numbering

Week 29 = Oct 19-25 (closed), week 30 = Oct 26 - Nov 1 (now). The gap week of Sep 28 counts.
Sessions are counted by night in Pacific time. Planned cadence from week 30: **build Monday, run
Wednesday** (Michael, 2026-10-21).

## 1. Week 29 in one paragraph

After EXP-079 refuted retraining the judge on longer walks, Michael chose a ceiling check. EXP-080
trains judges directly on true BFS distance, as a diagnostic upper bound, to ask whether any judge
on X's spiking encoder can rank states 9 to 11 moves from solved better than X. It cannot.
- A fresh head on the fine-tuned encoder (O-Xt) gains +0.0228 far-band margin (p 0.0547, Claim 1
  REFUTED).
- On the frozen encoder (O-Xf) it gains +0.0106 (Claim 2 REFUTED).
- A plain MLP on raw facelets with the same labels (O-raw, Gate V) gains +0.1778, ahead on 12 of 12
  seeds.

**VERDICT: ENCODER-LIMITED.** Every gate passed and the search arm was not run.

The work ran over two nights:
- **Tue 20 night:** specced, planned, built and merged (`005773e`, 1225 tests), then piloted.
- **Wed 21:** full run 18:24 to 23:32 laptop time (5.1 h).

## 2. EXP-080 facts a next stage will need

- **Where the code and records are:**
  - Code: `experiments/080_oracle_ordering/`.
  - Spec: `docs/superpowers/specs/2026-10-20-exp080-oracle-ordering-ceiling-design.md` (section
    13 is the dated pilot amendment).
  - Plan: `docs/superpowers/plans/2026-10-21-exp080-oracle-ordering.md`.
- **Tracked:**
  - 36 training records.
  - Gate 0's record.
  - Every judge's `judge.pt` and `state.json` under `outputs/`.
  - The pilot under `outputs_pilot/`.
- **The mint files are untracked** and stay on the laptop under `outputs/mint/` (about 4M states
  per seed). They hold states, true labels, probe states and probe-leaf distances. Each training
  record carries its mint's sha256. A follow-up that reuses the same draws should read them rather
  than re-mint.
- **Measured costs (laptop, 6 workers):**
  - O-Xt (spiking encoder fine-tuned): about 1.3 s per update, 87 to 92 min per 4000-update run,
    2.2 GB committed per worker.
  - O-Xf: about 45 min per run.
  - O-raw: about 13 min per run.
  - Mint: 11 min for 12 seeds.
  - A 200-state J4V depth-11 cell: about 40 min (pilot timing cell).
- **The open caveat is the budget.** O-Xt's loss was still drifting down at 4000 updates (about 3.1
  to 2.9 over the last 1,250). A longer-budget O-Xt is the cheapest way to rule the budget in or
  out before building a new encoder.
- **X-encoder oracle arms rank below X near the goal** (distance 7: 0.50 against 0.69). Any new
  encoder should be checked at distances 7 and 8 as well as 9 to 11.

## 3. Week 30 (Oct 26 - Nov 1)

1. **Read neuromorphic#27 first.** Do not choose the next stage without Michael. The candidates in
   the ask: a wider or different encoder or region; a longer-budget O-Xt; something else.
2. When he says "build it and run it", that covers the full run if the stop rule does not fire.
   Write the amendment, launch, and report pilot warnings in the launch message and with
   `pm note`. Do not park the run behind a go-or-hold ask (week 29 lost about 13 laptop hours
   that way).
3. The vault experiment log (`experiment-log.md`) stops at EXP-078; EXP-079 and EXP-080 rows are
   not written yet.

## 4. What NOT to do

- Do not read O-Xt's p 0.0547 as "nearly confirmed". Its mean, +0.0228, is under half the 0.05
  bar, so it would read SMALL even at p < 0.025.
- Do not compare O-raw with the X arms as a matched competitor. It is the protocol check: it trains
  on 600k states for 12 epochs, where the X arms see 4M states once.
- Do not use distance-to-solved as a model input. EXP-080's labels are a diagnostic ceiling only,
  never part of the system.
- Do not schedule work against a publishing date. There is none.
