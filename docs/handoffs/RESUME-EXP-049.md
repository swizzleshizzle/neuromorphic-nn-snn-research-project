# EXP-049 is in flight

**Dispatched 2026-08-23 22:01:51 laptop time from commit `8d316f0`.** Two sequential arms, ~15 h.
Do not dispatch anything else to `SwizzlesDuo` until it finishes.

## Where is it?

```bash
ssh -n laptop 'powershell -NoProfile -Command "cd C:\Users\mlgbr\Desktop\Projects\neuromorphic-nn-snn-research-project\experiments\049_second_round\outputs; Get-ChildItem *.log,*.json,*_encoder.pt -ErrorAction SilentlyContinue | Group-Object Extension | ForEach-Object { $_.Name + \" x\" + $_.Count }; @(Get-Process | Where-Object { $_.Name -match \"^python\" }).Count"'
```

| arm | writes | cost |
|---|---|---|
| **D** round-2 fine-tuning, E1 -> E2 | 12 `exp049_ft2_d6_*.json` + 12 `*_encoder.pt` | ~8.5 h |
| **E** E2 frozen + fresh head | 12 `exp049_fresh2_d6_*.json` | ~6.5 h |

Arm E is gated on all 12 E2 encoders existing. If the host sleeps, re-run
`launch049.ps1` - `--skip-existing` resumes from disk and the runs are seeded and
byte-identical across scheduling.

## When it finishes

```bash
scp 'laptop:C:/Users/mlgbr/Desktop/Projects/neuromorphic-nn-snn-research-project/experiments/049_second_round/outputs/*' experiments/049_second_round/outputs/
.venv/bin/python experiments/049_second_round/aggregate.py
```

## Read the result the way it was pre-registered

Spec: `docs/superpowers/specs/2026-08-24-exp049-second-round-design.md`, committed `aaca17c`.

> [!important] **Claim 1 is PREDICTED TO REFUTE.** The constant-return model fitted to arms C and
> B puts arm E near 0.354, i.e. `E - B` of about **+0.043**, below the +0.05 bar. Refutation is
> the prediction; **CONFIRMATION would be the surprise** and would mean the recipe compounds.
>
> The extra round's own compute is worth **+0.0432** on EXP-046's curve, so a delta between
> +0.0432 and +0.05 is pre-committed as **UNINTERPRETABLE**, not a near-miss.

**Claim 2 carries the answer**: arm E's excess over its budget-equivalent (0.3040), read beside
arm C's **+0.0628** and arm B's **+0.0504**.

- ~+0.05 -> **constant returns.** Each round buys a fixed increment over its own cost. Iterating
  is a better way to spend compute, not an escape from EXP-046's budget wall. The next move is a
  different second-stage objective, **not a round 3**.
- clearly >+0.08 -> **compounding.** Iterate further.
- clearly <+0.03 -> **diminishing.** Round 1 was special, most likely because E0 was pretrained
  on a different objective and had the most to gain.

Two falsifiable predictions from EXP-048's mechanism also ride along: `eval_revisit_rate` should
fall further and `optimality` rise further (Claim 4), and the probe should drift **down** again
(Claim 5, via `experiments/048_fresh_head/diagnose_probe_tension.py`).

## Context

The week's arc so far, all at depth 6, 10,000 episodes per stage:

| arm | what | compute | actual | excess over budget |
|---|---|---|---|---|
| A (EXP-043) | E0 frozen, fresh head | 1.00 | 0.1800 | - |
| C (EXP-047) | E1 fine-tuned, joint head | 1.33 | 0.2700 | +0.0628 |
| B (EXP-048) | E1 frozen, fresh head | 2.33 | 0.3112 | +0.0504 |
| D, E | **this experiment** | 2.66, 3.66 | ? | ? |
