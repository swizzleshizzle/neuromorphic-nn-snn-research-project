# EXP-074 results: a learned judge that reads the whole sensory region

**Both primaries CONFIRMED at depth 9, on all 12 seeds and on the 10-seed sensitivity line.**
A judge trained by value iteration from its own look-ahead (never from answers), reading all 192
sensory neurons, roughly triples the best recipe so far and beats the same judge reading only the
64-unit concept.

Spec: `docs/superpowers/specs/2026-10-07-exp074-wide-judge-design.md` (sections 11 to 13 are the
dated pilot amendment and the gate records). Plan: `docs/superpowers/plans/2026-10-07-exp074-wide-judge.md`.

## Disclosure first (spec section 2)

While diagnosing EXP-073, pilot judges were run through J3V on two EVALUATION cells (depth 9, seeds
0 and 3) before this spec existed, and those numbers motivated replacing EXP-073's training gate.
Every primary is therefore printed twice. **The two lines agree on both claims**, so there is no
disagreement to lead with.

| claim (depth 9) | 12 seeds | 10 seeds (0 and 3 dropped) |
|---|---|---|
| 1. J3V-W > P3V | +0.1304, p 0.0002, **CONFIRMED** | +0.1340, p 0.0010, **CONFIRMED** |
| 2. J3V-W > J3V-A | +0.0642, p 0.0010, **CONFIRMED** | +0.0605, p 0.0039, **CONFIRMED** |

One-sided exact sign-flip over all 4096 (12 seeds) or 1024 (10 seeds) flips, alpha 0.025 each.
The sensitivity line reuses the 12-seed gate verdicts; the spec does not say otherwise, and every
gate passed on every arm with wide margins, so recomputing on 10 seeds could not change a verdict.

## Mean held-out success (200 states per cell, 12 seeds)

| depth | J3V-W | J3V-A | J3V-B | P3V | R3V |
|---|---|---|---|---|---|
| 7 | **0.7908** | 0.6288 | 0.3488 | 0.3217 | 0.0033 |
| 8 | **0.3792** | 0.2725 | 0.1471 | 0.1496 | 0.0013 |
| 9 | **0.1933** | 0.1292 | 0.0663 | 0.0629 | 0.0004 |
| 11 (exploratory) | **0.1013** | 0.0596 | 0.0296 | 0.0308 | 0.0004 |

J3V-W: the wide judge (concept 64 + hidden 128 mean rates, encoder trained). J3V-A: the 64-unit
concept judge, encoder trained (EXP-073 arm A). J3V-B: as A, encoder frozen (EXP-073 arm B). P3V:
the best recipe so far (policy-scored 3-move tree, no-revisit). R3V: random-guided floor, same tree.

## Depth 9, per seed

| seed | J3V-W | J3V-A | P3V |
|---|---|---|---|
| 0 | 0.180 | 0.070 | 0.025 |
| 1 | 0.200 | 0.120 | 0.035 |
| 2 | 0.160 | 0.160 | 0.020 |
| 3 | 0.200 | 0.145 | 0.130 |
| 4 | 0.175 | 0.065 | 0.110 |
| 5 | 0.185 | 0.185 | 0.065 |
| 6 | 0.160 | 0.095 | 0.055 |
| 7 | 0.250 | 0.135 | 0.080 |
| 8 | 0.225 | 0.165 | 0.105 |
| 9 | 0.195 | 0.135 | 0.070 |
| 10 | 0.185 | 0.130 | 0.010 |
| 11 | 0.205 | 0.145 | 0.050 |

J3V-W beats P3V on **12 of 12** seeds and beats J3V-A on **10 of 12** (ties on seeds 2 and 5, no
losses).

**What the primaries could detect (spec section 8).** Measured paired sd at depth 9: W - P3V 0.0368,
W - A 0.0375, so se about 0.011 at n = 12. A difference of about 0.025 would have been detectable;
the observed +0.130 and +0.064 are about 12 se and 6 se.

## Gates (all PASS)

| gate | result |
|---|---|
| 0(a) determinism | J3V-W and J3V-A at depth 9 seed 0 re-run into a separate directory; each differs from its full-run copy only in `wall_s`. Records in `outputs_det/`. |
| 0(b) continuity | P3V at seed 0, depths 8 and 9, re-run equals EXP-072's records (47/200 and 5/200). |
| 0(c) concept path | A and B on pilot seeds 12 and 13 reproduce EXP-073 pilot 1's probe histories exactly (all 17 probes, four runs). |
| 1 resolution | no contrast in the 0.02 / 0.98 band; encoded in the verdict function (EXP-070's `gate1_verdict`). |
| E encoders | drift > 0 on every W and A seed; exactly 0.0 on every B seed. |
| L training worked | mean margin W 0.2243 (threshold 0.1018), A 0.1653 (0.0728), B 0.1159 (0.0368); p 0.0002 each. |
| R leaf ranking | every J arm at every depth, p 0.0002. Table below. |

**Gate R is the mechanism line.** The fraction of held-out states whose best-scored 3-move leaf is
closer to solved than the root (mean hit / chance), policy alongside:

| depth | chance | P (policy) | J-W | J-A | J-B |
|---|---|---|---|---|---|
| 7 | 0.1344 | 0.4458 | **0.7063** | 0.5458 | 0.3450 |
| 8 | 0.1486 | 0.3446 | **0.3658** | 0.3150 | 0.3021 |
| 9 | 0.1751 | 0.2792 | **0.3092** | 0.3088 | 0.2554 |

W has the best leaf rate at every depth, as it has the most solves. But leaf rate does not order the
other arms by solves: the policy's leaf rate exceeds B's at depth 7 and A's at depth 8, yet P3V solves
fewer than J3V-B at depth 7 and J3V-A at depth 8, and at depth 9 W and A have equal leaf rates (0.309)
while W solves 0.064 more. One-step leaf ranking at the root does not capture everything that
differs along a whole episode, so Gate R is a floor check, not a predictor of the claims.

## Secondary patterns (never confirmations)

All positive: depth 7 W - P3V +0.469 and W - A +0.162; depth 8 W - P3V +0.230 and W - A +0.107;
depth 9 A - P3V +0.066 and A - B +0.063. So EXP-073's two questions, asked at depth 9, also read
positive: the 64-unit learned judge beats the recipe, and training its encoder helps.

## Depth 11 (exploratory, no claims)

Each seed's DEPTH-9 policy agent, head and train seed (no depth-11 policy cell exists; amendment in
spec section 7), over exactly `heldout_states(11, seed)`, the set judge training excluded, with a
25-move budget. J3V-W solves 10.1% of typical random scrambles against P3V's 3.1%.

## Caveats and recorded rulings

- **P3V at depths 7 to 9 was not re-run.** It comes from EXP-071 (depth 7) and EXP-072 (8, 9)
  committed records. Gate 0(b) re-ran seed 0 at depths 8 and 9 on the same laptop and matched every
  outcome field. All EXP-074 cells ran on the same laptop.
- **`wall_s` on a resumed run would cover only the part after the resume** (parked at final review).
  Moot here: no run was relaunched, so every `wall_s` covers its whole run.
- **Launcher defect, found in the rank phase.** `Start-Process -PassThru` reports `ExitCode` as
  `$null` unless the process handle is read, so the rank phase reported 144 of 144 cells failed while
  writing 144 complete records. Reproduced on the laptop, fixed in `5ca5ead` before cont, eval and
  det (which then reported `PHASE OK`). The 144 rank records were validated directly: they parse,
  match their filenames, hold 200 states, and are written as each cell's last step. EXP-073's
  launcher has the same pattern; it never ran an evaluation phase.
- **Committed judges reproduce across machines.** `judge_W_s0` and `judge_B_s7`, loaded on the VPS,
  recompute their training records' Gate L margins to full float repr.
- Training wall time per run (W 16,901 s, A 14,766 s, B 20,900 s, means) reflects which wave ran
  and machine conditions, not per-arm cost.
- Training workers measured 0.69 to 0.94 GB each, about 4.5x the 195 MB CLAUDE.md records for cube
  evaluation workers.

## Provenance

- Machine: SwizzlesDuo laptop (Intel Ultra 9 185H, 22 cores, 31.4 GB), Windows, `.venv` Python 3.13.
- Pilot 2026-10-07/08, checkout `29fab9f`, 6 workers. Training 2026-10-08, `cf2a3d7`, 12 workers,
  three waves, about 14 h, no relaunches. Rank `48b3f4c`. Cont, eval and det `163e87a` (the launcher
  fix `5ca5ead` was copied into place before cont; it changes no Python). Eval 2026-10-08/09.
- Seeds 0 to 11 (pilot 12 and 13). Records: `outputs/exp074_train_*`, `exp074_rank_*`,
  `exp074_{J3V-W,J3V-A,J3V-B}_d{7,8,9,11}_s*`, `exp074_{P3V,R3V}_d11_s*`, `exp074_P3V_d{8,9}_s0`;
  judges `outputs/judge_{W,A,B}_s*/judge.pt`; pilot `outputs_pilot/`; determinism `outputs_det/`.

Regenerate, on the laptop from the main checkout:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File launch074.ps1 -Phase pilot -Workers 6
powershell -NoProfile -ExecutionPolicy Bypass -File launch074.ps1 -Phase train -Workers 12
powershell -NoProfile -ExecutionPolicy Bypass -File launch074.ps1 -Phase rank  -Workers 12
powershell -NoProfile -ExecutionPolicy Bypass -File launch074.ps1 -Phase cont  -Workers 2
powershell -NoProfile -ExecutionPolicy Bypass -File launch074.ps1 -Phase eval  -Workers 12 -SkipExisting
powershell -NoProfile -ExecutionPolicy Bypass -File launch074.ps1 -Phase det   -Workers 2
```

Aggregate (any machine with the committed records):

```bash
PYTHONPATH=src .venv/bin/python experiments/074_wide_judge/aggregate.py --determinism-ok
```

Retraining from a seed is not portable across machines (CLAUDE.md, reproducibility audit);
re-evaluating the committed judges is.
