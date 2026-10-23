# EXP-081: was the 4000-update budget the limit? O-Xt continued to 16,000 updates

Date: 2026-10-23. Chosen by Michael on neuromorphic#27 ("longer-budget O-Xt"), design approved in
chat the same night ("yea that works and build then run").

## 1. The question

EXP-080 read **ENCODER-LIMITED**: given the true BFS distance as its target, a fresh head on X's
fine-tuned spiking encoder (O-Xt) beat X's far-band margin (probe distances 9 to 11) by only
+0.0228 (p 0.0547), while a raw-facelet MLP beat it by +0.1778. Its one open caveat is the budget:
O-Xt trained 4000 updates and its loss was still drifting down (about 3.1 to 2.9 over the last
1,250). This experiment asks **whether 4x the budget changes the verdict**.

Distance-to-solved stays a diagnostic target only. O-Xt is an instrument that prices what X's
encoder can carry, never a candidate system.

## 2. Design: continue, do not retrain

EXP-080's O-Xt is continued from its update-4000 checkpoint to update 16,000. On the machine and
checkout that produced it, this is the same run as a fresh 16,000-update run, because:

- the optimizer (EXP-075 arm W: Adam, head 1e-3, encoder 1e-4) has a **constant** learning rate,
  and each checkpoint holds the optimizer state (`opt.pt`) and the encoding generator's state;
- training reads block u of the mint at update u, and the mint draws its blocks in order from one
  `random.Random(seed)`. A 16,000-block mint therefore has EXP-080's 4,000 blocks as its prefix
  and adds 12,000 new ones (Gate C (a) checks this rather than assuming it);
- resuming equals an uninterrupted run, which EXP-080's test suite already pins.

So the 4000-update point of every curve is EXP-080's published O-Xt, and every update past it sees
states no O-Xt judge has seen (each state is still seen once).

A fresh 16,000-update run is not done: it is equivalent and costs about 3 h more. O-Xf is not
continued: its loss was flat over its last 1,250 updates. O-raw and X are not retrained: X is
EXP-075's committed records, as in EXP-080.

## 3. Inputs and outputs

- Code: `experiments/081_oxt_long_budget/`, importing EXP-080's modules unchanged.
- Mint: EXP-080's `mint_seed` with `n_blocks = 16000`, block 1000, max length 40, same exclusion
  (held-out states, probe states and their 3-move leaves), written to
  `experiments/081_oxt_long_budget/outputs/mint/` (about 400 MB per seed, untracked).
- Source checkpoints: EXP-080's `outputs/judge_O-Xt_s<seed>/` (`judge.pt`, `state.json` tracked;
  `opt.pt` on the laptop only).
- Seeds 0 to 11. No pilot (section 10).

## 4. Gate C, continuity (voiding)

Checked per seed before that seed's continuation is created; a failure refuses the seed and
the experiment is VOID.

- **(a) Mint prefix.** The new mint's first 4,000,000 states and labels, and its probe states,
  probe distances, leaf states and leaf distances, are byte-identical to EXP-080's mint for that
  seed.
- **(b) Starting point.** EXP-080's checkpoint is at update 4000, and its judge, re-measured by
  `leaf_rank_margin` on the new mint's probe and leaf distances, equals EXP-080's committed
  `gate_l` for that seed exactly; and its optimizer state is the one the continuation depends
  on: every Adam parameter state in `opt.pt` has `step` 4000 and the shape of the judge
  parameter it belongs to.

Only then is the checkpoint copied into EXP-081's directory and its `mint_sha` re-stamped with
the new mint's sha (EXP-080's resume path refuses a checkpoint from another mint, correctly; the
re-stamp is legitimate only because (a) proved the first 4000 blocks identical).

## 5. Training

`train_x_arm` from EXP-080, unchanged, in two legs per seed:

1. to update **8000**: record `exp081_O-Xt_u8000_s<seed>.json` (gate_l on the minted probe,
   encoder drift, loss) and snapshot `judge_u8000.pt`;
2. to update **16,000**: record `exp081_O-Xt_u16000_s<seed>.json`.

Leg 1 is skipped when its record exists, so a run killed in leg 2 resumes in leg 2 and is never
re-recorded as the 8000 point. Checkpoints every 250 updates, as in EXP-080, so the 03:31 restart
costs at most 250 updates per run. 6 workers (EXP-080's measured 2.2 GB committed per worker,
plus about 0.3 GB for the larger mint).

## 6. Gates

- **Gate C** (section 4), voiding.
- **Gate T** (trained), EXP-080's unchanged: mean overall margin of O-Xt16k above **0.13564**
  (half X's 12-seed mean overall margin) by `gate_l_verdict`. Fails: UNRESOLVED (O-Xt16k
  UNTRAINED). Its line prints far-band and overall margin at 4000 and 16,000 side by side,
  because EXP-080 showed oracle training trades overall margin for far band (O-Xt 0.2499
  against X's 0.2713), so a fall in overall margin with a rise in far band is a trade, not an
  untrained judge, and the reader must be able to see which happened.
- Record checks: `n_updates` 16000, `resumed_from` at least 4000 (0 would mean a fresh run, not a continuation), `mint_meta`
  n_blocks 16000, encoder drift above 0, the probe's per-distance n equal to X's.

## 7. Claims

One-sided exact sign-flip over all 4096 flips, paired by seed, alpha **0.025** each (0.05 / 2).

| claim | contrast | CONFIRMED when |
|---|---|---|
| 1 | O-Xt16k minus X, far band | p < 0.025 and mean >= 0.05 (EXP-080's Claim 1 bar, unchanged); p < 0.025 with a smaller mean is CONFIRMED (SMALL) |
| 2 | O-Xt16k minus O-Xt4k (EXP-080's records), far band | p < 0.025 and mean > 0 |

**Power.** EXP-080's per-seed O-Xt minus X far-band differences have sd about 0.043, so the se at
n = 12 is about 0.0125. Claim 1's 0.05 bar sits about 2.2 se above EXP-080's +0.0228, so 4x the
budget must add about 0.027 of far band to clear it. Claim 2 resolves a gain of about 0.03 (2.4
se) and cannot resolve one much under 0.02; a NOT CONFIRMED Claim 2 says the budget did not move
the far band by more than about that, not that it moved it by nothing.

## 8. Verdict (in the aggregator, one line)

| condition | VERDICT |
|---|---|
| Gate C failed on any seed | VOID |
| Gate T failed | UNRESOLVED (O-Xt16k UNTRAINED) |
| Claim 1 CONFIRMED | BUDGET-LIMITED, then the search reading (section 9) |
| otherwise, Claim 2 CONFIRMED | BUDGET HELPS BUT FALLS SHORT |
| otherwise | ENCODER-LIMITED AT 4X BUDGET |

Claim 1 CONFIRMED (SMALL) falls through to the Claim 2 rows. BUDGET-LIMITED overturns EXP-080's
reading; the other two keep it, with the budget caveat closed at 4x.

## 9. Search (only if Claim 1 CONFIRMED)

EXP-080's search cell unchanged (J4V, mode C, k 4, goal check at 4, no-revisit, depth 11,
EXP-075's held-out sets and streams) with O-Xt16k as the judge, against EXP-076's J4V-X records,
read by EXP-080's `search_reading` (alpha 0.05, minimum 0.03, Gate 1 floor and ceiling):
BUDGET-LIMITED, PAYS / DOES NOT PAY AT DEPTH 11 / SEARCH UNRESOLVED. Otherwise not run and not
read.

## 10. No pilot

EXP-080 measured everything a pilot would: O-Xt's cost (1.3 s per update), memory (2.2 GB
committed per worker), the mint's cost, and the gates' attainability (Gate T passed at 0.2499
against 0.13564). The code adds a mint length, a continuation and a second record, each covered by
tests. The schedule is fixed here, not by a later amendment: mint 6 workers, train 6 workers.

## 11. Reported, never confirmations

- Far band at 4000 / 8000 / 16,000 per seed and on average (the curve's shape).
- Margins at distances 7 and 8 at each point, against X (does the near-goal regression in EXP-080,
  0.50 against X's 0.69 at distance 7, recover or deepen).
- Overall margin, `spearman_7_11` and mean J at distances 9 to 11 from the 250-update probes.
- Training loss at 4000, 8000 and 16,000.

## 12. Cost and schedule

- Mint: 16M states per seed, 12 seeds at 6 workers, about 45 min.
- Gate C: minutes.
- Training: 12,000 updates at about 1.3 s, about 4.3 h per run; 12 runs at 6 workers in two
  waves, about 8.7 h.
- Search, only if Claim 1 confirms: about 1.5 h at 6 workers.
- Total about 9.5 h without the search. Launched Friday 2026-10-23 night, so it crosses 03:31; a
  restart costs at most 250 updates per run. Every chain stage skips finished work, and a VPS
  watcher polls the laptop and re-runs the chain if it died, until the verdict file exists.

## 13. What this does not answer

- Whether any **other** encoder carries the ordering. A BUDGET-LIMITED reading says this one does
  with more training; the other two readings send the next stage to the encoder or region.
- Whether O-Xf would move with more budget (its loss was flat, so it is not run).
