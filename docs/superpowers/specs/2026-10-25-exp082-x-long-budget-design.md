# EXP-082: is value iteration budget-limited too? X continued to 16,000 updates

Date: 2026-10-25. Chosen by Michael on neuromorphic#28 ("option 1, train X 4x longer"), design
approved in chat the same day ("yes").

## 1. The question

EXP-081 read **BUDGET-LIMITED, PAYS**: a judge on X's encoder trained on true BFS distance for
16,000 updates beat X's far band by +0.0678 (p 0.0012) and, as the depth-11 search judge, solved
0.4088 against J4V-X's 0.1950. That judge is an oracle-trained ceiling, not a solver. X itself
was trained by value iteration from its own look-ahead (no true distances) for 4000 updates. This
experiment asks **whether X is budget-limited too, and whether 4x its budget pays in search**.
Unlike EXP-081, a positive result here is a real solver number.

Distance-to-solved stays an instrument only: it is the probe's yardstick and the search's success
test, never a training target or a model input.

## 2. Design: continue, do not retrain

EXP-075's X judge is continued from its update-4000 checkpoint to update 16,000, as arm **X16k**.
On the machine and checkout that produced it, this is the same run as a fresh 16,000-update run,
because `vi.train_judge` (`src/neuromorphic/training/value_iteration.py`):

- draws its walks online from one `random.Random(seed)` and its encoding from one
  `torch.Generator`, and checkpoints both states (`state.json`), so no mint is needed;
- checkpoints the judge (`judge.pt`), the target network (`jt.pt`) and the optimizer (`opt.pt`,
  EXP-075 arm W: Adam, head 1e-3, encoder 1e-4, constant rate) every `probe_every` updates;
- resumes from that checkpoint to the same trajectory as an uninterrupted run, which EXP-075's
  test suite pins.

So the 4000-update point of every curve is EXP-075's published X, and every update past it is new
training. The checkpoints exist only on the laptop (`experiments\075_wide_region\outputs\
judge_X_s<seed>\`, all four files on all 12 seeds; `judge.pt` and `state.json` are tracked and
the laptop copies hash equal to the tracked ones). Each is **copied** into EXP-082's directory
before training; EXP-075's directories are never written.

Settings are EXP-075's X, unchanged: sync_every 100, probe_every 250, draws 1, batch 1000,
max walk length 14, readout `wide`, hidden 512, EXP-075's exclusion set and probe set.

A fresh 16,000-update run is not done: it is equivalent and costs about 14 h more.

## 3. Inputs and outputs

- Code: `experiments/082_x_long_budget/`, importing EXP-075's `train.run`, `cells` and EXP-074's
  `gate_l` unchanged; it changes only where checkpoints and records are written.
- Source checkpoints: EXP-075's `outputs/judge_X_s<seed>/` on the laptop.
- Baselines: EXP-075's `exp075_train_X_s<seed>.json` (gate_l at 4000) and EXP-076's
  `exp076_J4V-X_d9_s<seed>.json` and `exp076_J4V-X_d11_s<seed>.json` (search), all tracked.
- Seeds 0 to 11. No pilot (section 10).

## 4. Gate C, continuity (voiding)

Checked per seed before that seed's copy is trained; a failure refuses the seed and the
experiment is VOID.

- **(a) Identity.** The source `judge.pt` and `state.json` hash equal to the tracked EXP-075
  files, and `state.json` says update 4000.
- **(b) Optimizer and target.** Every Adam parameter state in `opt.pt` has `step` 4000 and the
  shape of the judge parameter it belongs to; `jt.pt` loads into the judge's architecture.
- **(c) Starting point.** The source judge, re-measured by `gate_l.leaf_rank_margin` on EXP-075's
  probe for that seed, equals the committed `gate_l` in `exp075_train_X_s<seed>.json` exactly.

Only then is the checkpoint directory copied. The Gate C record (`exp082_gatec_s<seed>.json`)
holds the hashes, the `step` values and the re-measured margin.

## 5. Training

EXP-075's `run` for arm X, unchanged, in two legs per seed on the copied checkpoint:

1. to update **8000**: record `exp082_X16k_u8000_s<seed>.json` (gate_l, encoder drift, loss,
   the 250-update probes) and snapshot `judge_u8000.pt`;
2. to update **16,000**: record `exp082_X16k_u16000_s<seed>.json`.

Leg 1 is skipped when its record exists, so a run killed in leg 2 resumes in leg 2 and is never
re-recorded as the 8000 point. Checkpoints every 250 updates, so the 03:31 restart costs at most
250 updates per run. **4 workers**: EXP-075 measured 4.16 s per update at 4 workers and about
12 s at 8 (throughput does not rise above 4), and X workers commit about 3.6 GB each (8 faulted
the pool once).

## 6. Gates

- **Gate C** (section 4), voiding.
- **Gate T** (trained), EXP-080's bar unchanged: mean overall margin of X16k above **0.13564**
  (half X's 12-seed mean overall margin) by `gate_l_verdict`. Fails: UNRESOLVED (X16k
  UNTRAINED). Its line prints far band and overall margin at 4000, 8000 and 16,000 side by side.
  The bar is attainable: X itself clears it at 4000 (0.2713), so a fail means training
  degraded the judge, which is itself the finding.
- Record checks: `n_updates` 16000, `resumed_from` at least 4000, encoder drift above 0,
  `n_probe` equal to EXP-075's record for the seed, `git_commit` the same on all 12 records.

## 7. Claims

One-sided exact sign-flip over all 4096 flips, paired by seed.

| claim | contrast | CONFIRMED when |
|---|---|---|
| 1 (primary) | J4V-X16k minus J4V-X (EXP-076), depth-11 success rate | p < 0.025 and mean >= 0.03; p < 0.025 with a smaller mean is CONFIRMED (SMALL) |
| 2 (mechanism) | X16k minus X (EXP-075), far band (probe distances 9 to 11) | p < 0.025 and mean >= 0.05 (EXP-080's bar); smaller mean is CONFIRMED (SMALL) |

The search cell is EXP-081's, unchanged except for the judge: J4V, mode C, k 4, goal check at
4, no-revisit, EXP-075's held-out sets and streams, 200 scrambles per seed, through
`e75.JudgeCritic`. A judge short of 16,000 updates is refused.

**Power.** J4V-X's depth-11 success rate is 0.1950 with per-seed sd 0.0262 (EXP-076). The paired
sd of EXP-081's depth-11 diffs against it was 0.0428, close to the binomial floor for two arms at
200 scrambles near 0.2 (about 0.040), so the se at n = 12 is about 0.0124. Claim 1's 0.03 bar
sits about 2.4 se up: a true gain of about 0.035 (0.195 to 0.23, roughly 7 more solves per seed)
confirms more often than not, and gains under about 0.02 do not resolve. A NOT CONFIRMED Claim 1
bounds the gain, it does not show zero. Claim 2 reuses EXP-081's
far-band se (about 0.0125 at n = 12): the 0.05 bar needs a gain of about 0.05, and gains under
about 0.03 do not resolve.

## 8. Verdict (in the aggregator, one line)

| condition | VERDICT |
|---|---|
| Gate C failed on any seed | VOID |
| Gate T failed | UNRESOLVED (X16k UNTRAINED) |
| Claim 1 CONFIRMED | BUDGET-LIMITED, PAYS |
| otherwise, Claim 2 CONFIRMED | ORDERING IMPROVES, SEARCH DOES NOT PAY |
| otherwise | VALUE ITERATION SATURATED AT 4X BUDGET |

CONFIRMED (SMALL) on either claim falls through to the next row. Claim 1 is read whatever Claim 2
says: the search is the outcome that matters, so it runs on every seed regardless of the probe.

## 9. Reported, never confirmations

- **Depth-9 search**, J4V-X16k against EXP-076's J4V-X at depth 9, with p and the per-seed
  count. This catches a near-goal trade-off like EXP-079's: a judge that orders far states
  better but loses depth-9 solves.
- Far band and margins at distances 7 and 8 at 4000 / 8000 / 16,000, per seed and on average.
- Overall margin and `spearman_7_11` from the 250-update probes (the curve's shape).
- Training loss at 4000, 8000 and 16,000; encoder drift.
- Search secondaries: optimality, mean steps, goal-fired fraction.
- X16k against EXP-081's O-Xt16k at depth 11 (value iteration against the oracle ceiling at the
  same budget), descriptive only.

## 10. No pilot

EXP-075 measured what a pilot would: X's cost per update at 4 and 8 workers, its committed memory
per worker, and Gate T's attainability (X at 0.2713). EXP-081 measured the search cell's cost.
The code adds a copy, Gate C and a second record, each covered by tests. The schedule is fixed
here: train 4 workers, search 12 workers.

## 11. Cost and schedule

- Gate C and copy: minutes.
- Training: 12,000 updates at about 4.2 s (4 workers), about 14 h per run; 12 runs in three
  waves, **about 42 h**. EXP-081 saw a second wave run 50% slower than its first, so 42 h may
  stretch toward 55 h.
- Search, depths 9 and 11: about 4 h at 12 workers.
- **Total about 46 h**, crossing at least two 03:31 restart windows. Every chain stage skips
  finished work, a restart costs at most 250 updates per run, and a VPS watcher polls the laptop
  and re-runs the chain if it died, until the verdict file exists.
- Build Monday 2026-10-26, run from Wednesday 2026-10-28; verdict about Friday evening.

## 12. What this does not answer

- Whether a different value-iteration objective (target, sync rate, walk length) would do better
  at the same budget. This changes the budget only.
- Whether the gain, if any, keeps growing past 16,000 updates. The 8000 point shows the slope.
- A SATURATED reading says the limit is the objective, given EXP-081's oracle judge shows the
  encoder can carry far more at this budget.
