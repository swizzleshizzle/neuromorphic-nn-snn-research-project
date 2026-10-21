# EXP-080: can a judge on X's encoder order far states at all? (a ceiling check)

**Written 2026-10-20, week 29.** Michael chose this in session on 2026-10-20 ("ceiling check
first", answering neuromorphic#25), and chose to include one search arm (O-Xt only). Roadmap:
`docs/superpowers/specs/2026-10-05-lookahead-roadmap-design.md`.

## 1. The question

EXP-079 retrained the X recipe on walks of 1 to 40 moves (S). Both claims were REFUTED, and its
Gate M, the quantity that discriminates the arms, read **+0.0117, p 0.231**. S's values far
from solved moved nearer the truth, but S ordered states at distances 9 to 11 no better than X.
The search uses only that ordering.

That leaves two accounts, and they point Wednesday's next run in different directions:

- **The training signal is the limit.** A one-step Bellman target bootstrapped from the judge's
  own values cannot teach far ordering, but X's spiking encoder could carry it. Next: a judge
  trained on an ordering objective.
- **The representation is the limit.** X's encoder cannot carry far ordering, whatever it is
  trained on. Next: the encoder or region.

**This experiment trains the same judges on the TRUE distance**, as a measuring instrument and
an upper bound, and asks whether they then order distances 9 to 11 better than X. No arm here is
a candidate system. CLAUDE.md's rule that distance-to-solved is never a model input holds for
the method; these arms exist to price it.

## 2. What the measurement says

X's own far-band ordering is the baseline (computed with EXP-079's `margin_band`, so every arm
and X share one definition): the 3-move lowest-J leaf hit rate minus chance
(EXP-074's `leaf_rank_margin`) at probe distances 9 to 11, from EXP-075's committed training
records. Over 12 seeds it is **mean 0.1195, sd 0.0402**, per seed 0.048 to 0.179.

A throwaway calibration on the VPS (2026-10-20, seed 0, before this spec; script not tracked)
fitted a raw-facelet MLP to true distance on 1 to 40-move walks, excluding seed 0's held-out
and probe states:

| oracle | far-band margin (9 to 11) | overall margin (7 to 11) | X on seed 0 |
|---|---|---|---|
| 144-128-1, 300k states, 8 epochs (EXP-074's net) | 0.2345 | 0.3944 | far 0.175, overall 0.302 |
| 144-512-512-1, 600k states, 12 epochs | **0.3545** | 0.5464 | |

So the probe resolves a large gain at 9 to 11 (+0.18 over X on seed 0), and the ceiling depends
on the oracle's capacity. The bigger net is the one used here. Per distance it read 0.569, 0.301
and 0.193 at 9, 10 and 11: even this ordering thins at distance 11, where chance is highest. It
held 1.9 GB resident, mostly the unbounded BFS table (`ExactBFSDistance(max_depth=None)`, 97 s to
build on the VPS).

## 3. Arms

Each arm is trained per seed on seeds 0 to 11 (pilot 12 and 13) with MSE against true BFS
distance on random walks of length 1 to 40 (`random_walk_states(..., max_len=40)`), excluding
the seed's held-out states (EXP-073's `exclusion_set`), **its probe states** (`probe_set`, 550
states) **and every 3-move leaf of the probe states at distances 7 to 11** (`tree_levels(s, 3)[3]`,
at most 250 x 216): `leaf_rank_margin` ranks those leaves, and with true labels an arm that had
trained on a leaf would have seen the answer. X and S never had labels, so a leak could only
inflate the oracle arms, which is the direction of a false "learnable".

**Labels are minted once per seed, before training.** A mint phase draws each arm's training
states with `random_walk_states` from a recorded RNG seed, looks up their true distances in the
unbounded BFS table, and writes states and labels to an untracked file under the experiment's
output folder. Training workers read that file and never load the BFS table (1.9 GB resident),
so the worker count is set by the encoder alone. Sampling is the same function S used; only the
draws are made ahead of time. One draw per seed serves all three arms: 4000 blocks of 1000 states,
drawn as `random_walk_states(1000, 40, random.Random(seed), ...)` in update order. O-Xf and O-Xt
train on block u at update u, so the two are paired on identical states, and O-raw trains on the
first 600,000.

| arm | judge | trained | budget |
|---|---|---|---|
| **O-raw** | non-spiking MLP 144-512-512-1 on one-hot facelets | all | 600,000 states, 12 epochs, batch 512, Adam 1e-3 |
| **O-Xf** | **X's committed judge** (EXP-075 `judge_X_s<seed>/judge.pt`), sensory region **frozen**, a fresh wide head | head only, Adam 1e-3 | 4000 updates, batch 1000 |
| **O-Xt** | **S's recipe** (EXP-079 `make_judge`: EXP-075's pretrained `enc_X_s<seed>.pt`, a fresh wide head), sensory trainable | head 1e-3, sensory 1e-4 (EXP-074's arm W optimizer) | 4000 updates, batch 1000 |

- **O-Xt differs from S in two things**: the target (the true distance in place of the one-step
  Bellman backup) and the exclusion set (probe states and their leaves). Same encoder start,
  optimizer, budget and walk distribution.
- **O-Xf asks whether X's trained features already hold far ordering**, read out by a fresh head.
- **O-raw is the protocol check**: no spiking encoder at all, so if it cannot beat X, nothing here
  can be read (Gate V).

**Search arm (Michael's choice): J4V-O-Xt at depth 11**, EXP-076's J4V (mode C, k 4, k_goal 4,
no-revisit) with O-Xt as the judge, 200 held-out states per seed, against EXP-076's committed
J4V-X records (0.1950). It asks how much a far-ordering judge on this encoder would buy at the
typical scramble. **It is an upper bound in a second sense too:** the 4-move trees of 200 held-out
states reach most of the cube, so the search scores states O-Xt was trained on, with their true
labels. That cannot be excluded, and RESULTS.md must disclose it.

## 4. Code change

New experiment folder `experiments/080_oracle_ordering/`: a training driver for the three arms,
a margin cell (`leaf_rank_margin` on the seed's probe for a trained judge), the J4V-O-Xt
evaluation cell (EXP-079's `evaluate.run_cell` path with a judge passed in), an aggregator, and a
laptop launcher and chain in EXP-079's form. Nothing under `src/` changes. Distance labels are
looked up once per seed by the mint phase (section 3), which also writes the seed's probe set and
the true distance of every probe leaf, so neither training nor any margin cell loads the BFS
table. The judge's forward never sees a label. Gate 0 re-measures X through those minted
distances, so it also checks the mint.

## 5. Gates

- **Gate 0 (continuity):** X's far-band margin on seed 0, re-measured through this experiment's
  margin cell on the laptop, must equal EXP-075's committed record exactly (EXP-079's Gate M
  continuity cell did the same and passed). Failure voids everything.
- **Gate V (the protocol resolves), voiding:** O-raw's far-band margin exceeds X's per seed by an
  exact one-sided sign-flip, p < 0.05, **and** its mean difference is at least 0.05. Otherwise the
  verdict is UNRESOLVED: true labels on the strongest net here do not show better far ordering,
  so a null in the X arms would mean nothing. Attainability: the calibration measured +0.18 on
  seed 0, and the margin is bounded by 1 minus chance.
- **Gate T (the oracles trained), voiding for its arm:** each arm's overall Gate L margin (7 to
  11), averaged over the 12 seeds, is at least **0.13564**, half X's 12-seed mean (0.27127, from
  EXP-075's committed records): the Gate L form EXP-074 to 079 used. An arm that fails is reported
  as untrained, not as a ceiling.
- **Gate 1 (resolution) of the search arm:** EXP-070's `gate1_verdict` (0.02 floor, 0.98
  ceiling); J4V-X's 0.1950 is well inside it.

## 6. Pilot (seeds 12 and 13, into `outputs_pilot/`)

- Train all three arms on seeds 12 and 13 (2 workers), measure their far-band and overall
  margins against EXP-075's X pilot records on the same seeds, seconds per update, and peak
  committed memory per worker (the `python3.13` child, sampled correctly: EXP-079's per-cell
  sampler read the redirector outside training, so this launcher samples the child in every
  phase and the pilot checks the reading is above 500 MB).
- Time one J4V-O-Xt depth-11 cell on 20 held-out states. Seed 12 has no depth-9 policy cell
  (EXP-062 published seeds 0 to 11 only), so the timing cell runs seed 12's pilot O-Xt judge on
  seed 0's cell, first 20 held-out states. It measures time only; its record is marked as a
  smoke record and may not enter a verdict.
- **Stop rule (the protocol, before any seed 0-11 run):** if O-raw's mean pilot far-band margin
  is below **0.14532** (X's mean pilot far-band margin on seeds 12 and 13, 0.09532 from EXP-075's
  pilot records, plus 0.05), stop and report. The calibration's
  +0.18 would then not have reproduced, and Gate V would be likely to fail.
- **A dated amendment then fixes**, and goes to Michael with `pm ask`: the worker counts (total
  committed memory under 24 GB; the BFS table, about 2 GB, is held by mint workers only) and the
schedule.

## 7. Pre-registered claims (one-sided exact sign-flip, 4096 flips)

Paired per seed against X's committed far-band margin.

| claim | quantity | alpha |
|---|---|---|
| **1** | O-Xt far-band margin minus X's | 0.025 |
| **2** | O-Xf far-band margin minus X's | 0.025 |

Each is CONFIRMED when p < alpha **and** the mean difference is at least **0.05**; CONFIRMED
(SMALL) when p < alpha and the mean is below 0.05; REFUTED otherwise. 0.05 is about four times
EXP-079's Gate M reading and about a quarter of the calibration's oracle gain. VOID if Gate 0 or
Gate V fails, or Gate T fails for that claim's arm.

**Search reading (pre-registered, alpha 0.05, one-sided):** J4V-O-Xt minus J4V-X at depth 11.
PAYS when p < 0.05 and the mean difference is at least **0.03**; otherwise DOES NOT PAY. The 0.03
is borrowed from EXP-078's detectable difference at depth 11, a different pairing; it is a
pre-registered choice, not a power statement for this pairing. It is read only if Claim 1 is
CONFIRMED (not SMALL).

## 8. The verdict (encoded in the aggregator, one line)

| condition | verdict | what it points the next run at |
|---|---|---|
| Gate 0 or Gate V fails, or O-raw fails Gate T | **UNRESOLVED** | nothing; report and ask |
| O-Xt fails Gate T (Claim 1 VOID) | **UNRESOLVED (O-Xt UNTRAINED)** | nothing; report and ask |
| Claim 1 CONFIRMED and the search PAYS | **ORDERING LEARNABLE, PAYS** | a judge trained to order states, without true labels |
| Claim 1 CONFIRMED and the search DOES NOT PAY | **ORDERING LEARNABLE, DOES NOT PAY AT DEPTH 11** | not the judge's ordering alone |
| Claim 1 CONFIRMED and the search fails Gate 1 | **ORDERING LEARNABLE, SEARCH UNRESOLVED** | nothing; report and ask |
| Claim 1 CONFIRMED (SMALL) | **ORDERING LEARNABLE BUT SMALL** | nothing with confidence; the search is not read |
| Claim 1 REFUTED | **ENCODER-LIMITED** (Claim 2 reported beside it) | the encoder or region |

The search cells run only when Claim 1 is CONFIRMED (not SMALL), the one case where they are
read; the chain stops before them otherwise and saves about 4 h.

**Clarification, 2026-10-21, before any code or number:** section 3's shared draw, section 4's label lookup, section 6's
timing cell and memory line, and the three verdict rows for Gate T and Gate 1 were added while
writing the plan. They make the spec agree with section 3 and give every gate outcome a verdict
line of its own; no threshold changed.

The next stage is still Michael's call. The verdict states what the measurement supports, not
the choice.

## 9. Reported, not gating

**O-Xt minus S's far-band margin**, paired per seed (S's records are tracked from EXP-079): the
same recipe and walks, differing only in target and exclusion, so this is the cleaner reading of
"the training signal is the limit". Per-distance margins (9, 10, 11) for every arm; spearman_7_11 (`probe_judge`); mean J by
distance; training loss; for J4V-O-Xt, mean steps to solve and optimality beside J4V-X's and
J4V-S's (EXP-079).

## 10. Cost

- **Training:** O-raw is minutes per seed (about 5 on the VPS). O-Xt is one spiking forward and
  backward per state against S's seven forwards, so likely 1 to 3 s per update, about 1 to 3.5 h
  per seed. O-Xf has no encoder backward. The pilot measures all three.
- **Mint:** about 4M walk states per seed for O-Xf and O-Xt and 600k for O-raw, plus one BFS
  build per mint worker (97 s); minutes per seed, 2 workers.
- **Margins:** seconds per arm per seed.
- **Search:** 12 J4V-O-Xt depth-11 cells, about 2.1 h each at 6 workers (EXP-079), two waves,
  about 4.2 h.
- **Pilot about 2 to 4 h. The full run is about 9 to 17 h**: at about 4.2 GB per O-Xt worker,
  5 workers fit under 24 GB, so the 12 O-Xt trainings are three waves (3 to 10.5 h), then O-Xf,
  then the 4.2 h search. The pilot's measured seconds per update fix the number.
  The 03:31 restart costs at most one training checkpoint or one cell.

## 11. Order

1. Michael reviews this spec, then the plan; build and test on the VPS (subagent-driven).
2. Pilot (section 6), stop rule, dated amendment, `pm ask`.
3. On Michael's go: chain (Gate 0, train, margins, Gates V and T, search).
4. Aggregate, RESULTS.md, records and judges tracked.

## 12. Not in scope

- Any judge trained on an ordering objective without true labels (the next stage, if this one
  points there).
- Searches other than J4V, depths other than 11 for the search arm.
- Wider or different encoders.
