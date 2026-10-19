# EXP-079: a judge trained where the typical scramble lives

**Written 2026-10-18, week 28.** Michael chose this direction in chat on 2026-10-18 ("option 1"):
retrain the judge rather than search deeper. Roadmap:
`docs/superpowers/specs/2026-10-05-lookahead-roadmap-design.md`.

## 1. The question

EXP-076 to EXP-078 deepened the search with EXP-075's 12 committed X judges and nothing retrained.

| search, X judges | depth 9 | depth 11 |
|---|---|---|
| full 4-move tree (EXP-076) | 0.4004 | **0.1950** |
| width-64 beam, 5 moves (EXP-077) | **0.5571** | 0.1475 |
| full 5-move tree (EXP-078) | 0.5558 | 0.1854 |

**Checked 2026-10-18 against EXP-078's RESULTS.md (before any S judge was trained).** Every
EXP-078 gate passed. Its Claim 1 (full 5-move tree over full 4-move tree at depth 11) is REFUTED
at -0.0096, about a third of what that run could detect, and its Claim 2 (the same at depth 9) is
CONFIRMED. Its Gate R pattern fits the question below: from depth 11 the judge's lowest-J 5-move
leaf is closer to solved no more often than a random leaf (0.531 against 0.540), where at 4 moves
it was 0.12 above chance. That instrument has little headroom at depth 11 (RESULTS.md, Gate R),
so it motivates this experiment and decides nothing. **No claim in section 8 changes**: the
baseline is still EXP-076's J4V-X (0.1950 and 0.4004), and the detectable difference comes from
EXP-075's paired sd, not from EXP-078.

A deeper search pays off near the goal and stops paying at depth 11, even when it scores every
leaf. The working account: the judge's values far from solved are not reliable enough for
another level to help. **This experiment asks whether the judge is weak far from solved because it
was trained there too rarely.**

## 2. What the measurement says

EXP-074 and EXP-075 train the judge by approximate value iteration
(`neuromorphic.training.value_iteration`). Each update draws 1000 states by
`random_walk_states(batch, max_len=14)`: a walk from solved of uniform length 1 to 14. The target
is a **one-step** backup, `1 + min` over the 6 children's values under a frozen copy of the
judge (`bellman_targets`).

The resulting training states, by true BFS distance (20,000 walks; measured 2026-10-18 on the VPS;
the BFS table is an instrument here and never reaches training):

| distance | 0-5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13+ |
|---|---|---|---|---|---|---|---|---|---|
| walks of length 1-14 (EXP-075) | 0.591 | 0.113 | 0.094 | 0.087 | 0.051 | 0.045 | **0.013** | 0.007 | 0.000 |
| walks of length 1-40 (this experiment) | 0.219 | 0.053 | 0.063 | 0.071 | 0.108 | 0.175 | **0.193** | 0.107 | 0.010 |
| all 3,674,160 states, uniform | 0.001 | 0.002 | 0.009 | 0.031 | 0.098 | 0.253 | **0.368** | 0.213 | 0.025 |

**The regime the judge runs in.** An evaluation cell's held-out states sit at exactly the scramble
depth (EXP-073's `heldout_states` draws from the BFS shell). So a depth-11 episode starts at
distance 11, and its first moves are chosen among children at distances 10 to 12. The committed X
judges saw distances 10 to 12 in **6.5%** of their training states, and distance 11 in 1.3%.
Walks of length 1 to 40 put **47.5%** there, and still keep 21.9% within 5 moves of solved. Value
iteration needs that near-goal share to bootstrap from the solved state outward.

## 3. Arms

| arm | judge | training walks | search | source |
|---|---|---|---|---|
| **J4V-S** | **S**: EXP-075's X recipe with `max_len` 40 | length 1 to 40 | full 4-move tree, no-revisit (EXP-076's J4V) | new: 12 trainings, 24 evaluation cells |
| **J4V-X** | X: EXP-075's committed judges | length 1 to 14 | the same | EXP-076's committed records, reused under Gate 0(b) |

**The only difference is `max_len`.** S starts each seed from that seed's committed EXP-075
pretrained encoder (`experiments/075_wide_region/outputs/encoders/enc_X_s<seed>.pt`). Everything
else is EXP-075 section 5 for X:
- the wide readout, hidden 512;
- 4000 updates, batch 1000, target sync every 100, probe every 250, 1 draw;
- optimizer `make_optimizer(judge, "W")`;
- seed handling and exclusion set unchanged.

The search is EXP-076's J4V unchanged: k 4, k_goal 4, no-revisit, depths 9 and 11, seeds 0 to 11,
paired with X by seed.

**Why `max_len` 40, fixed now.** It was chosen from the histogram above, never from an outcome.
The histogram shows it places most training mass at distances 9 to 12 while keeping a fifth of
it near the goal. No other length is tried (section 11).

**Why there is no deeper-target arm.** A k-step target scores 6^k leaves per training state
instead of 6. At k = 2 that is 36,000 judge evaluations per update instead of 6,000. EXP-075's X
update took about 12 s at 8 laptop workers, so even k = 2 is roughly 6x the forward cost and puts
one seed's training near 80 h. At k = 4, the 4-move tree's depth, it is 216x. Neither fits.
S tests the cheaper lever first. If S confirms, target depth was not needed to get there.

**Exclusion.** Longer walks land in a seed's held-out set more often. `random_walk_states` already
rejects and redraws them, so this changes the draw count and nothing else.

## 4. Code change

- `experiments/079_judge_trained_deep/`: `cells.py` (S's judge directory, `make_judge` from the
  committed X encoder, the arm table), `train.py` (EXP-075's driver with `--max-len`, recording
  `max_len` in every training record), `evaluate.py` (EXP-076's J4V cell with S's judge),
  `aggregate.py`, `launch079.ps1`, `chain079.ps1`.
- **The S launcher pins `max_len` 40 and refuses any other value**, so a full run cannot silently
  use 14.
- Each training record also carries **the realised training-distance histogram** of its last
  probe interval's walks, computed after the fact from the BFS instrument. It is a diagnostic,
  never an input.
- No library change is expected: `train_judge` already takes `max_len`. A unit test pins that the
  `max_len` passed reaches `random_walk_states` (mutation-test it by hardcoding 14).
- Chains launch with OS-level redirection, per the remote-run playbook section 3 (EXP-077's lost
  pipe).

## 5. Disclosure

- No S judge has been trained and no S cell evaluated.
- The section 2 histograms used seed-free random walks and the full BFS table, no evaluation state.
- EXP-078's 12 depth-11 J5V-X records were read before this spec (section 1), and EXP-077's
  results are published. Neither is an S outcome, but they are why the question is asked.
- The pilot (section 6) uses seeds 12 and 13 only, which are outside the evaluation seeds. **There
  is no 10-seed sensitivity line**: no evaluation seed has been seen by any S judge.

## 6. Pilot (seeds 12 and 13, into `outputs_pilot/`)

- Train S on seeds 12 and 13 at section 3's settings, the full 4000 updates (2 workers).
  EXP-075's pilot encoders for seeds 12 and 13 are reused if they are still on the laptop and match
  their pilot records. Otherwise they are re-pretrained by EXP-075's `pretrain.py` (pilot seeds
  only).
- Measure: Gate L margin overall and per probe distance 7 to 11, seconds per update, peak
  committed memory per worker (read from the `python3.13` child).
- **S must be a working judge (CLAUDE.md, EXP-064).** If S's mean pilot Gate L margin is below the
  X threshold EXP-075 used, **0.1298**, stop and report. The longer walks would then have cost the
  judge its competence, and Claim 1 would test a broken arm. There is no amendment in that case.
- Reported, not gating: S's per-distance pilot margins against EXP-075's X pilot judges on the same
  two seeds.

**A dated amendment then fixes**, before any seed 0-11 training, and goes to Michael with `pm ask`:
- the Gate L threshold (section 7's form);
- the training worker count (total committed memory under 24 GB);
- the schedule.

## 7. Gates

**Gate L (VI worked).** EXP-074's instrument, unchanged: on the seed's probe states at true
distances 7 to 11, the margin is the 3-move lowest-J leaf hit rate minus chance. It passes when,
over the 12 seeds:
- (a) the per-seed margin exceeds 0 by an exact one-sided sign-flip test, p < 0.05; and
- (b) the mean margin is at least half S's mean pilot margin.

The probe and the tree are the ones X was gated on. Only the trained judge differs, so the
instrument runs in the regime it was calibrated in. A failure voids both claims.

**Gate E (encoders trained).** Drift from the seed's committed pretrained encoder is > 0 on every
seed.

**Gate M (mechanism, reported, the quantity that discriminates the arms).** Per seed, S's Gate L
margin restricted to probe distances 9 to 11, minus X's on the same probe states, by exact
one-sided sign-flip. It does not void anything, but it binds the reading. **If Claim 1 confirms
and Gate M is not above 0 at p < 0.05, RESULTS.md leads with that.** The gain would then not be
attributable to better ranking far from the goal.

**Gate 0.**
- (a) Determinism: J4V-S at depth 9, seed 0, re-evaluated into `outputs_det/`. Identical except
  `wall_s` and `git_commit`.
- (b) Continuity: J4V-X at seed 0, depths 9 and 11, re-run through this experiment's evaluation
  code. It must equal EXP-076's records in every outcome field (78/200 and 36/200).

**Gate 1 (resolution).** UNRESOLVED if both arms are below 0.02 or above 0.98 (EXP-070's
`gate1_verdict`). X's 0.1950 and 0.4004 are well inside the band, so the contrast can come out
either way.

**Gate R at k 4 (reported).** As EXP-076: per depth, the fraction of evaluation held-out states
whose lowest-J 4-move leaf is closer to solved than the root, against chance, for S. It is
compared with EXP-076's X rates.

## 8. Pre-registered claims (one-sided exact sign-flip, 4096 flips; alpha 0.025 each, 0.05 / 2)

**Claim 1 (primary): a judge trained on walks of up to 40 moves solves more of the typical random
scramble.** mean(J4V-S - J4V-X) > 0 at depth 11.

**Claim 2 (primary): the same at depth 9.** mean(J4V-S - J4V-X) > 0 at depth 9.

Verdicts: CONFIRMED, CONFIRMED (SMALL) under 0.01, REFUTED (mean <= 0), NOT SIGNIFICANT,
UNRESOLVED (Gate 1), VOID (Gate L, Gate E or Gate 0). The aggregator returns each one as a verdict.
**Gate E added to the VOID set 2026-10-19, while writing the plan, before any S judge was trained.**
Section 7 listed Gate E without saying what its failure does. A seed whose encoder never moved is
not the S recipe, and EXP-075 voided on Gate E for the same reason, so this is a tightening.

**How each reading reads, fixed now:**
- **Claim 1 CONFIRMED:** the judge's training distribution was a limit at the typical scramble,
  and one-step targets suffice to move it.
- **Claim 1 NOT SIGNIFICANT or REFUTED:** the judge is not the limit **by this lever**. That is not
  evidence that search is the limit. A deeper target (section 3) or a different judge remains
  open.
- **Claim 1 CONFIRMED, Claim 2 REFUTED:** a trade. S gave up near-goal training share (0.591 to
  0.219 at distances 0 to 5) and depth 9 paid for it. That is reported as a trade, not a win.

**What the primaries can detect.** EXP-075's paired per-seed sd between two differently trained
judges under the same search (J3V-X minus J3V-W) was 0.0388 at depth 9 and 0.0410 at depth 11.
That makes the se about 0.012 at n = 12. At alpha 0.025, a true difference of about 0.033 is
detectable with 80% power. That is the right scale: retraining moves more than search does
(EXP-076's search-only sd was 0.031 to 0.032).

**Secondary (patterns, never confirmations):** Gate M and Gate R; probe Spearman per distance;
mean steps to solve and optimality; the realised training histogram; EXP-078's full 5-move tree
alongside at depth 11.

## 9. Cost

- **Pilot:** 2 S trainings. EXP-075's X update took about 12 s, so 4000 updates is about 13 h per run at
  2 workers. Pretraining is added only if the pilot encoders must be rebuilt.
- **Training:** 12 S trainings. EXP-075's X and Y training (24 runs at 8 workers) took about 19 h.
  12 runs should be about 13 to 19 h, depending on the amended worker count.
- **Evaluation:** 24 J4V-S cells, about 27 cell-hours (EXP-076's J4V-X: 1.35 h per depth-11 cell,
  0.86 h per depth-9 cell). That is about 6 to 8 h at 6 workers, with the slowdown EXP-077
  measured.
- **About 35 to 40 h of laptop time in all**, plus Gate 0 and rank. Training checkpoints every 250
  updates and evaluation banks per cell, so a 03:31 restart costs one probe interval or one cell.

## 10. Order

1. Spec review by Michael, then plan, then build and test on the VPS (subagent-driven).
2. EXP-078 closed out, including its RESULTS.md. Section 1's provisional row and section 8 are
   then checked against it. Any change to a claim is made and dated **before** step 3.
3. Pilot (section 6); stop rule; dated amendment; `pm ask`.
4. On Michael's go: `chain079.ps1` with phases cont (Gate 0(b)), train, gate L and E, eval
   (depth 11 first), det, rank.
5. Aggregate, RESULTS.md, records and S judges tracked.

## 11. Not in scope

- Deeper training targets: the ST arm, priced in section 3.
- Other walk lengths, or a sampled mixture.
- Warm-starting from the X judges.
- Searches other than J4V.
- The narrow (Y) or EXP-074 judges.
