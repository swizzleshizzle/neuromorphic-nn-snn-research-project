# EXP-085: push training outward. X16k continued on longer walks, against a control on X's walks

Date: 2026-11-03. Michael chose #29 options 1 and 2 ("do both, start with the GPU port",
2026-10-31); this is option 2, the first of the two. Goal stated in chat 2026-11-03: "depth 11
moving, that's the real goal". Hardware chosen in chat the same day: "desktop now and check in, if
it crashes we will have to swap to laptop later".

## 1. The question

EXP-082 continued X's value-iteration judge from 4000 to 16,000 updates (X16k). The gain went near
the goal: margin at distance 8 rose 0.31 to 0.57, and depth 9 search rose 0.40 to 0.62 (confirmed
on never-trained-on states by EXP-083, 0.5537 vs 0.3279). Distances 9 to 11 barely moved (margins
0.18, 0.15, 0.11), and depth 11 stayed at 0.2083 against 0.1950 (not significant).

The training walks barely reach far out. X's walks are uniform in length 1 to 14, which puts
**11% of training states at distances 9 to 11 and 1.3% at distance 11** (measured 2026-11-03,
20,000 walks against the full BFS table). Random depth-11 scrambles sit mostly at distances 10 to 12.

EXP-079 tried walks up to 40 moves and lost at both depths. It trained a judge from the start of
training, when nothing near the goal was learned yet, so the far states' targets (one plus the best
child's value) were built from unlearned children. X16k now orders distance 8 well, so a distance-9
state's best children have good values.

**This experiment asks whether, starting from X16k, moving training mass outward moves depth 11.**

Distance-to-solved stays an instrument only. The lever is **walk length**, a property of how
training states are generated, never the true distance: selecting or weighting training states by
BFS distance would feed the answer key into training and is out of scope.

## 2. Design: two arms from one start

Both arms continue X16k from its update-16,000 checkpoint for **8000 more updates, to 24,000**, on
the desktop GPU, with every setting of EXP-082 unchanged (sync_every 100, probe_every 250, draws 1,
batch 1000, readout `wide`, hidden 512, Adam head 1e-3 and encoder 1e-4, EXP-075's exclusion and
probe sets) except the one lever:

| arm | walk max length over updates 16,000 to 24,000 |
|---|---|
| **C** (control) | 14 throughout, as X and X16k |
| **O** (outward) | rises linearly from 14 at update 16,000 to **30** at update 20,000, then holds at 30 |

Walk lengths stay uniform in 1 to max length, so short walks keep appearing in arm O and the
near band keeps being trained. Measured share of training states by distance:

| max length | distances 0 to 8 | 9 | 10 | 11 | 12+ |
|---|---|---|---|---|---|
| 14 (C, and O at the start) | 0.884 | 0.051 | 0.045 | 0.013 | 0.007 |
| 22 (O at about update 18,000) | 0.667 | 0.095 | 0.112 | 0.082 | 0.043 |
| 30 (O from update 20,000) | 0.524 | 0.108 | 0.149 | 0.141 | 0.079 |
| 40 (EXP-079, for reference) | 0.409 | 0.106 | 0.171 | 0.195 | 0.118 |

At 30, arm O sees about ten times X's share at distance 11 and keeps over half its training near
the goal, about 12 points more than EXP-079 kept. The ramp moves mass outward in the order value
iteration can use it. The schedule depends on the update count only.

Arm C separates "outward walks helped" from "more training helped". It is needed: EXP-082 showed
4x budget alone is worth about +0.01 at depth 11, and arm C measures what 8000 more updates on X's
walks give from the same start.

**Pairing.** Both arms start from the same copied checkpoint: same judge, target network, optimizer
state, walk RNG state and encoding stream state. They differ in walk lengths only, which makes the
seed pairing as tight as it can be.

**The probe is excluded from training in both arms (my call).** The training exclusion set covers
only the held-out search starts. The 50-per-distance probe that Gate T and Claim 2 score was open
to the walks, and at max length 30 arm O would draw about 1 M distance-11 training states over 8000
updates, so roughly half the distance-11 probe states would be trained on, against about 7% for
arm C. Claim 2 would then partly measure "trained on these exact states more often". So, from
update 16,000, both arms exclude the union of EXP-075's exclusion set and the seed's probe set.
This changes arm C from X's walks in that one respect only, and applies equally to both arms. The
primary is unaffected either way, because the search starts were already excluded.

## 3. Inputs

- X16k: EXP-082's `outputs/judge_X_s<seed>/` for seeds 0 to 11. `judge.pt` and `state.json` are
  tracked. `jt.pt` and `opt.pt` exist only on the laptop
  (`C:\Users\mlgbr\Desktop\Projects\neuromorphic-nn-snn-research-project\experiments\082_x_long_budget\outputs\judge_X_s<seed>\`,
  all 12 seeds, checked 2026-11-03, and the laptop is reachable). The copy is the chain's first
  stage, laptop to VPS to desktop, never edited in place; EXP-082's directories are never written.
  `jt.pt` matters little: `train_judge` re-syncs the target network on every update divisible by
  100, so 16,000 overwrites it on the first resumed update.
- Baselines for the reported block: EXP-082's `exp082_J4V-X16k_d9_s<seed>.json`,
  `exp082_J4V-X16k_d11_s<seed>.json` and `exp082_X16k_u16000_s<seed>.json`, all tracked.
- Code: `experiments/085_outward_walks/`, reusing EXP-075's `run` and `cells`, EXP-074's
  `gate_l`, EXP-082's continuation pattern and EXP-084's device path. The walk max length becomes a
  per-update schedule passed into training. The default stays the constant 14, and existing CPU
  runs must stay byte-identical (EXP-084's golden pins must keep passing).

## 4. Precondition: EXP-084 must read GPU PORT REPRODUCES

EXP-085 trains on the GPU, and EXP-084's Gate T is what shows GPU training learns. Gate T is
running on the desktop now (relaunched 2026-11-03 11:12 desktop time on Michael's "desktop now"),
as EXP-084's own chain, not a stage of this one. If EXP-084 ends in any other verdict, or Gate T
cannot finish because the desktop crashes three times, EXP-085 does not run on the desktop, and
the choice goes back to Michael by `pm ask`. This is a precondition, not a gate: no EXP-085 number
exists yet either way. The build does not wait for it; only the launch does.

## 5. Moving a CPU checkpoint onto the GPU

X16k was trained on the laptop CPU, and `train_judge` refuses to resume a checkpoint whose
`generator_device` differs from the requested device (EXP-084). A **rebase** step writes a new
`state.json` for the copy:

- the walk RNG state is carried over unchanged;
- the encoding generator restarts on the device, seeded deterministically from the seed and update
  16,000 (formula left to the builder, recorded in the Gate C record);
- `generator_device` is set to the device; update count, probes and every tensor file are
  unchanged.

The 16,000-update point is X16k exactly. Training past it is GPU training, which is not
byte-equal to what the laptop would have done, and does not need to be: both arms share it.

## 6. Gates

- **Gate C** (continuity, voiding). Per seed, before training:
  - (a) `judge.pt` and the pre-rebase `state.json` hash equal to EXP-082's tracked files, and
    `state.json` says update 16,000;
  - (b) every Adam state in `opt.pt` has `step` 16,000 and the right shape (the pairing rests on
    the shared optimizer state); `jt.pt` is copied but not gated;
  - (c) the judge, re-measured by `gate_l.leaf_rank_margin` on the CPU, equals the committed
    `gate_l` in `exp082_X16k_u16000_s<seed>.json` exactly.
  
  Gate C runs on the VPS, where all three files can be checked against git.
- **Gate T** (trained). Each arm's mean overall margin is above **0.13564**, EXP-080's bar
  unchanged. X16k starts at 0.3689, so a fail means continued training degraded the judge, which
  is itself the finding.
- **Gate H** (hardware, voiding). The desktop has logged memory faults ("Windows removed bad memory
  regions" 2026-10-31, bugcheck 0x1A 2026-11-01), so silent corruption is possible. There are two
  checks:
  - (a) **Search replay:** 2 search cells per depth, chosen by a fixed seed before the run, are run
    twice and must match exactly. EXP-084's Gate E showed GPU search is deterministic, 25 of 25
    cells.
  - (b) **Training replay**, only if a pre-flight shows GPU training is deterministic (two
    100-update runs from the same rebased start hash equal). If it is, each arm snapshots its
    last checkpoint before 24,000 (update 23,751, where the 250-update save lands), and 2 seeds per arm replay from it to 24,000 and must hash equal. If
    pre-flight shows GPU training is not deterministic, (b) is recorded as unavailable and the
    verdict line says so.
  
  A replay mismatch means one of the two runs was corrupted.
- **Record checks:** `n_updates` 24,000, the schedule recorded per record, device `cuda`, the
  same `git_commit` on every record.

## 7. Claims

One-sided exact sign-flip over all 4096 flips, paired by seed. The search cell is EXP-082's,
unchanged except for the judge: J4V, mode C, k 4, goal check at 4, no-revisit, EXP-075's held-out
sets and streams, 200 scrambles per seed, run on the GPU.

| claim | contrast | CONFIRMED when |
|---|---|---|
| **1** (primary) | J4V-O minus J4V-C, **depth-11** success | p < 0.025 and mean >= 0.03; p < 0.025 with a smaller mean is CONFIRMED (SMALL) |
| 2 (mechanism) | O minus C, far band (probe distances 9 to 11) | p < 0.025 and mean >= 0.05; smaller mean is CONFIRMED (SMALL) |
| guard | J4V-O minus J4V-C, depth-9 success | the guard **trips** when the mean is at or below -0.05 and a one-sided test for a drop gives p < 0.025 |

**Power.** EXP-082 measured the paired sd of depth-11 diffs at about 0.043 for two arms at 200
scrambles near 0.2, so the se at n = 12 is about 0.0124. The 0.03 bar sits about 2.4 se up. A true
gain of about 0.035 (roughly 7 more solves per seed) confirms more often than not, and gains under
about 0.02 do not resolve. Arms O and C share their start exactly, so the paired sd is likely
lower, never higher, than EXP-082's.

## 8. Verdict (in the aggregator, one line)

| condition | VERDICT |
|---|---|
| Gate C failed on any seed | VOID |
| Gate H failed | VOID (DESKTOP NOT REPRODUCIBLE) |
| Gate T failed on either arm | UNRESOLVED (ARM UNTRAINED) |
| Claim 1 CONFIRMED and the guard tripped | OUTWARD WALKS PAY AT DEPTH 11, COST DEPTH 9 |
| Claim 1 CONFIRMED | OUTWARD WALKS PAY AT DEPTH 11 |
| otherwise, Claim 2 CONFIRMED | ORDERING IMPROVES FAR OUT, SEARCH DOES NOT PAY |
| otherwise | OUTWARD WALKS DO NOT MOVE DEPTH 11 |

CONFIRMED (SMALL) falls through to the next row. The search runs on every seed whatever the
probe says.

## 9. Reported, never confirmations

- J4V-C and J4V-O against EXP-082's J4V-X16k at depths 9 and 11 (what 8000 more updates bought
  each arm; EXP-084's Gate E makes the GPU and CPU search numbers comparable).
- Margins at distances 7 to 11, per arm, at 16,000, 20,000 and 24,000.
- The share of each arm's training states by true distance, measured on a sample of its walks
  (instrument use only).
- Training loss, encoder drift, s/update on the GPU, and search wall time per cell.
- Search secondaries: optimality, mean steps, goal-fired fraction.

## 10. Pilot

No search pilot is needed: EXP-084 measured GPU search cost and exactness. One pre-flight is
needed, run before the full chain: rebase seed 0, train 100 updates twice per arm, and record
s/update and whether the two runs hash equal (it decides Gate H (b)). It also checks that arm O's
walks at update 20,000 have the measured distance shares within 0.02 of the table in section 2.

## 11. Cost, schedule and the crash plan

- Copy laptop to VPS to desktop, and Gate C: under 1 h.
- Training: 2 arms x 12 seeds x 8000 updates = 192,000 updates. EXP-084's Gate T seed 0 record
  measured **0.097 s/update** on the RX 9070 XT, one process at a time (2026-11-03), so about
  **5 to 6 h**; arm O's longer walks cost a little more per batch to generate. Every 250
  updates is a checkpoint.
- Search: 48 cells on the GPU (24 per depth). EXP-084 measured 4 to 6 min per depth-9 cell, and
  depth 11 is longer, so about **6 to 8 h**.
- **Total about 12 to 15 h**, so a verdict about a day after launch, inside this week's slot.
- **Crash plan:** a crash costs at most 250 updates, and the chain resumes every stage from its
  records and checkpoints. The watcher relaunches the chain on a crash. After a **third** crash,
  it stops and asks Michael. The laptop fallback is a restart of the whole experiment on the CPU,
  both arms and all seeds, because a seed may not mix devices and the arms must share one. On the
  laptop that is about 3.5 days.

## 12. What this does not answer

- Whether a different target (option 1, the next experiment) reaches further at the same budget.
- Whether the gain, if any, keeps growing past 24,000 updates or with a longer max length.
- A DO NOT MOVE reading with Claim 2 refuted says outward walks did not improve far ordering from
  X16k at this budget. It does not show the near band was not the bottleneck in EXP-079.
