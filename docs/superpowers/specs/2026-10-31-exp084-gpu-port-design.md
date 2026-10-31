# EXP-084: port judge training and the look-ahead search to the desktop GPU

Date: 2026-10-31. Chosen by Michael on neuromorphic#29 ("do both, start with the GPU port"), the
prerequisite for options 1 (a longer-reaching target) and 2 (push training mass outward). The
acceptance test is his: "re-evaluate an old judge on GPU and match its numbers", where "within a
couple solves is fine".

## 1. Why

EXP-082's 16,000-update training took about 65 h on the laptop, and EXP-083's search alone took
10.6 h. Options 1 and 2 both need new training runs at that scale or larger. The profile on
2026-10-30 (`compute-hardware-options` in the vault) found one value-iteration update is 78% the
spiking network's forward pass over 6,000 look-ahead children, under 2% Python. That is tensor
math. On the desktop's RX 9070 XT (ROCm PyTorch `2.9.1+rocm7.2.1`, verified working on Windows 10)
a prototype update took **0.108 s against 2.54 s** on one laptop worker.

The prototype was timing only. This experiment makes the port real and proves it computes what the
CPU code computes before anything is built on it.

## 2. Goals, non-goals, anti-goals

**Goals**

- Judge training (`value_iteration.train_judge` and what it calls) and the look-ahead search
  (`lookahead.evaluate_lookahead` through `JudgeCritic`) run on a chosen torch device.
- Re-evaluating committed judges on the GPU reproduces their published search results within 2
  solves per 200 (section 5, Gate E).
- A GPU-trained judge is a trained judge (section 5, Gate T), and its checkpoints load on any
  machine, CPU included.
- Measured speed of both paths on the desktop, so options 1 and 2 can be priced.

**Non-goals**

- Bit-identical retraining. Retraining is not reproducible even between two CPUs
  (`docs/reproducibility-audit.md`), so the training check is a sanity gate, not a match.
- Porting anything outside the judge and search path (policy training, older experiments).
- Choosing or starting option 1 or 2. Their specs come after this verdict.

**Anti-goals** (failures even if the GPU path works)

- **Any change to the CPU path's numbers.** The default device is CPU, and on CPU every existing
  function must produce byte-identical results. Every committed record must stay re-evaluable on
  the laptop and the VPS exactly as today.
- A judge that can only be loaded on the desktop.
- Evaluation whose random stream depends on the device. A judge trained on the GPU must score the
  same when searched on the laptop as on the desktop, within Gate E's tolerance.

## 3. How the port works

**Device parameter, CPU default.** A `device` argument (default `"cpu"`) is threaded through
`Judge`, `bellman_targets`, `train_step`, `probe_judge`, `train_judge`, `JudgeCritic` and the
search's judge calls. With the default, no tensor moves and no code path changes.

**Two encoding streams.**

- **Evaluation stream (exact).** Spikes are drawn by `encode_cube` on the CPU from the same CPU
  `torch.Generator`, in the same order as today, then moved to the device. The random stream is
  therefore identical to the CPU run's, and the only remaining difference is float rounding in
  the network's matrix math. Search and probes always use this stream, on every device.
- **Training stream (fast).** Training may draw spikes on the device from a device generator
  seeded from the run seed. Moving 6,000 x 32 x 144 spike tensors from the CPU on every update
  would cost more than the forward pass the port removes. A GPU-trained run is therefore not the
  same trajectory as a CPU run from the same seed. That costs nothing that exists today, because
  retraining was never reproducible across machines.

The builder may instead use the exact stream for training too, if measurement shows the transfer
is cheap (for example, by sending spikes as `uint8` or `bool`). Either way the choice and its
measured cost go in RESULTS.md.

**A rounding difference stays inside its own state.** In mode C, `choose_move` never reads the
cell's shared evaluation generator: it scores the 6^4 leaves through the judge using an
imagination generator seeded fresh per state and step (`imag_seed_for(imag_seed, i, t)`). The
shared generator only feeds the policy logits, which mode C uses only for a fallback that never
fired in EXP-083 (fraction 0.0000). So if GPU rounding flips one move, that state's own path
changes and no other state's does. Gate E's per-cell count can therefore move by about the number
of states whose path changes, not by a fresh draw of the whole cell, so a 2-solve bar is
reachable when flips are rare. Gate P measures how rare they are before Gate E runs.

**Full fp32.** TF32 and any reduced-precision matmul are disabled on the device
(`torch.backends.cuda.matmul.allow_tf32 = False`, `torch.backends.cudnn.allow_tf32 = False`,
`torch.set_float32_matmul_precision("highest")`). ROCm exposes the GPU as `"cuda"`.

**Portable checkpoints.** `_save_checkpoint` writes CPU tensors (judge, target, optimizer state),
so a checkpoint saved on the desktop loads on the laptop with no `map_location`. A device-stream
run also stores its device generator's state, so a resumed run continues the same trajectory on
the same machine. The walk `random.Random` stays in plain Python on the CPU, unchanged.

**What stays on the CPU.** Walk generation, `apply_move`, solved checks, and the policy agent's
one forward per real search move (cheap next to the judge's thousands of children per move). The
builder may move more if it measures a gain.

## 4. Where it runs

The desktop `desktop-ma4deik` over the `ssh desktop` alias (Windows 10, user `michael greene`):

- venv `D:\gpu083\venv` (uv, Python 3.12, `torch 2.9.1+rocm7.2.1`, plus snntorch, gymnasium,
  matplotlib, pandas, pyyaml, seaborn, pytest);
- repo clone `D:\gpu083\repo`, pulled to the merge commit before the run.

`scp` to paths containing a space fails, so use `D:/` paths. The desktop is Michael's daily
machine. One GPU process at a time, so runs are sequential.

The committed judges it needs (EXP-075's `judge_X_s*` at 4000, EXP-082's at 16,000) and EXP-083's
sets and records are tracked in git, so a pull brings them.

## 5. Gates and verdict

**Gate C (CPU unchanged, voiding).** On the VPS, at the merge commit, rerun EXP-083's J4V-X16k
depth-9 seed-0 cell on its first 20 states with the default device, plus one
`train_step` and one probe from a fixed seed. Each output must equal the same computation at
`9ffe730` (EXP-083's commit) exactly. The test suite pins this too, but this gate is run once on
real checkpoints and recorded.

**Gate P (decision agreement, run before Gate E).** For every state in the 24 EXP-083 Gate E
cells (12 seeds, both arms), compute the search's first move exactly as the cell does at t = 1
(mode C, k 4, goal check at 4, the same imagination seed), once on the CPU and once on the GPU.
Report the fraction of the 4,800 first moves that agree, the largest absolute judge-value
difference over all scored leaves, and how many of the disagreements were near-ties. This is the
mechanism under Gate E and its calibration.

**Stop rule:** if more than **0.2%** of first moves disagree (more than 9 of 4,800), stop before
Gate E and ask Michael. A state takes about 10 moves, so at that rate roughly 2% of states (4 per
cell) would change path, and the 2-solve bar would be likely to fail on noise. Changing the bar is
his call, and it is legitimate only before any Gate E number exists.

**Gate M (machine control).** On the desktop's **CPU**, with the default device, rerun the EXP-082
J4V-X16k depth-9 seed-0 cell (published 127). The laptop runs torch `2.13.0+cpu` and the desktop
`2.9.1+rocm7.2.1`, and re-evaluation has been shown portable across machines only for some
experiments (CLAUDE.md, EXP-070 pre-flight). If Gate E fails, Gate M says whether the GPU or the
machine and torch build is to blame. Pass: within 2 solves of 127, the same bar as Gate E.

**Gate E (re-evaluation, Michael's acceptance test).** On the desktop GPU, rerun these committed
cells exactly as published (same states, same seeds, same streams):

- EXP-083 J4V-X16k depth 9, seeds 0 to 11 (published 124, 115, 93, 99, 107, 111, 93, 114, 125,
  111, 118, 119 of 200);
- EXP-083 J4V-X depth 9, seeds 0 to 11 (77, 66, 61, 52, 62, 64, 62, 62, 72, 69, 73, 67);
- EXP-082 J4V-X16k depth 9 seed 0 on EXP-075's held-out set (127, EXP-083's Gate R cell).

**Pass:** every one of the 25 cells is within **2 solves** of its published count. The
aggregator also reports the number of exact matches and the summed absolute difference.

Depth 9 is chosen because it has the most solves to disagree about (0.33 to 0.63), so it is the
strictest test, and both judge budgets are covered. Depth 10 adds cost and no new code path.

**Gate T (GPU training learns).** Train X from scratch on the desktop GPU to 4000 updates for all
12 seeds, with EXP-075's arm, batch, walk length, exclusion set, sync and probe schedule
unchanged. **Pass:** the mean overall margin (distances 7 to 11) is above **0.13564**, EXP-080's
trained bar, which EXP-082 also used. EXP-075's CPU X at 4000 measured 0.2713.

If the builder chose the exact stream for training (section 3), that is a speed choice only. The
float math still diverges, so the paired numbers below are framed as "different machine" either
way.

Reported beside it, never gating: the paired GPU-minus-CPU overall and far-band margins, ahead
count and exact sign-flip p, the per-update wall time, and the loss curve. Retraining differs
across machines by design, so a difference here is information for options 1 and 2, not a
failure. A GPU mean far below the CPU's would be worth stopping on and is called out in RESULTS.md
if it happens.

**Speed (reported).** Per-update training time at batch 1000, and per-cell search time for the
Gate E cells, against the laptop's 2.54 s per update alone (about 6.2 s each at 4 workers) and
EXP-083's 2,602 to 4,977 s per depth-9 cell.

**Verdict (one line, in the aggregator):**

| condition | VERDICT |
|---|---|
| Gate C failed | VOID |
| Gate E failed, Gate M failed | DESKTOP DIFFERS FROM LAPTOP ON CPU TOO |
| Gate E failed, Gate M passed | GPU PORT DOES NOT REPRODUCE |
| Gate E passed, Gate T failed | GPU SEARCH REPRODUCES, TRAINING DOES NOT LEARN |
| Gate E and Gate T passed | GPU PORT REPRODUCES |

If Gate E fails, the first suspects in order are reduced-precision matmul still active, a
spike tensor drawn on the device in the evaluation path, and ties between children broken
differently. Diagnosing them is follow-up work, not part of this verdict.

## 6. Code

- `src/neuromorphic/training/value_iteration.py` and `lookahead.py`: the device parameter and the
  two streams; portable checkpoints.
- `experiments/075_wide_region/evaluate.py`: `JudgeCritic` and `load_judge` take a device (judges
  load to CPU, then move).
- `experiments/084_gpu_port/`: `gatec.py` (Gate C, run on the VPS), `decisions.py` (Gate P),
  `reeval.py` (Gates E and M), `train.py` (Gate T), `aggregate.py`, and a PowerShell chain for the
  desktop in the shape of EXP-083's, with each stage skipping finished records.
- Run order: Gate C on the VPS, then on the desktop Gate P, the stop rule (the chain halts and
  writes why if more than 0.2% of first moves disagree), Gate E, Gate M, Gate T, aggregate.
- Tests: CPU byte-identity against the pre-change code (the existing suite plus explicit
  pins of a `train_step`, a probe and a short search on fixed seeds); a checkpoint saved from a
  device run loads with no `map_location`; the aggregator's verdict rows, including the
  2-solve boundary (2 passes, 3 fails). GPU tests skip unless `torch.cuda.is_available()`, so the
  VPS and laptop suites stay CPU-only.

## 7. Cost

- Build: about a day.
- Gate E: 25 search cells. At the prototype's speed-up, minutes each, so under 2 h sequential.
- Gate T: 12 x 4000 updates at about 0.11 s is about 7 min per seed plus probes, about 1.5 h.
- Gate P: 4,800 first moves on CPU and GPU, under 1 h. Gate M: one cell on the desktop CPU,
  about 1 h. Gate C: minutes on the VPS.
- **Total desktop time about 5 to 6 h**, sequential, in one sitting or across two.

## 8. What this does not answer

- Whether options 1 or 2 work. Each gets its own spec, on GPU-trained judges, evaluated with the
  exact stream so their results re-evaluate on any machine.
- Whether GPU-trained judges are as good as CPU-trained ones at 16,000 updates. Gate T checks
  4000 and reports the difference.
- Multi-seed parallelism on one GPU (several seeds per process, or several processes). Worth
  measuring once a real run needs it.
