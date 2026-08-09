# Session Handoff - 2026-08-10 (Mon) - Week 19 session 2

> **EXP-040 is IN FLIGHT on the laptop.** Dispatched 2026-08-09 14:40 at commit `bb2b3e5`.
> Phase 1 (12 pretrained encoders) is **done**; phase 2 is 36 policy runs, ~3.24M env steps,
> **ETA around midday Monday**. The repo is clean at `bb2b3e5` and pushed.
>
> Read `CLAUDE.md` first, then this file. Strategy is **not** in this repo: it is the vault at
> `300 Efforts/Active/Coding/Neuromorphic Development/road-to-a-solved-cube.md`.

## 0. State check

```bash
git log --oneline -1                 # expect bb2b3e5
git status --short                   # expect clean
scp scripts/laptop/probe_run.ps1 laptop:C:/Users/mlgbr/probe_run.ps1
ssh -n laptop 'powershell -NoProfile -ExecutionPolicy Bypass -File C:\Users\mlgbr\probe_run.ps1 -OutDir "C:\Users\mlgbr\Desktop\Projects\neuromorphic-nn-snn-research-project\experiments\040_pretrained_encoder_policy\outputs"'
```

`probe_run.ps1` prints `VERDICT=HEALTHY | ORPHANED | STALLED | NO_RUN` and exits non-zero on the
bad ones. It checks the **process tree** (is the root python's powershell parent alive, or did
the script die leaving workers computing into the void?) and **effective cores sampled inside a
single command**. Expect ~7 effective cores at 10 workers.

> [!warning] Three things that look like problems and are not
> - **Client-side `exit 124` or `255` says nothing about the job.** Both EXP-038 and EXP-040
>   outlived their dispatching ssh being killed by its own timeout. Windows has no SIGHUP
>   semantics; `ssh -n` plus no `Start-Process` is what makes that hold. **Probe before reacting.**
> - **A record count frozen for hours is normal.** Runs of one depth start together and finish
>   together. Effective cores is the live signal; the count is a step function.
> - **`log_growth_bytes=0` is normal here.** Progress lines land ~30 min apart.

## 1. What week 19 established - two results that fit together

| exp | question | answer |
|---|---|---|
| **038** | Do the trainer stabilizers help at depth 6, where collapse IS the diagnosed failure? | **REFUTED, and closed.** They de-collapsed it exactly as intended (modal 0.975 -> 0.631) and bought nothing (0.0021 vs a 0.0008 floor, p 0.45). The depth-5 arm came back **significantly negative**: 0.0396 -> 0.0046, W-L-T 1-10-1, **p 0.0020**. |
| **039** | Does inverse-model pretraining raise the EXP-033 probe ceiling? | **CONFIRMED, both bars.** Depth-4 probe 0.447 -> **0.786**, W-L 12-0, p 0.0005. It **beats the raw-facelet linear ceiling** (0.742), which width provably cannot do. |

**The two together are the story.** EXP-038 closes the trainer; EXP-039 opens the encoder.

> [!important] Refuted and CLOSED. Do not revisit without a new reason.
> - **Width** (EXP-033) - the representation is not fixable by making it wider
> - **Volume alone** (EXP-034)
> - **Curriculum stage weighting** (EXP-037)
> - **Starvation at depth 6** (EXP-037)
> - **Trainer stabilizers, entropy + advantage normalization** (EXP-038)
>
> **Nothing cheap remains inside the current architecture.** Every knob that could be turned
> without changing what the network *is* has been turned and measured.

### EXP-039's most useful number is not the headline

The trained-vs-**facelets** margin grows with depth, and this was **not** pre-registered so it is
an observation rather than a claim:

| depth | trained | facelets | margin | W-L | p |
|---|---|---|---|---|---|
| 3 | 0.908 | 0.906 | +0.003 | 5-7 | 1.0000 |
| 4 | 0.786 | 0.742 | +0.044 | 9-3 | 0.0107 |
| 5 | 0.660 | 0.618 | +0.042 | **12-0** | 0.0005 |
| 6 | 0.575 | 0.488 | **+0.087** | **12-0** | 0.0005 |

At depth 3 the encoder merely **matches** the observation - raw facelets are already 0.906
decodable, so there is nothing to add. **The encoder helps most exactly where linear
separability collapses**, which is Wall 1. Note the pre-registered Claim 3 (trained vs *frozen*)
reads **NOT MONOTONE**; both are in `RESULTS.md`, the rule's answer first.

## 2. EXP-040 - what it asks and how to read it

**Does a raised representational ceiling become a better policy?** EXP-033 Finding 2 is the
reason this is open: at depth 3 an oracle probe supported **0.48** while REINFORCE extracted
**0.22**. Less than half.

**Exactly one variable changes vs EXP-036:** which weights the **frozen** encoder holds. The head
is still `Linear(64 -> 6)`, **390 trainable parameters**. Fine-tuning end-to-end would confound
"a better representation" with "the encoder kept learning", so it is deliberately deferred.

```bash
.venv/bin/python experiments/040_pretrained_encoder_policy/aggregate.py
```

| claim | test |
|---|---|
| **1 (PRIMARY)** | depth 4: **>= +0.05** vs EXP-036's 0.1591 at p <= 0.05. The **powered** arm - the only depth where the policy works, so the only one where improvement is measurable |
| **2** | does the break point move? depth 5 reaching **0.10** would be the most consequential cube result to date |
| **3** | mechanism: does a gain arrive **with modal fraction falling**? Read modal WITH entropy |
| **4** | **the null is pre-committed and informative** |

> [!important] If Claim 1 refutes, that is a RESULT, not a disappointment
> EXP-039 raised the depth-4 probe past the facelet ceiling at p 0.0005. If that stands and the
> policy does not move, the finding is that **the representation was never the binding
> constraint - the readout or the learning signal is.** That is EXP-033 Finding 2 writ large and
> the strongest case yet for **Stage 3**: a value function on the currently idle `neuromod`
> pathway, which is also the only brain region not on the cube critical path.
>
> Report it as a positive redirection. **Do not describe it as "inconclusive, needs a bigger
> encoder".**

## 3. When it lands

1. **Fetch first.** `outputs/*.json` is gitignored and the laptop is the only copy. The
   `*_head.pt` and `*_encoder_*.pt` files ARE tracked - commit them from the VPS; the attic
   mechanism handles the resulting collision on the next dispatch automatically.
2. `aggregate.py`, then write `experiments/040_pretrained_encoder_policy/RESULTS.md` with each
   claim marked confirmed or refuted, plus provenance (seeds, dates, machine, wall clock,
   regeneration command). Todo `^ea23`.
3. **Read a real trace, not only the verdicts.** A cube frame labelled `solved: yes` on a
   scrambled cube once passed every unit test in the suite.

**Then the fork**, which EXP-040's Claim 1 decides:
- **Claim 1 confirmed** -> the encoder is the lever. Next is fine-tuning it during RL, and
  longer pretraining (the objective had **not** saturated at 40 epochs).
- **Claim 1 refuted** -> Stage 3, the value function on `neuromod`.

Either way `^7741` (the EXP-030 memory re-ask) is still cheap and still waiting: 96 heads from
EXP-036/037, 48 from EXP-038, 36 more from EXP-040.

## 4. Operational facts this session paid for

### The VPS cannot run parallel Python. One worker, maximum.

Two EXP-039 workers drove available memory to **311 MB**, started swapping, and pushed load to
**3.77** on a 2-core box. `CLAUDE.md` says not to do this and I did it anyway.

> [!danger] Swap thrash leaves NO evidence
> `journalctl` showed **zero OOM kills in 30 days**, which looks like "this box has never had a
> memory problem". It has not: a thrashing 2-core box becomes **unresponsive without the kernel
> ever firing the OOM killer**. **Absence of OOM records is not absence of memory trouble.**

At 1 worker: 380 MB, ~0.9 core, 722 MB free, load 1.31, and three monitored hours with zero
pressure events. Use `nice -n 10`.

**The structural constraint is not experiments.** ~2 GB was held by five long-running `claude`
processes. Until those are reclaimed this box effectively has ~2 GB, not 4.

### Do not extrapolate throughput across concurrency levels

**Three times this session.** EXP-038's mid-run re-forecast (28.9 h) was worse than the up-front
model (21 h actual 22.8 h). EXP-040's phase 1 was estimated at 15 min and took **1.6 h**.

The mechanism is measured: phase 1's last two seeds ran in **2,100 s** where the batch of ten
took **3,700 s** on identical work - **1.8x faster with 2 workers instead of 10**. The laptop is
**memory-bound, not core-bound**, so per-worker throughput falls as workers are added. Estimate
from throughput measured **at the concurrency you will actually run**, and do not re-forecast
from a partial sample whose completed work is not representative of what remains.

### An agent has no reliable clock between tool calls

Computing utilisation from inferred wall time produced "the run is at 15% and probably broken"
when a controlled 60-second sample said **7.34 effective cores**, i.e. 73%. Sample inside a
single command.

### Instrument traps hit this session

- **`LastWriteTime` is unusable on an open log.** Windows does not flush it to the directory
  entry while the writer holds the file, so `probe_run.ps1` reported `log_age_min=191` on a log
  whose last line was seconds old. Now measures **byte growth** instead.
- **A waiter condition must require well-formed output.** `grep -qv "^0,0$"` fired on a
  transient ssh hiccup and reported a phase transition that had not happened. Require
  `^[0-9]+,[0-9]+$` **and** a real threshold.
- **cmd.exe eats trailing backslashes and interprets `|` before PowerShell sees it.** Bit twice
  more this session (`$d + "run.log"` became `outputsrun.log`; a `Select-String` alternation
  became a shell pipe). **For anything non-trivial, `scp` the log down and read it locally.**

### `sync_repo.ps1` worked on its first real test

48 EXP-038 checkpoints collided with `origin/main` and were moved to
`C:\Users\mlgbr\repo-attic\20260809-144011` automatically, with `behind=0` verified afterwards.
The trap the 08-07 handoff predicted would recur did recur, and cost nothing.

### Capture a neutrality baseline BEFORE changing the trainer

`CubeConfig.encoder_state_path` touches the file every cube number since EXP-029 depends on. A
baseline was captured from the pre-change commit and the default path reproduces it to **1e-6**
on success, modal fraction and entropy across two seeds
(`tests/training/test_encoder_seam.py`, marked `slow`). **Once the change is in, that baseline
cannot be captured** - and a non-neutral seam would silently invalidate every prior result with
no signal that anything had changed.

## 5. Open debt

Tracked (`secretary todo list --project neuromorphic`):

1. `^ea23` **EXP-040 in flight** - read the verdicts and write its `RESULTS.md`.
2. `^bbd0` Audit other experiments for the wrong-budget **0.354** modal anchor. It is the 9-step
   figure; depth 6 runs 15 steps and its measured anchor is **0.309**, depth 5's is 0.321.
3. `^7741` The EXP-030 memory re-ask.
4. `^f772` cubeNet.ts B/D face orientation - pinned with `it.fails`, needs a browser. **Remove
   the `.fails` when fixed or the test starts erroring.**
5. `^0576` Nothing in the cube dashboard has been seen rendering in a browser.
6. `^0817` Phase 0 / Phase 1 checkpoints unticked in the vault `progress-tracker`. **Needs
   Michael.**
7. `^c16a` `requirements.txt` omits the `[server]` extra.
8. `^3ef2` Playbook: exit-255 guidance should mention the tailscale peer last-seen.

Not tracked:

9. **EXP-039 did not serialise its encoders** - 3.1 h of VPS compute produced no reusable
   artifact. Fixed for EXP-040 (`save_encoder`/`load_encoder`). Note EXP-039's encoders would
   not have been reusable anyway: they excluded the *probe's* split, not the RL split.
10. `verify_instrument_neutrality.py` reports a `config` mismatch as one opaque key.
11. EXP-025 still has no committed `RESULTS.md`; cite ADR 0001 Amendment 2.
12. Live trace streaming during training is a spec, not built.
13. The Aug 8 calendar milestone duplicates the recurring coding block in the same slot, and its
    description still says "next up: curriculum tuning", which EXP-037 refuted.

## 6. Pointers

- Results: `experiments/03{8,9}_*/RESULTS.md`
- Pre-registrations: `docs/superpowers/specs/2026-08-0{7,8,9}-*.md`
  (**EXP-039 spec section 5a** is the calibration and its two corrections)
- Analysis, re-runnable any time: `experiments/04?_*/aggregate.py`
  (**EXP-036's records are the comparator for EXP-038 and EXP-040** and are gitignored -
  re-fetch from the laptop if `outputs/` is empty)
- Stage 2 machinery: `src/neuromorphic/training/encoder_pretrain.py`
- Remote runs: `docs/playbooks/remote-experiment-runs.md` (**sections 1 and 4 rewritten**)
- Laptop scripts: `scripts/laptop/sync_repo.ps1`, `scripts/laptop/probe_run.ps1`
- **Strategy: vault `Neuromorphic Development/road-to-a-solved-cube.md`**
- Previous handoff: `SESSION-HANDOFF-2026-08-08.md`
