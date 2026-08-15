# Session Handoff - 2026-08-10 (Mon) - Week 19 session 1

> [!important] **SUPERSEDED. Start from `docs/handoffs/SESSION-HANDOFF-2026-08-13.md`.**
> EXP-042 finished and is written up; EXP-043 is now the run in flight. Sections 4 and 10 of
> this file (operational lessons, the manim blocker) are still accurate.

> [!note] Week boundaries, corrected 2026-08-10
> **Week 18 is Mon 2026-08-03 to Sun 2026-08-09**, per the week-18 vault note's own frontmatter
> and the recurring Sunday 10:00 Weekly Review. EXP-038, EXP-039 and EXP-040 were all designed
> and dispatched inside week 18; only EXP-040's landing (02:09 Mon) falls in week 19.
>
> The 08-08 handoff was mislabelled "Week 19 session 1" and is now **week 18 session 4**. This
> file is the genuine start of week 19.

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

## 9. Week 19 session 1 (Mon night) - what happened, and what is running

**Week 18 was wrapped first.** Its vault note covered only EXP-036/037 and stopped on Aug 7;
EXP-038, EXP-039 and EXP-040 were all week-18 work and were missing. Now added, with a
close-out section, and the title changed to reflect the actual arc (the curriculum closes, the
trainer closes, the encoder opens). Handoff week labels corrected at `52d7c3f`.

### THE DIAGNOSIS LANDED - root cause found. See `experiments/041_seed_collapse_diagnosis/RESULTS.md`

> [!success] Curriculum stage 1 pays 33% for the worst possible policy
> EXP-040's two dead seeds die in **stage 1 (depth 1)** and never recover: entropy 0.005 and
> flat, where working seeds end stage 1 at 0.09-0.23 and **rebound to ~0.4** in stage 2.
>
> They solve **0.331 / 0.322** of depth-1 episodes - which is 1/3, which is exactly what a
> constant-action policy scores. Verified by enumeration: all six constant policies solve 2/6.
> A face move has **order 4**, so from a 1-move scramble any repeated move either inverts it
> (1 step) or cycles back to solved (3 steps), and the `2d+3 = 5` budget covers both.
>
> | depth | budget | constant-action | uniform-random |
> |---|---|---|---|
> | **1** | 5 | **0.3333** | 0.2208 |
> | 2 | 7 | 0.0370 | 0.0509 |
> | 3 | 9 | 0.0000 | 0.0104 |
>
> **Depth 1 is the ONLY stage where degeneracy beats exploring.** The trap predates the encoder -
> it did not fire with a frozen random encoder because learning was slow enough for entropy to
> survive stage 1. A pretrained encoder sharpens the gradient and reaches an attractor sooner,
> which is **one mechanism producing both of EXP-040's tails**: the tripled mean and the 2/12
> failures.
>
> **No fix applied.** Four candidates with trade-offs are in the RESULTS. The cleanest incentive
> flip - a depth-1 budget of 2, dropping constant-action to 0.167, below random - touches
> `max_steps_for`, which every cube experiment depends on. That is a comparability decision to
> take deliberately. **Whichever is chosen needs a pre-registered arm against the 2/12 rate.**

### The diagnosis run (completed), for reference

**Deliberately not on the laptop** - it is offline (last seen 17 h ago) and Michael wanted it to
stay off. Everything needed was already fetched here, so the run is self-contained.

```bash
# progress
grep -c "done:" /tmp/.../scratchpad/diag.log     # 4 seeds expected, ~2 h each, sequential
# the payload
cat /tmp/.../scratchpad/diag_out/diag_summary.json
```

Seeds **2, 4** (fail) and **0, 1** (work) at depth 4, EXP-040's exact configuration, 1 worker,
`nice 10`, ~248 MB. Started ~23:55, so **done around 08:00**.

> [!important] Phase 1 already eliminated the two obvious explanations
> - **The encoders are fine.** From the serialised weights, seeds 2 and 4 sit mid-pack on mean
>   rate, dead units and weight scale, and **seed 2 has the HIGHEST across-state discrimination
>   of all twelve** (0.207). Encoder quality does not even correlate with policy success: seed 8,
>   the best performer, has the *lowest* across-state sd.
> - **The initial policy is fine.** Before any training every seed sits at **96-99% of the log-6
>   entropy ceiling**, failing seeds included.
> - **It fails on the TRAIN side too** (train_success 0.000), so it is not generalisation.
>
> So the collapse **develops during training**, and `mean_train_entropy` - one number per run -
> structurally cannot say when. Hence the new `stage_trace` telemetry (`8a640b1`), which is
> additive and passed the encoder-seam neutrality check unchanged.

**What to read when it lands:** the reproduction check first. Each seed should reproduce its
EXP-040 value **exactly** - that equality proves the telemetry perturbed nothing and that the
failure is deterministic rather than a fluke. **If seed 2 comes back non-zero, that is a
different and more interesting problem** and the per-stage trace should be ignored until it is
explained.

Then the per-stage entropy trace: **at which curriculum stage does entropy die, and does the
failing seed solve its early stages *more* successfully than the working ones?** The live
hypothesis is premature convergence - a pretrained encoder makes depths 1-2 trivially learnable,
the policy locks onto a constant action, and nothing is left to explore with by depth 4.
**That hypothesis is NOT yet tested; it is what the trace is for.**

### Also built: the visual story (`viz/manim/`)

A three-act arc from EXP-029's 2.2% to EXP-040 moving the break point, for talks and video.
`viz/manim/story.md` is the plan; `data.py` reads the committed records so a scene cannot drift
from its experiment; four scenes drafted needing no LaTeX. **Never rendered** - written while
the VPS was busy, so layout is unverified.

Use **ManimCE** (`pip install manim`), not `3b1b/manim`, which is Grant's unsupported personal
codebase. Records absent locally for EXP-027/029/030; their published means cover every scene,
and they would only be needed for per-seed scatter.

**Content Day is SUN Aug 16**, and both drafted titles for Video 8 are now out of date in the
project's favour.

### Still queued for week 19

1. **Choose and pre-register a fix for the depth-1 trap** (root cause now known)
2. fine-tune the encoder during RL (frozen already works)
3. longer pretraining - the objective had not saturated at 40 epochs
4. push past depth 7 and find where the break point now sits
5. `^bbd0` audit the wrong-budget 0.354 modal anchor
6. `^7741` the EXP-030 memory re-ask - now 192 serialised heads
7. `^0817` **needs Michael**: Phase 0 / Phase 1 checkpoints in `progress-tracker`

## 10. Week 19 session 2 (Wed night) - EXP-042 IN FLIGHT, and the render pass failed

### EXP-042, running on the laptop

Dispatched 2026-08-12 18:55 at `9c8e286`. 36 runs, ~2.91M env steps, **~18 h, landing around
13:00 Thursday**. Healthy at 7.4-7.5 effective cores, zero tracebacks.

```bash
ssh -n laptop 'powershell -NoProfile -ExecutionPolicy Bypass -File C:\Users\mlgbr\probe_run.ps1 -OutDir "C:\Users\mlgbr\Desktop\Projects\neuromorphic-nn-snn-research-project\experiments\042_depth1_trap\outputs"'
.venv/bin/python experiments/042_depth1_trap/aggregate.py    # NOT WRITTEN YET - see below
```

Three arms on EXP-040's pretrained encoders, closing the EXP-041 trap:

| arm | change | depth-1 constant-action reward |
|---|---|---|
| baseline | none | 0.3333 |
| **capped** | `max_steps_by_depth=((1,2),)` | **0.1667** (below random's 0.2208) |
| skipped | `curriculum=(2,3,4)` | n/a |

> [!important] Read the spec's section 3 BEFORE reading any result
> The effect is **2 of 12 seeds**. A paired permutation test where two seeds carry the whole
> difference gives **p about 0.5 BY CONSTRUCTION**; Fisher's exact on 2/12 against 0/12 gives
> 0.48. **No arrangement of 12 seeds can prove the failures were eliminated.**
>
> So the PRIMARY claim is **entropy entering the final stage** - a per-seed mechanism measure
> that varies across all twelve - and the failure count is reported with **no p-value at all**.
> A Claim 1 refutation whose effect is confined to the two previously-failing seeds **must** be
> reported as *underpowered by construction*, not as "the fix does not work".

**`aggregate.py` was not written.** The block ran out on the manim attempt. It needs to apply
the four claims in `docs/superpowers/specs/2026-08-12-exp042-depth1-trap-design.md`, reading
`stage_trace[-1]["entropy_first_10pct"]` as the primary. **Write it before reading the records**,
so the rules are on disk before the numbers - the standing habit.

### The manim render pass FAILED, and the attempts are recorded

`viz/manim/README.md` now leads with the blocker. **Do not repeat these:** plain install,
`--no-binary` source build, and cython with `--no-build-isolation` all leave
`site-packages/manimpango/` holding Cython sources and **no compiled `.so`**.

Fixed along the way and worth keeping: **`libcairo2-dev`, `libpango1.0-dev`, `pkg-config` are
now installed**, so `pycairo` - the original blocker - builds.

> [!warning] The manim venv is isolated ON PURPOSE
> `/root/scratch/manim-venv`, not the project venv. Manim's dependency tree pulls **scipy**, and
> this repo deliberately has none - which is **why the exact permutation tests exist**.
> Installing into `.venv` would silently remove the constraint the methodology rests on.
> Verified after installing: `import scipy` in `.venv` still fails, as it should.

**Shortest path is probably the laptop**, which is the intended final-render box anyway and more
likely to have matching prebuilt wheels. This VPS blocker may not be worth solving.

**Content Day is SUN Aug 16 and the scenes remain unrendered and unverified.** That is the
standing risk.

### serverlocal is not a good render target

Probed: GPU genuinely idle (378 MiB of 8192, 0% util) - but **manim's default renderer is
CPU-only cairo**, so the GPU goes unused, and that box's **FX-8350 / DDR3** is far behind both
the laptop and this VPS per core. It also serves Jellyfin/Ollama to other people and runs the
QuantConnect live pipeline. Its GPU would only help for NVENC encoding, which is not the
bottleneck.

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
