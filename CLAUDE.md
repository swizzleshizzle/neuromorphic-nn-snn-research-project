# CLAUDE.md

Standing knowledge for this repo. Session-specific state lives in `docs/handoffs/`; this file is only for things that stay true.

## Commands

Sessions moved from a Windows desktop to a Linux VPS on 2026-07-30. Use whichever applies:

```bash
.venv/bin/python -m pytest tests/ -q -m "not slow"           # Linux VPS: fast inner loop, about 13 min
.venv/bin/python -m pytest tests/ -q                         # Linux VPS: full suite
```

```powershell
.venv\Scripts\python.exe -m pytest tests/ -q -m "not slow"   # Windows (laptop, old desktop)
.venv\Scripts\python.exe -m pytest tests/ -q                 # Windows: full suite
```

Always run python via the venv, never a bare `python`. One test is marked `slow` (it builds the **unbounded** cube BFS table, all 3,674,160 states, about 67s). The `slow` marker is registered in `pyproject.toml` and is NOT deselected by default, deliberately: a default run stays honest.

**`ExactBFSDistance(max_depth=N)` is not the slow path.** A bounded build is near free (depth 6 is 11,913 states, about 0.04s). Only `max_depth=None` costs the 67s. Do not restructure code to avoid constructing a bounded provider; that optimisation buys nothing.

**MEASURED PER-FILE RUNTIMES (re-measured 2026-09-01, this VPS), because the
600 s Bash ceiling is the binding constraint on this suite, not agent
discipline. The 2026-08-28 table this replaces was wrong in both directions and
its chunking recipe no longer completes.**

The slow files under `-m "not slow"` are NOT the ones the old table named:

| file | `-m "not slow"` | note |
|---|---|---|
| `test_cube_baseline.py` | **355 s** | run alone |
| `test_ablation_run_smoke.py` | **258 s** | ONE test, `test_mint_and_cell_smoke` |
| `test_curriculum.py` | **246 s** | 3 tests over 47 s each |
| `test_seed_split.py` | **158 s** | |
| `test_ablation_hook.py` | **135 s** | |
| `test_encoder_finetune_seam.py` | **1.3 s** (2 of 8) | the old table said 51 s and "must be run alone"; both are wrong under `not slow` |

Everything else in `tests/training` is seconds. The old table's `test_critic_seam.py`
and `test_encoder_seam.py` are cheap once their slow-marked tests are deselected.

**Counts:** 370 tests 2026-07-30, 521 2026-08-28, 561 2026-09-01, 623 2026-09-13,
663 2026-09-17,
703 2026-09-20,
**742 as of 2026-09-25 (720 not slow, 22 slow)**, verified by `--collect-only`. The `dashboard/` JS app is separate: **88 vitest + 2 Playwright e2e**, run with `npx vitest run` and `npx playwright test` from `dashboard/`.

**A chunking that actually works.** The `tests/training` remainder is ~837 s and
**cannot** fit in one call at any timeout, so background it deliberately and let
the harness re-invoke on exit:

```bash
.venv/bin/python -m pytest tests/ -q -m "not slow" --ignore=tests/training      # 326, 42 s
.venv/bin/python -m pytest tests/training/test_cube_baseline.py -q -m "not slow" # 31, 355 s
# the rest of tests/training, ~837 s: RUN IN BACKGROUND, not in a foreground call
```

**To find the expensive files rather than bisecting**, loop with a per-file
timeout inside ONE call; whatever gets killed is the culprit and 12 files fit
comfortably under the ceiling:

```bash
for f in tests/training/test_*.py; do
  timeout 40 .venv/bin/python -m pytest "$f" -q -m "not slow" >/dev/null 2>&1 || echo "SLOW: $f"
done
```

The 15 slow-marked tests are a separate `-m slow` run and also belong in the
background.

Always pass an explicit tool timeout: the Bash default is 120 s, so even a
250 s file auto-backgrounds without one. See the global CLAUDE.md gotcha.

Ruff is configured in `pyproject.toml` but is not installed in the venv, so lint is not mechanically enforced.

## Commit conventions

- Plain messages. **No `Co-Authored-By` trailer. No "Generated with" line.**
- **No em-dashes** anywhere in code, docs, or commit messages. (Vault notes under `Documents/Second Brain` are exempt; they use the vault's own style.)
- Merge feature branches with `--no-ff` and a `Merge <branch>: <summary>` subject. The repo is otherwise `main`-only; delete branches after merging.
- **Never `git add -A` while a subagent is working.** Stage explicit paths. Doing otherwise once swept an implementer's in-progress work into a docs commit (`a097436`), which now contains Task 2's implementation under a documentation message.

## The test-strength rule

**Never write an assertion that cannot fail.** Four real defects in this repo hid behind one, and each survived multiple review passes:

| Assertion | What it hid |
|---|---|
| `assert 0.0 <= success_rate <= 1.0` | a hardcoded `random.Random(0)` that made the pre-registered chance floor one realisation replayed across 12 seeds (reported 33.3% vs a true 20.3%) |
| `assert count_nonzero(W_rec) > 0` | `Hippocampus.store()` assigning instead of accumulating, so it held exactly one pattern |
| `assert not torch.equal(a, b)` | a recall code that was 99.8% identical across all inputs |
| a regression test that re-implemented its own fix inline | the fix site could be deleted and the test would still pass |

Practical consequences:
- Prefer a **measured numeric threshold** to a qualitative check. Prototype the behavior first, then set the bar with margin: "assert cosine < 0.95" beats "assert it discriminates".
- A test must **fail against the pre-fix code**. If it passes either way, it is documentation, not a test.
- **Never weaken a passing threshold to make another fix land.** If a change would require that, stop and say so.
- If a plan hands an implementer a test body, that body carries the plan author's errors verbatim. Review test bodies as carefully as production code.

## The gate-calibration rule

Companion to the test-strength rule above. That one says never write an assertion that cannot
**fail**. This one says never write a validity gate that cannot **pass**, and never set its
threshold in a regime other than the one it will run in.

A validity gate is the pre-registered condition that decides whether an experiment's claims may be
read at all. **Two of the three experiments that have used one got it wrong**, both in 2026-09:

| gate | required | what happened |
|---|---|---|
| EXP-057 Claim 4 | `critic_within_rms < 1e-6` | Calibrated on a **depth-3** smoke run measuring 6.8e-10, evaluated at **depth 7**, where 15.6 steps per episode against a return RMS of 4.42 push float reassociation to 5.0e-07. **Passed with 2.0x margin**, not the three orders the spec claimed. A near-miss false VOID. |
| EXP-058 Claim 3 | `mean_n_stored > 10` | `mean_n_stored` is bounded above by episode length, and depth-6 episodes average **7.76 steps** because the policy solves them before the 15-step cap. **Unsatisfiable by construction.** It voided the experiment. |

**Both are the same shape: a threshold chosen in one regime and applied in another.**

Practical consequences:

- **Compute the gate's maximum attainable value before committing it.** `mean_n_stored` cannot
  exceed `mean_steps`; one line of arithmetic would have caught it. If the maximum is below the
  threshold, the gate cannot pass and the experiment is dead before it starts.
- **Calibrate at the depth, budget and scale it will actually run at.** Both failures came from a
  cheap proxy measurement standing in for the real regime. A smoke run is the wrong instrument for
  setting a threshold, however convenient it is.
- **Prefer a RATIO to an absolute.** EXP-057's gate would have had six orders of margin instead of
  two as `critic_within_rms / return_within_rms`. A ratio is scale-free; an absolute silently
  tracks episode length and signal magnitude.
- **Gate the quantity that DISCRIMINATES the arms.** EXP-058 gated on storing, but its arms differ
  at the READ site: `use_memory = (readout != "concept")` means all three arms store, and the
  amnesic arm zeroes `W_rec` on read. Arm A's 6.02 against arm M's 6.17 is the same quantity
  measured twice. Ask what the arms actually differ in, and measure that.
- **A gate is more expensive to get wrong than an assertion, and it gets less review**, because it
  reads like bookkeeping next to the claims. Give it the same scrutiny as the claim it guards.
- **GATE THE COMPARISON'S RESOLUTION, NOT JUST THE ARM'S MECHANISM.** EXP-064 cost 14 hours and
  both of its gates PASSED: the regions genuinely trained (`region_drift` 2.71) and the spiking
  pathway genuinely fired (`motor_rate_mean` 0.153, zero silent steps). ==Both arms then scored
  exactly **0.0000**.== A contrast between two arms on the floor is 0 by construction, so the
  primary was decided before the run started. Both gates guarded the *arm*; neither could see
  that the *contrast* had no resolution. **Ask what reading would prove the comparison could
  have come out either way, and gate on that too.**
- **A CONTROL MUST BE A WORKING REFERENCE, NOT JUST A MATCHED ONE.** EXP-064's control was
  capacity-matched to 0.36% and was never checked for **competence**. EXP-043 had run the same
  config with a *linear* head at depth 5 and scored **0.3229**; adding `head_hidden=218` to match
  parameters dropped it to **0.0000** with entropy 0.0120 and modal action 1.000. ==Matching cost
  0.32 and bought a dead policy.== The floor was calibratable **before dispatch from a number
  already in the repo**. **Before using an arm as a reference, confirm it still clears the floor
  in the regime it will run in.** This is the third distinct matching failure: EXP-030 matched a
  control so closely it became bit-identical to its arm, EXP-063 matched a control to its arm but
  neither to the baseline, and EXP-064 matched capacity but not competence.
- **A TOLERANCE CALIBRATED ON A SAME-MACHINE COMPARISON CANNOT BE REUSED ACROSS MACHINES.**
  EXP-067 imported EXP-036's 0.02 replication tolerance unchanged, on the stated and correct
  principle that a replication inventing its own bar is not a replication. But that tolerance was
  set where the replication noise is **exactly zero** (same machine, byte-identical). Across
  machines the per-seed sd is **0.1496**, so the se at n=12 is **0.0432** and the bar sits at
  **0.46 se**: under the null of equivalent machines it fails about **two times in three**
  (P(pass) = 0.326, exact sign-flip null). ==Reusing a threshold is right; reusing it without a
  POWER statement is the same regime error in new clothes.== And it was visible beforehand, which
  is the EXP-064 shape again: the week-25 audit had **printed the retrained seed-0 success rate on
  screen the day before**, under a header saying it was comparing checkpoints and *"not a success
  rate"*. One subtraction against the published value gave 3.3x the tolerance.
  **Before reusing a threshold, state what it can detect in the NEW regime.**
- **PUT THE UNRESOLVED BAND IN THE VERDICT FUNCTION, NOT ONLY IN THE PROSE.** EXP-068's spec did
  the hard part right: it stated before dispatch that a share between **0.25 and 0.45** could not
  be cleanly resolved and must be reported as **unresolved** rather than rounded to the nearest
  verdict. It measured **0.267**. But the band lived only in the spec's prose, so `aggregate.py`
  compared against the bar and printed **CONFIRMED**, with the band as a separate warning line.
  ==A reader running the aggregator sees the verdict the spec forbids.== The code was deliberately
  NOT edited afterwards, even though the edit would have made the result *weaker*: "it made the
  result weaker" is exactly the argument that justifies the reverse next time. **A threshold you
  can state is a threshold you can encode. If a spec names a band where no verdict may be read,
  the aggregator must return that band as a verdict of its own.**
- **Amending a gate is legitimate only before a number exists.** EXP-057's threshold was amended
  that way and the amendment is dated in its spec. EXP-058's was not amended, deliberately: by then
  the numbers existed, and editing it would have been the outcome-dependent editing the whole
  practice exists to prevent. **A gate you got wrong costs an experiment. Rewriting it afterwards
  costs the method.**

## Architecture invariants

- **`recall=False` means only the sensory region is on the policy path.** The policy head reads the sensory concept, which is computed upstream of the hippocampus, so 318 of the five-region brain's 510 neurons are off-path. Any "does architecture X help?" comparison must first establish what is actually on the policy path, or it measures width rather than topology.
- **Action-space width comes from `N_ACTIONS` / `len(MOVES)` / `env.action_space.n`, never a literal.** The 2x2 cube is 6 moves; a 3x3 is 12 or 18.
- **The 6-move cube set is a 2x2-only simplification.** A 2x2 has no centres, so `U == D'`, `R == L'`, `F == B'`; holding the DLB corner still removes the redundancy. A 3x3 has fixed centres and needs all six faces.
- **Distance-to-solved is an instrument, never a model input.** The observation is raw facelets.
- **The TRAINING call passes `feature_fn=readout` for every readout, including `"concept"`.** Only
  the two evaluation calls pass `None`. So `MemoryReadout` is on the policy path of every cube run
  ever recorded, not just the memory arms. Its `__call__` used to wrap the whole body in
  `torch.no_grad()`, which detached the concept: EXP-047's first fine-tuning implementation
  trained nothing, `fc1.weight` moved by exactly 0.0, and the run produced a perfectly ordinary
  success rate. The concept branch now returns before that `no_grad` (inert for frozen runs).
  **Any future "make X trainable" change must verify the gradient ARRIVES AT THE PARAMETER, not
  that the switch is set.** A frozen-vs-trainable comparison where both arms are secretly frozen
  looks exactly like a null result.
- `brain.step` costs about 90 ms and dominates every runtime estimate.

## There is no publishing deadline

**Content Day is defunct - Michael does not post anything** (stated 2026-08-15). The recurring
"Content Day" calendar events, and the Video 8-9 / Written Post #6 / subscriber-and-Patreon review
they carry, are leftovers from the original 2026-03-31 plan. The vault recorded the
media/monetization track as dropped 2026-06-25; this goes further.

Several handoffs treated "Content Day is Aug 16" as a hard deadline and prioritised rendering
against it. **That urgency was invented by the docs, not by Michael.** Visual work is still
wanted - the manim scenes were asked for and liked - but as explanatory artefacts, not as content
with a ship date. Do not schedule work against a publishing date, and do not let a handoff
reintroduce one.

## Research habits

- **Per-experiment `RESULTS.md`, committed.** Standing since the 2026-07-13 audit found EXP-027's numbers living only in a gitignored `outputs/` folder. Include provenance: seeds, date, machine, regeneration command.
- **Pre-register the interpretation contract before the numbers exist**, and mark each claim confirmed or refuted afterwards. EXP-028's headline refuted its own pre-registration, which is exactly why this is worth doing.
- **n >= 12 seeds.** n=5 lied in EXP-026 and the de-noised result flipped.
- **THE SEED IS A CONFOUND WORTH 0.09, AND IT IS INHERITED.** A cube seed fixes the E0 encoder, the
  train/held-out split and the head init at once, so seed quality is one persistent property that
  every downstream stage inherits. Measured over 8 depth-5 arms from five experiments: per-seed
  correlation **+0.419, positive in 28 of 28 arm pairs**, sd **0.0906**. That is **0.0388** of noise
  on any comparison between two DISJOINT seed sets, against published effects of 0.05 to 0.09, and
  **exactly zero** on a paired same-seed comparison. EXP-060's "unexplained level shift" was this
  and nothing else (exact p **0.5066**), and its encoder-manufacturing suspect was wrong.
  **Never compare arms across different seed sets, and run `scripts/seed_effect.py` before
  explaining a block difference.** Full note with the eliminations: `docs/seed-effect.md`.
- **Measure the chance floor, do not assume it.** On the cube it is 21% at depth 1, not 1/6, because a random walk with a `2d+3` budget can stumble into solved.
- **Ask what a control holds fixed besides the thing you named.** A shuffle-null that varies the query state also varies "features of the current observation"; a path-matched control can turn out bit-identical to the arm it is controlling for. EXP-030 is the worked example: `memory` beat the shuffle-null by 10.8 points (p 0.078) and beat the amnesic control by 1.2 (p 0.91). The primary comparison was measuring the harm of *incorrect* memory, not the benefit of correct memory. Three arms would have published a false positive.
- **FIVE INSTRUMENTS ARE RETIRED and must not gate a decision**: the EXP-033 probe, pretraining
  move-accuracy, the entropy trace, `S`, and `critic_ev`. Use `revisit_rate` and `optimality`.
  Each works as a THRESHOLD and fails as a GRADIENT, and "unanimous at p 0.0005" measures an
  instrument's consistency rather than its link to the outcome. Full note with the evidence and
  the four checks to run before adopting a new instrument: `docs/retired-instruments.md`.
- **Prefer a mechanism measurement to a performance measurement.** "Memory did not help" is weak and unactionable. "Memory was on the policy path, cycles were abundant, and the revisit rate did not fall" localises the failure to the readout. Instrument the mechanism the intervention is supposed to drive, not just the score.
- **An instrument that cannot detect the defect it exists for is the gate-that-cannot-fail trap in
  disguise, and only MUTATION TESTING finds it.** EXP-061's leak detector compared the real recall
  against the noise vector, both taken BEFORE the substitution, so it read ~0 by construction and
  was blind to `recall = 0.5 * recall + 0.5 * noise` - a 50% leak that passed every assertion. Six
  of seven mutations were caught; that one survived and exposed the blind spot. **Break the
  implementation deliberately and check each test fails against the bug it names.** Same practice
  found a missing PARTIAL-failure case in EXP-059's aggregator gate.
- **A TEST'S FIXTURE CAN DISARM THE TEST, and mutation testing is what shows it.** EXP-063's
  encoder-freeze test built its brain output under `torch.no_grad()`, the convenient thing to do.
  The concept therefore arrived already detached, so deleting the readout's own `no_grad` changed
  nothing, the test still passed, and the mutation survived. The test only became able to fail
  once its input was built WITH a live graph. **Ask what the fixture already guarantees before
  trusting what the assertion appears to check** - a test that cannot see the defect is worse
  than no test, because it is counted as coverage. 18 of 18 mutations caught after that one fix.
- **Read real output, not only green tests.** A cube frame labelled `solved: yes` on a scrambled cube passed every unit test in the suite; two minutes reading an actual recorded trace found it. Same pattern as the filename collision that was visible in an implementer's own smoke output. Tests prove what you thought to assert; output shows what you did not.
- **No scipy in the venv.** For n around 12, an exact paired permutation test over all `2**n` sign flips is cheap, assumption-free, and better than a normal approximation. 12 seeds is 4096 flips.

## Running long experiments on the laptop over SSH

**Full procedure lives in `docs/playbooks/remote-experiment-runs.md`. Read it before dispatching.**
Established 2026-07-09, revised 2026-07-30 for the move to a VPS.

**Reach the laptop over Tailscale, never the LAN address, and always via the `ssh laptop` alias.**
Spelling the host out as `mlgbr@swizzlesduo.tailda519d.ts.net` **fails with `Permission denied
(publickey)`**: ssh matches `Host` patterns against what you typed, not the resolved name, so the
fully-qualified form misses the config block, never offers `id_ed25519_backup`, and falls back to
an encrypted key that cannot sign under `BatchMode`. Verified 2026-08-03.
(Tailscale IPv4 `100.120.6.78`, ED25519 host key `SHA256:uKE4XW17ZJ106FoTifyv+WEahvbNkn3DhhovrXsEB6Y`).
The old `192.168.50.62` is an RFC1918 address and is **unreachable from anywhere but the home network**.
`SwizzlesDuo` is an Intel Ultra 9 185H, 22 cores, 31.4 GB. Its remote default shell is `cmd.exe`, not POSIX,
so wrap everything in `powershell -NoProfile -Command`.

- **Budget RAM from measurement, not from a rule of thumb.** Cube workers peak around **195 MB each**
  (1.58 GB across 8). The older "about 1.5 GB per worker" figure came from a heavier grid workload and is
  roughly 7x too conservative for cube runs.
- **Falling free memory is usually not a leak.** Windows `FreePhysicalMemory` excludes standby and cache,
  so it drops steadily through a run that writes many files and recovers at the end. It fell 14 GB to
  4.6 GB during EXP-030 while worker resident memory stayed flat. Check per-process `WorkingSet64` before
  concluding anything.
- **Worker processes appear as `python3.13.exe`, not `python.exe`.** `Get-Process python` misses them completely. Match on `^python` or check both names before concluding a run has died.
- **A parent process at ~0 CPU is normal.** It only waits on workers. The real health signal is worker count against outstanding tasks, plus the record-file count climbing.
- **An SSH drop does not kill the run.** Windows has no SIGHUP semantics. Client `exit code 255` means the
  connection dropped, not that the job died: reconnect and probe before reacting. Always `Tee-Object` to a
  log file so the record survives the pipe.
- **Do not chain `if not exist X mkdir X && python ... > log` under `cmd`.** It wedges silently at 0.016 s CPU with no output. Use PowerShell, and `python -u` so the log is not fully buffered.
- **Quoting through `cmd.exe` eats trailing backslashes and interprets `|` before PowerShell sees it.**
  For anything non-trivial, `scp` a `.ps1` over and run it with `powershell -NoProfile -ExecutionPolicy Bypass -File`.
- Progress is best read from the per-run JSON record count, not the log.
- **`ssh -n` makes an interactive gate stop cleanly** (`input()` raises `EOFError`), which is what you want
  when a driver prints a pre-flight number you must read. Pass `--skip-gate` only after reading it.
- **Seeded runs are byte-identical across worker scheduling ON ONE MACHINE, and NOT across
  machines.** Re-running a seed and diffing the records is a free correctness check on the seeding
  discipline, and it stays that. But measured 2026-09-22: retraining EXP-036 depth 3 seed 0 on a
  different x86 box shares **0 of 390** parameters with its published head (cosine 0.524), and that
  experiment uses **no pretrained encoder**. **Re-evaluating a tracked checkpoint IS portable** -
  every headline metric matches to full float repr, with one derived mean differing by 1 ULP. So
  the reproducibility guarantee is *re-evaluate the checkpoints*, never *retrain from the seed*.
  See `docs/reproducibility-audit.md`. **EXP-067 then measured what SURVIVES:
  2 of 3 pre-registered verdicts replicated on a second machine and the numeric bar did not, and
  the task layer (shells, held-out splits, scramble streams) is byte-identical on 12 of 12 seeds,
  so every bit of the divergence is in training.** See
  `experiments/067_cross_machine_replication/RESULTS.md`.
