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

**Counts:** 370 tests 2026-07-30, 521 2026-08-28, **561 as of 2026-09-01 (546 not
slow, 15 slow)**.

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
- **Measure the chance floor, do not assume it.** On the cube it is 21% at depth 1, not 1/6, because a random walk with a `2d+3` budget can stumble into solved.
- **Ask what a control holds fixed besides the thing you named.** A shuffle-null that varies the query state also varies "features of the current observation"; a path-matched control can turn out bit-identical to the arm it is controlling for. EXP-030 is the worked example: `memory` beat the shuffle-null by 10.8 points (p 0.078) and beat the amnesic control by 1.2 (p 0.91). The primary comparison was measuring the harm of *incorrect* memory, not the benefit of correct memory. Three arms would have published a false positive.
- **Prefer a mechanism measurement to a performance measurement.** "Memory did not help" is weak and unactionable. "Memory was on the policy path, cycles were abundant, and the revisit rate did not fall" localises the failure to the readout. Instrument the mechanism the intervention is supposed to drive, not just the score.
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
- **Seeded runs are byte-identical across worker scheduling.** Re-running a seed and diffing the records is
  a free correctness check on the seeding discipline.
