# Session Handoff - 2026-07-31 (Fri) -> first working VPS session

> **RUN COMPLETE. EXP-031 concluded: the EXP-030 memory null was measured on a collapsed
> policy.** At depth 3, seven of twelve concept seeds play ONE action for all nine steps of
> every episode. Full numbers in `experiments/031_policy_collapse/RESULTS.md`.
>
> **Next job is section 8.** Read `CLAUDE.md` first, then
> `docs/playbooks/remote-experiment-runs.md`. **That playbook has two errors this session found
> the hard way**; see section 5.

## 1. The VPS is now a real dev box

The 2026-07-30 handoff left first-time setup undone. It is done:

```bash
uv venv --python 3.10 .venv
uv pip install --python .venv/bin/python --index-strategy unsafe-best-match \
    --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
uv pip install --python .venv/bin/python -e ".[server]"
```

- The CPU torch index matters: the default PyPI resolution drags in about 2.5 GB of `nvidia-*`
  wheels this box has no use for. Result is torch 2.13.0+cpu, 1.3 GB total.
- `-e .` is required, not optional. The repo is a `src/` layout, so without it `import
  neuromorphic` fails.
- **`requirements.txt` is incomplete.** `fastapi`, `uvicorn` and `httpx` exist only as the
  `[server]` extra in `pyproject.toml`, so installing from `requirements.txt` alone leaves
  `tests/server/` failing at collection with `ModuleNotFoundError: fastapi`. Not yet fixed;
  see section 6.
- Verified: **369 passed, 1 deselected** in 10m07s. That plus the slow BFS test is the 370
  in `CLAUDE.md`.
- `dashboard/node_modules` installed with `npm ci` (the lockfile is npm, not pnpm).

## 2. What shipped: the policy-collapse instrument

Branch `week17-collapse-instrument`, commit `0017053`, pushed. Not merged, no PR yet.

The 2026-07-30 handoff named this the next job: is the depth-3 policy collapsed to a constant
action, which would make the EXP-030 memory null a fact about a degenerate policy rather than
about memory?

**That handoff proposed measuring it across EXP-030's existing seeds. That is not possible.**
`cube_baseline.py` never saves a policy (no `torch.save`, no `state_dict`) and `outputs/` is
gitignored and lived only on the laptop. No artifact survives a run. The measurement has to be
taken *during* a re-run, which is why the instrument had to exist before anything could be
measured.

Two readouts, because the `F'`x9 observation was made under the GREEDY policy while
`entropy_beta=0.0` is a TRAINING setting, and those can fail independently:

| field | where | meaning |
|---|---|---|
| `greedy_modal_action_frac` | `evaluate_states` | fraction of a rollout spent on its single most-common action, averaged per episode |
| `mean_train_entropy` | `run_cube_baseline` | per-episode policy entropy, already computed by `reinforce.py:188` and previously discarded |

Both are additive. No existing record field changes.

**Thresholds are measured, not assumed.** Over 20,000 simulated uniform rollouts on 6 actions
the modal fraction averages 0.354 at a 9-step budget and 0.429 at 5 steps; a collapsed policy
scores exactly 1.0. The finished implementation then measured **0.357** on the real
random-policy path, matching a prediction made before it existed.

Interpretation caveat worth carrying: an untrained 3-episode depth-1 run scores 0.833, because
a 5-step budget inflates the statistic. **Depth 3 (9 steps) is the meaningful read**, where
uniform is 0.354 and collapse is 1.0.

The random-policy test bar (0.25 to 0.60) also discriminates the per-episode design from a
pooled one, which would read about 0.17. That is deliberate: it fails if the metric is ever
rewired to pool.

## 3. What the run was, and what it found

**Completed 2026-08-01 00:37 laptop local, 3h14m, 144/144 records, exit 0, zero errors.**

Neutrality passed first: all 144 records reproduce EXP-030's pre-existing fields **exactly**
(`scripts/verify_instrument_neutrality.py`, 144 identical / 0 differing), so nothing is
attributable to the code change.

Depth 3, where 99% of episodes run the full 9-step budget and the metric is clean, concept arm
per seed:

```
0.684  0.719  0.789  0.996  0.996  1.000  1.000  1.000  1.000  1.000  1.000  1.000
```

against a 0.354 uniform floor. Training entropy is 17-30% of the log 6 ceiling across arms.

**The finding was already latent in EXP-030's published numbers.** It reported 7 of 12 seeds at
exactly 0.667 greedy revisit rate at depth 3; a quarter turn has order 4, so a 9-step rollout
visits 10 states of which 4 are unique: `(10-4)/9 = 0.667`. Two instruments sharing no code
identify the same seven seeds.

**Depth 1 is discarded, not reported.** 83% of episodes end early and a one-move solve scores 1.0
by construction; `corr(modal, success)` is +0.254 there versus -0.920 and -0.547 at depths 2 and 3.
`aggregate.py` enforces this in code and marks depth 1 `DISCARD` automatically.

## 3b. Original dispatch notes (kept for the recovery procedure)

144 records, 4 arms x depths 1-3 x seeds 0-11, `--workers 16`, `--skip-gate`. Identical
configuration to EXP-030 except for the two new fields.

`--skip-gate` is correct here: the revisit gate was already read and passed in EXP-030
(greedy 0.089 / 0.327 / 0.604). This re-run instruments a decided configuration, it does not
re-decide it.

**Free correctness check when the records land.** The instrument consumes no randomness, so
every pre-existing field should come back **byte-identical to EXP-030's** numbers in
`experiments/030_memory_engagement/RESULTS.md`. If they differ, the instrument perturbed the
experiment and the new numbers cannot be trusted. Check this before interpreting anything.

**What the result would mean:**

- Depth-3 `greedy_modal_action_frac` near 1.0 -> the policy is collapsed, and the depth-3
  memory null says nothing about memory. Next step is a re-run of engagement with
  `entropy_beta > 0` and `normalize_advantages=True`, which is the fix EXP-025 already applied
  once.
- Near 0.354 -> the policy is not collapsed, the `F'`x9 trace was one unlucky seed, and the
  EXP-030 conclusion stands as written.
- Splits by arm -> more interesting than either, and would mean the memory arms and concept
  arm degenerate differently.

## 4. Recovering the run

Output dir: `experiments/030_memory_engagement/outputs_instrumented/` **on the laptop**
(deliberately NOT EXP-030's `outputs/`, so nothing is overwritten).

```bash
ssh -n swizzlesduo 'powershell -NoProfile -ExecutionPolicy Bypass -File C:\Users\mlgbr\probe.ps1'
# -> records=N pyprocs=P free_gb=F errors=E ; 144 records means done
```

Scripts already on the laptop at `C:\Users\mlgbr\`: `probe.ps1` (poll), `launch2.ps1`
(the working launcher), `launch.ps1` (the BROKEN detached one, see section 5).

Retrieve with:

```bash
scp "swizzlesduo:C:/Users/mlgbr/Desktop/Projects/neuromorphic-nn-snn-research-project/experiments/030_memory_engagement/outputs_instrumented/*.json" ./local_dir/
```

**The run is a direct child of an SSH session held open by the dispatching session.** If that
session is gone, check whether the run survived before assuming it did. If it did not, re-launch
with `launch2.ps1`; seeds are deterministic so nothing is lost but time.

## 5. Two playbook corrections

**`Start-Process` detaching does NOT work.** The first launch used
`Start-Process -WindowStyle Hidden -PassThru` to survive SSH disconnect. It returned a pid and
the process was dead moments later with **empty stdout and empty stderr**, which reads exactly
like "never launched". Windows OpenSSH tears down the session's job object on session close and
takes detached children with it. The working pattern is the playbook's original one: a direct
child of the session, `Tee-Object` to a log. If genuine detachment is ever needed, it has to be
a Scheduled Task, not `Start-Process`.

**The playbook's SSH checklist item 3 was never satisfied, and the fix is not obvious.** Two
independent problems, both silent:

1. **`mlgbr` is a local Administrator** (verified). Windows `sshd` then reads
   `C:\ProgramData\ssh\administrators_authorized_keys` and **ignores `~/.ssh/authorized_keys`
   entirely**. A key added to the latter fails with no diagnostic. The file also needs
   `icacls <file> /inheritance:r /grant "SYSTEM:F" /grant "Administrators:F"` or sshd silently
   refuses it.
2. **`~/.ssh/id_ed25519` on the VPS is passphrase-encrypted and cannot sign under BatchMode.**
   The automation key is `~/.ssh/id_ed25519_backup`
   (`SHA256:PSClMTuneO8tnmwyB7+YBsU35rlVsUGzazRH6rxv8W0`). This was already documented in the
   `serverlocal-wsl` block of `~/.ssh/config` and was still gotten wrong once this session.
   **Read `~/.ssh/config` before handing anyone a key.**

`~/.ssh/config` now has the laptop block corrected: user `mlgbr`, the backup key, the MagicDNS
name, and both gotchas in comments.

## 6. Open debt

Carried forward from 2026-07-30 and still open:

1. Nothing in the merged dashboard cube work has been seen rendering in a browser.
2. `cubeNet.ts` applies row-major ordering to all six faces; B and D are probably not
   physically coherent. Cosmetic, needs eyes.
3. Move correctness and net geometry are pinned independently, their composition is not.
4. The held DLB corner highlight (facelets 12, 16, 21) has no test pinning its placement.
5. The shuffle control is diluted at shallow depth (`unshuffled_frac` 0.321 at depth 1).
6. Live trace streaming during training is still a spec, not built.
7. `test_random_arm_scores_above_zero_but_well_below_one` at
   `tests/training/test_cube_baseline.py:70` asserts `0.0 <= x <= 1.0`, which cannot fail. It
   now sits three lines above tests that do carry measured bars. Left alone this session only to
   avoid scope-creeping the instrument commit.

New this session:

8. `requirements.txt` does not install the `[server]` extra, so a fresh checkout fails
   `tests/server/` collection. One line to fix.

## 8. THE JOB NEXT: sweep the collapse fix, then re-test memory on a healthy policy

Full reasoning in `experiments/031_policy_collapse/RESULTS.md` under "Lead for the next
experiment". In short:

1. **Do not assume the known fix transfers.** ADR 0001 established on the GRID WORLD, with an
   MLP head and 4 actions, that `entropy_beta=0.01` alone did nothing and that advantage
   normalization **plus** `entropy_beta=0.05` eliminated collapse. This is the cube: 6 actions,
   linear head. Sweep `entropy_beta` in `{0.0, 0.01, 0.05, 0.1}` with
   `normalize_advantages=True`, depth 3, 12 seeds, concept arm only. 48 runs, about an hour at
   `--workers 16`.
2. **Pre-register the collapse gate before looking**: the memory arms may not launch unless the
   depth-3 concept arm reaches `greedy_modal_action_frac < 0.60` and `mean_train_entropy > 1.2`.
3. **Judge the sweep on the instruments, not on success rate.** The claim under test is that the
   policy reads its input at all. Depth-3 success may stay near its current 2.2% even if collapse
   is fixed, and that would still be progress.

`CubeConfig` already carries both `entropy_beta` and `normalize_advantages`, so no production
code change is needed to run the sweep, only a driver.

**Caveat worth carrying:** depth 2 is where EXP-030's +10.8 point primary effect lives, and it is
partially collapsed (2 to 5 of 12 seeds). Whatever the sweep finds at depth 3 should be checked at
depth 2 before EXP-030's headline is either defended or retired.

## 7. Pointers

- Standing knowledge: `CLAUDE.md`
- Remote runs: `docs/playbooks/remote-experiment-runs.md` (see section 5 for its two errors)
- **EXP-031 results (this session's finding): `experiments/031_policy_collapse/RESULTS.md`**
- **The collapse fix, as actually established: `docs/adr/0001-multi-region-training-strategy.md`,
  Amendment 2. Note EXP-025 has `run.py` and `aggregate.py` but NO committed `RESULTS.md`, so the
  ADR is the only record. Do not cite "EXP-025" for the fix; cite the ADR.**
- EXP-030 results: `experiments/030_memory_engagement/RESULTS.md`
- EXP-030 design and pre-registration:
  `docs/superpowers/specs/2026-07-27-memory-engagement-design.md`
- Previous handoff: `docs/handoffs/SESSION-HANDOFF-2026-07-30.md`
- Vault: `Weekly Notes/week-17-memory-engagement.md`
