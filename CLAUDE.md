# CLAUDE.md

Standing knowledge for this repo. Session-specific state lives in `docs/handoffs/`; this file is only for things that stay true.

## Commands

```powershell
.venv\Scripts\python.exe -m pytest tests/ -q -m "not slow"   # fast inner loop, about 13 min
.venv\Scripts\python.exe -m pytest tests/ -q                 # full suite incl. the slow BFS test
```

Always run python via `.venv\Scripts\python.exe`. One test is marked `slow` (it builds the full 3,674,160-state cube BFS table, about 67s). The `slow` marker is registered in `pyproject.toml` and is NOT deselected by default, deliberately: a default run stays honest.

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

## Architecture invariants

- **`recall=False` means only the sensory region is on the policy path.** The policy head reads the sensory concept, which is computed upstream of the hippocampus, so 318 of the five-region brain's 510 neurons are off-path. Any "does architecture X help?" comparison must first establish what is actually on the policy path, or it measures width rather than topology.
- **Action-space width comes from `N_ACTIONS` / `len(MOVES)` / `env.action_space.n`, never a literal.** The 2x2 cube is 6 moves; a 3x3 is 12 or 18.
- **The 6-move cube set is a 2x2-only simplification.** A 2x2 has no centres, so `U == D'`, `R == L'`, `F == B'`; holding the DLB corner still removes the redundancy. A 3x3 has fixed centres and needs all six faces.
- **Distance-to-solved is an instrument, never a model input.** The observation is raw facelets.
- `brain.step` costs about 90 ms and dominates every runtime estimate.

## Research habits

- **Per-experiment `RESULTS.md`, committed.** Standing since the 2026-07-13 audit found EXP-027's numbers living only in a gitignored `outputs/` folder. Include provenance: seeds, date, machine, regeneration command.
- **Pre-register the interpretation contract before the numbers exist**, and mark each claim confirmed or refuted afterwards. EXP-028's headline refuted its own pre-registration, which is exactly why this is worth doing.
- **n >= 12 seeds.** n=5 lied in EXP-026 and the de-noised result flipped.
- **Measure the chance floor, do not assume it.** On the cube it is 21% at depth 1, not 1/6, because a random walk with a `2d+3` budget can stumble into solved.
- **Ask what a control holds fixed besides the thing you named.** A shuffle-null that varies the query state also varies "features of the current observation"; a path-matched control can turn out bit-identical to the arm it is controlling for.

## Running long experiments on the laptop over SSH

Established 2026-07-09, hard-won details as of 2026-07-27.

```powershell
ssh mlgbr@192.168.50.62      # SwizzlesDuo, Intel Ultra 9 185H, 22 cores. Remote shell is cmd.exe, not POSIX.
cd C:\Users\mlgbr\Desktop\Projects\neuromorphic-nn-snn-research-project
git pull --ff-only origin main
New-Item -ItemType Directory -Force experiments\NNN_x\outputs | Out-Null
.venv\Scripts\python.exe -u experiments\NNN_x\run.py --seeds 0 1 2 3 4 5 6 7 8 9 10 11 --workers 16 | Tee-Object -FilePath experiments\NNN_x\outputs\run.log
```

- **RAM is the binding constraint, not cores.** Budget about 1.5 GB per worker. Check free memory first: `[math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB,1)`. A run launched with 4.7 GB free lost all 16 workers mid-flight.
- **Worker processes appear as `python3.13.exe`, not `python.exe`.** `Get-Process python` misses them completely. Match on `^python` or check both names before concluding a run has died.
- **A parent process at ~0 CPU is normal.** It only waits on workers. The real health signal is worker count against outstanding tasks, plus the record-file count climbing.
- **Do not chain `if not exist X mkdir X && python ... > log` under `cmd`.** It wedges silently at 0.016 s CPU with no output. Use PowerShell, and `python -u` so the log is not fully buffered.
- Progress is best read from the per-run JSON record count, not the log.
- Pass `--skip-gate` for any driver with an interactive prompt; `input()` raises `EOFError` over non-interactive SSH.
