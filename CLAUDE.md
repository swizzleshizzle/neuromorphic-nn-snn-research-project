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

Full suite is 370 tests as of 2026-07-30.

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
