# Playbook: dispatching experiment runs to the laptop

How to run a long experiment on `SwizzlesDuo` (the laptop) from wherever the session is hosted.
Written 2026-07-30, when sessions moved from the desktop to a VPS. Every command and value below was
verified on that date.

## 0. The one thing that breaks when you move off the home network

The old recipe in `CLAUDE.md` used `ssh mlgbr@192.168.50.62`. That is an RFC1918 private LAN address.
**A VPS cannot reach it.** Anything hosted outside the house must go over Tailscale.

| | value |
|---|---|
| Preferred host | `swizzlesduo.tailda519d.ts.net` |
| Tailscale IPv4 | `100.120.6.78` (fallback; the DNS name survives IP changes) |
| User | `mlgbr` |
| Host key (ED25519) | `SHA256:uKE4XW17ZJ106FoTifyv+WEahvbNkn3DhhovrXsEB6Y` |
| Hardware | Intel Core Ultra 9 185H, 22 cores |
| Remote default shell | **`cmd.exe`, not POSIX.** Always wrap in `powershell -NoProfile -Command`. |

Prefer the MagicDNS name. Use the raw `100.x` address only when DNS is not resolving.

### Always `ssh laptop`, never `ssh mlgbr@swizzlesduo.tailda519d.ts.net`

Corrected 2026-08-03, after the raw form failed at the start of a dispatch. Every command in
this playbook used to spell the host out in full. **That form no longer authenticates:**

```
$ ssh mlgbr@swizzlesduo.tailda519d.ts.net 'echo ok'
mlgbr@swizzlesduo.tailda519d.ts.net: Permission denied (publickey,password,keyboard-interactive).
$ ssh laptop 'echo ok'
ok
```

The VPS's `~/.ssh/config` carries a `Host swizzlesduo laptop` block pinning
`IdentityFile ~/.ssh/id_ed25519_backup` with `IdentitiesOnly yes`. **ssh matches `Host` patterns
against the name you typed on the command line, not against the resolved hostname**, so the
fully-qualified form matches nothing, never offers that key, and falls back to
`~/.ssh/id_ed25519`. That key is aes256-ctr encrypted and cannot sign under `BatchMode`, so the
failure surfaces as a bare "Permission denied (publickey)" that reads exactly like a missing
`authorized_keys` entry and sends you debugging the wrong machine.

The alias is also what makes the ConnectTimeout and keepalive settings apply.

### First connection from a new machine

The VPS will refuse to connect until it trusts the host key. Do not reach for
`StrictHostKeyChecking=no`. Verify the fingerprint against the table above:

```bash
ssh-keyscan -t ed25519 swizzlesduo.tailda519d.ts.net 2>/dev/null | ssh-keygen -lf -
# expect: SHA256:uKE4XW17ZJ106FoTifyv+WEahvbNkn3DhhovrXsEB6Y
```

If it matches, trust it:

```bash
ssh-keyscan -t ed25519 swizzlesduo.tailda519d.ts.net 100.120.6.78 >> ~/.ssh/known_hosts
ssh -n laptop 'powershell -NoProfile -Command "$env:COMPUTERNAME"'
# expect: SWIZZLESDUO
```

If it does not match, stop. Something is wrong with the tailnet, not with your quoting.

### Checklist before you trust any of this

1. The VPS is logged into the **same tailnet** as the laptop (`tailscale status` on both).
2. The laptop is **awake**. Tailscale does not wake a sleeping machine. A run cannot start on a laptop
   that has suspended, and a running job will stall if the machine sleeps mid-flight.
3. The VPS's SSH public key is in the laptop's `authorized_keys`.

## 1. Sync the repo on the laptop

**Use the script. A bare `git pull` here has silently failed twice.**

```bash
scp scripts/laptop/sync_repo.ps1 laptop:C:/Users/mlgbr/sync_repo.ps1
ssh -n laptop 'powershell -NoProfile -ExecutionPolicy Bypass -File C:\Users\mlgbr\sync_repo.ps1'
# expect the last line to be exactly: SYNCED     (exit 0)
```

The laptop keeps its own clone at `C:\Users\mlgbr\Desktop\Projects\neuromorphic-nn-snn-research-project`.
It is a separate checkout: local branches on the VPS do not exist there until pushed and pulled.

### Why a bare `git pull` is not enough

Head checkpoints are **generated on the laptop and committed from the VPS**. `.gitignore` ignores
`experiments/*/outputs/*` but then negates `!experiments/*/outputs/*_head.pt`, so those checkpoints are
tracked deliberately (EXP-030's memory re-ask needs them; retraining to recover them costs 20 h).

So after every run the laptop holds **untracked** copies at exactly the paths the VPS has since
committed. Git refuses to overwrite an untracked file on merge, the pull fails, and **the checkout stays
on an old commit while the dispatch proceeds against stale source.**

- Before the EXP-037 dispatch the laptop was **12 commits behind** and `curriculum_weights` appeared
  **zero times** in its source. Caught only by checking.
- It recurred before EXP-038: **48 colliding files**, stuck at `022d8b8`.

`sync_repo.ps1` moves aside exactly the untracked files the incoming tree contains, into a timestamped
attic at `C:\Users\mlgbr\repo-attic\<stamp>\` (moved, never deleted - that is the undo). It deliberately
does **not** `git clean`, which would also destroy the **gitignored `outputs/*.json` records**; those are
often the only copy of an experiment's data, and EXP-036's records are the comparator for every EXP-037
claim. It aborts rather than touching **modified tracked files**, which may be real work.

It exits non-zero unless `git rev-list --count HEAD..origin/main` is **0**.

> [!warning] Verify the symbol too, not just the exit code
> `SYNCED` proves the commit is right. It does not prove the code does what you think. Also grep the
> laptop's source for the **new symbol you just added**, every dispatch:
>
> ```bash
> ssh -n laptop 'powershell -NoProfile -Command "cd C:\Users\mlgbr\Desktop\Projects\neuromorphic-nn-snn-research-project; (Select-String -Path src\neuromorphic\training\*.py -Pattern \"<new_symbol>\").Count"'
> ```

## 2. Check headroom, then choose `--workers`

```bash
ssh -n laptop 'powershell -NoProfile -Command "$os=Get-CimInstance Win32_OperatingSystem; \"free_gb=\" + [math]::Round($os.FreePhysicalMemory/1MB,1); \"pyprocs=\" + (Get-Process | Where-Object { $_.Name -match \"^python\" }).Count"'
```

**Budget from measurement, not from the old 1.5 GB per worker rule of thumb.** EXP-030's cube workers
peaked at about **195 MB each, 1.58 GB total across 8 workers**. The 1.5 GB figure came from a heavier
grid workload and is roughly 7x too conservative for cube runs. With 31.4 GB installed, 16 workers is
comfortable for cube work.

Also confirm nothing is already running (`pyprocs` should be 0), or you will be sharing cores silently.

### Choose the worker count from MEMORY HEADROOM, not from core count

**Measured 2026-08-05, and it inverts the obvious choice: 10 workers beat 16.**

| workers | private each | system commit | utilisation | effective workers |
|---|---|---|---|---|
| 16 (EXP-036) | 920 MB | **48.6 / 50.4 GB (96%)** | 43.1% | 6.90 |
| **10 (EXP-037)** | 914 MB | **25.4 / 52.4 GB (49%)** | **74.2%** | **7.42** |

Ten workers deliver **more** total throughput than sixteen, from 37.5% fewer processes. At 16
the machine is over-committed and the workers spend most of their time waiting on the pagefile
rather than computing. `SwizzlesDuo` has 22 logical cores, so the core count says 16 is fine;
the core count is not the constraint.

**Budget from PRIVATE bytes, not working set.** These workers show ~80 MB working set and
**~920 MB private**. The old "195 MB per worker" figure in this playbook is a working-set number
and understates the real footprint by roughly 4.7x, which is exactly what made 16 look safe.

Rule of thumb: `workers ~= (commit_limit_gb * 0.5 - baseline_commit_gb) / 0.92`. Check the
baseline before launching, since it varies with whatever else is open:

```bash
ssh -n laptop 'powershell -NoProfile -Command "$os=Get-CimInstance Win32_OperatingSystem; \"commit_used_gb=\" + [math]::Round(($os.TotalVirtualMemorySize - $os.FreeVirtualMemory)/1MB,1) + \" / \" + [math]::Round($os.TotalVirtualMemorySize/1MB,1)"'
```

**Whether fewer than 10 is better again is unmeasured.** 10 is the only point below 16 that has
been tested; do not extrapolate the curve from two points.

### Estimate wall clock from measured throughput, not from the 90 ms figure

`CLAUDE.md` says `brain.step` costs about 90 ms. That is single-step latency and it **understates a
parallel run by roughly 70%**. Calibrating against EXP-035's own wall clock - 3,359,916 steps in
8h55m across 16 workers, so 143 core-hours - the achieved rate is **153 ms/step**. Using 90 ms put
the first EXP-036 estimate at 6.7 h when the honest figure was 11.8 h.

Count steps properly. An episode at depth `d` runs up to `2d+3` steps, and a curriculum SPLITS the
budget across its stages rather than multiplying it, so:

```python
def steps(depth, episodes):
    stages = list(range(1, depth + 1))
    per = episodes // len(stages)
    return sum(per * (2 * d + 3) for d in stages)
```

Add the evaluations: each is `n_states * (2d+3)` steps, and since EXP-036 there are two of them
per run (held-out and train-side).

### ROUND UP TO WHOLE WAVES. Dividing by `workers` is wrong and it is wrong by a lot.

**Corrected 2026-08-22, after EXP-047 was estimated at 23 h and took 42 h.** The old formula here
was `wall_hours = total_steps * 0.153 / 3600 / workers`. That assumes the cells pack perfectly
into the worker pool. They do not: **12 cells on 10 workers is TWO waves**, and the second wave
runs 2 workers busy against 8 idle for its whole duration. The pool does not refill.

```python
import math
def wall_hours(cells, workers, steps_per_cell, s_per_step=0.153):
    return math.ceil(cells / workers) * steps_per_cell * s_per_step / 3600
```

Measured against EXP-047's four phases, all on `SwizzlesDuo`:

| phase | cells | workers | waves | actual | naive `/workers` | **wave-rounded** |
|---|---|---|---|---|---|---|
| pilot | 6 | 6 | 1 | 4.24 h | 5.62 h | 5.62 h |
| confirmatory | 12 | 10 | 2 | **11.95 h** | 6.75 h | **11.25 h** |
| fallback d5 (capped) | 12 | 10 | 2 | **7.80 h** | 4.55 h | **7.58 h** |

**The 0.153 s/step constant was never the problem** - back-solving the per-cell time gives 0.163
and 0.157 s/step at 10 workers, right on it. The formula was.

**Fewer workers is faster per cell**: the 6-worker phase back-solved to **0.115 s/step** against
10 workers' ~0.16. Same memory-bound story as "10 beat 16". So prefer a worker count that
**divides the cell count** - 12 cells on 6 workers is 2 clean waves and each cell runs faster
than it would at 10.

### PRETRAINING is far more contended than RL. Measured 2026-08-25.

EXP-050's phase 1 ran 12 encoder-pretraining jobs at 10 workers, so wave 1 had **10 concurrent**
and wave 2 had **2**. The same job, same machine, minutes apart:

| concurrency | per-seed wall | |
|---|---|---|
| 10 | **6,985 s** | |
| 2 | **1,995 s** | **3.50x faster per seed** |

That is **2.86 effective cores from 10 workers** - far worse than the ~7.4 the RL cells get. Ten
still wins on total throughput (5.15 seeds/h against 2 workers' 3.61) but the returns are brutal,
and the phase took **2.49 h against a 1.7 h estimate** because of it.

**Estimate pretraining separately from RL.** The 0.153 s/step constant and the ~7.4 effective
cores are RL-cell figures; inverse-model pretraining is much more memory-bandwidth-bound. Somewhere
around 4-6 workers is probably the knee, but only 2 and 10 have been measured - **do not
extrapolate the curve from two points**, which is the same warning this playbook already gives
for the 10-versus-16 comparison.

> One EXP-047 phase (EXP-040 depth 5, 12 cells at 10 workers, overnight) ran at **0.301 s/step**,
> roughly double its near-identical sibling phase. Never explained. It changed no result -
> seeded runs are byte-identical regardless of scheduling - but do not treat a single phase's
> throughput as a constant you can plan the next run against.

## 3. Launch

```bash
ssh -n laptop 'powershell -NoProfile -Command "cd C:\Users\mlgbr\Desktop\Projects\neuromorphic-nn-snn-research-project; New-Item -ItemType Directory -Force experiments\NNN_x\outputs | Out-Null; .venv\Scripts\python.exe -u experiments\NNN_x\run.py --seeds 0 1 2 3 4 5 6 7 8 9 10 11 --workers 16 | Tee-Object -FilePath experiments\NNN_x\outputs\run.log"'
```

- `-n` on ssh redirects stdin from `/dev/null`. That matters: a driver with an interactive gate calls
  `input()`, which raises `EOFError` and stops cleanly instead of hanging. **That is the desired
  behaviour when you want to read a pre-flight gate before committing to the full run.** Pass
  `--skip-gate` only after you have read the gate number.
- `python -u` keeps the log unbuffered.
- Run this in the background from the session's side. It is normal for it to take hours.

**Do not** chain `if not exist X mkdir X && python ... > log` under `cmd`. It wedges silently at
0.016 s CPU with no output. Always go through PowerShell.

## 4. Monitor

**Use `scripts/laptop/probe_run.ps1`.** One command, and it separates a healthy run from
orphaned workers that look identical to one:

```bash
scp scripts/laptop/probe_run.ps1 laptop:C:/Users/mlgbr/probe_run.ps1
ssh -n laptop 'powershell -NoProfile -ExecutionPolicy Bypass -File C:\Users\mlgbr\probe_run.ps1 -OutDir "<exp>\outputs"'
```

Prints `VERDICT=HEALTHY | ORPHANED | STALLED | NO_RUN`, non-zero exit on the bad ones. It checks
the **process tree** (is the root python's powershell parent still alive, or did the script die
and leave workers computing into the void?) and **effective cores sampled inside one command**.

> [!warning] Never compute utilisation from wall time inferred between tool calls
> An agent has no reliable clock between calls. Doing that once during EXP-038's pilot produced
> "the run is at 15% and probably broken" when a controlled 60-second sample said 7.34 effective
> cores, i.e. 73%, matching the documented 74.2%.

> [!note] Client-side exit 124 or 255 says nothing about the job - verified 2026-08-07
> EXP-038's dispatching ssh hit its own 50-minute client timeout (exit 124). The whole process
> tree survived, launcher powershell included. `ssh -n` plus no `Start-Process` is what makes
> that hold. Probe before reacting.

### Manual probe (the older recipe)

Record-file count is the real progress signal, not the log. Fighting shell quoting on every poll is a
waste, so put a probe script on the laptop once:

```powershell
# scp this to C:\Users\mlgbr\probe.ps1, then run it each poll
$ErrorActionPreference = 'SilentlyContinue'
$d = 'C:\Users\mlgbr\Desktop\Projects\neuromorphic-nn-snn-research-project\experiments\NNN_x\outputs'
$n = @(Get-ChildItem -Path $d -Filter '*.json').Count
$p = @(Get-Process | Where-Object { $_.Name -match '^python' }).Count
$free = [math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB, 1)
$bad = 0
$log = Join-Path $d 'run.log'
if (Test-Path $log) { $bad = @(Select-String -Path $log -Pattern 'Traceback|MemoryError|Killed|BrokenProcessPool').Count }
"$n $p $free $bad"
```

```bash
scp probe.ps1 laptop:C:/Users/mlgbr/probe.ps1
ssh -n laptop 'powershell -NoProfile -ExecutionPolicy Bypass -File C:\Users\mlgbr\probe.ps1'
# -> "64 10 12.4 0"  = 64 records, 10 python procs, 12.4 GB free, 0 error markers
```

Healthy looks like: record count climbing, `procs` equal to `workers + 2`, error count 0.

## 5. Retrieve

```bash
scp "laptop:C:/Users/mlgbr/Desktop/Projects/neuromorphic-nn-snn-research-project/experiments/NNN_x/outputs/*.json" ./local_dir/
```

Note the **forward slashes on the remote side of `scp`** even though it is a Windows host.

## 6. Gotchas, all of these cost real time to rediscover

**Worker processes are named `python3.13.exe`, not `python.exe`.** `Get-Process python` misses them
entirely. Match on `^python`.

**A parent process at ~0% CPU is normal.** It only waits on workers. Health is worker count against
outstanding tasks, plus the record count climbing.

**An SSH drop does not kill the run.** Windows has no SIGHUP semantics, so the remote process keeps
going after the client disconnects. `exit code 255` means the client lost the connection, not that the
job died. Reconnect and probe before concluding anything. Because of this, `Tee-Object` to a log file
matters: the file survives even when the stdout pipe does not.

**Free memory declining is usually not a leak.** Windows `FreePhysicalMemory` excludes standby and
cache, so it falls steadily during a run that writes many files and recovers when the run ends. It fell
from 14 GB to 4.6 GB during EXP-030 while the workers held a flat 1.58 GB the whole time. Check
per-process `WorkingSet64` before you panic:

```bash
ssh -n laptop 'powershell -NoProfile -Command "Get-Process | Where-Object { $_.Name -match \"^python\" } | ForEach-Object { $_.Name + \" ws_mb=\" + [math]::Round($_.WorkingSet64/1MB,0) }"'
```

**Quoting through `cmd.exe` is the main source of wasted cycles.** Three specific traps:

- A trailing backslash before a closing quote gets eaten, so `$d + "file.log"` silently becomes
  `...outputsfile.log`. Use `Join-Path`, or pass `-Path $d -Filter name` instead of concatenating.
- A `|` inside a double-quoted PowerShell string is interpreted by `cmd.exe` as a pipe before
  PowerShell ever sees it. Regex alternations like `Traceback|Error` break. Put them in a `.ps1` file.
- Bash single quotes wrapping `powershell -Command "..."` means every inner quote must be `\"`.

The reliable escape hatch for anything non-trivial is: write a `.ps1`, `scp` it over, run it with
`powershell -NoProfile -ExecutionPolicy Bypass -File`.

**Seeded runs are byte-identical across worker scheduling.** EXP-030's 36 concept records were produced
twice by independent invocations at different worker counts and matched byte for byte. That is a free
correctness check: if a re-run diverges, the seeding discipline is broken, not the scheduler.

## 7. If the laptop is unreachable

In rough order of what to try:

1. `tailscale status` on the VPS. Is the laptop listed and online?
2. Is the laptop awake? Check whether it has suspended; sleep is the most common cause.
3. `tailscale ping swizzlesduo` from the VPS to distinguish a routing problem from an SSH problem.
4. Fall back to the LAN address only if the session is physically on the home network.

Do not silently drop to `StrictHostKeyChecking=no` or force-add keys to get past an error. A host key
mismatch on a tailnet address deserves a look, not a workaround.
