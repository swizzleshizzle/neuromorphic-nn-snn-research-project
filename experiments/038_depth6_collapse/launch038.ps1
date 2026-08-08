# launch038.ps1 - dispatch EXP-038 on SwizzlesDuo. Copy over with scp, run with
#   ssh -n laptop 'powershell -NoProfile -ExecutionPolicy Bypass -File C:\Users\mlgbr\launch038.ps1'
#
# THREE FAILURE MODES ARE DELIBERATELY AVOIDED HERE. All three were paid for in week 17 and
# are documented in launch3.ps1 / launch032.ps1 / launch036.ps1 / launch037.ps1. Do not
# "simplify" them back in. All three were re-introduced by accident during the EXP-038 design
# session and caught before they cost the run, which is why they are restated rather than
# assumed known.
#
# 1. NO Start-Process. Windows OpenSSH tears down the session job object when the ssh
#    connection closes, which kills a detached child. The run must be a direct child of this
#    script, and the script must be launched with `ssh -n` so it survives the pipe closing.
#
# 2. NO $ErrorActionPreference = 'Stop' combined with 2>&1. PowerShell escalates a native
#    command's FIRST stderr line into a terminating error, and reinforce.py:188 emits a
#    UserWarning on the very first episode. That combination killed a run and orphaned 16
#    workers that went on looking healthy for 35 minutes.
#
# 3. PLAIN FILE REDIRECTION, never a pipeline. No Tee-Object. Orphaned workers inherit the
#    stdout handle and block a Tee-Object reader forever, so the log stops updating while the
#    run is still alive - which reads exactly like a hang.
#
# `python -u` because a buffered log tells you nothing until it is too late to act on.
#
# TEN WORKERS, NOT SIXTEEN. Measured: 10 gave 74.2% utilisation against 16 at 43.1%, because
# the laptop is MEMORY-bound at ~920 MB private per worker, not core-bound despite 22 logical
# cores. Confirmed again by the EXP-038 pilot at 6.8-7.3 effective cores.

$ErrorActionPreference = 'Continue'

$repo = 'C:\Users\mlgbr\Desktop\Projects\neuromorphic-nn-snn-research-project'
$exp  = Join-Path $repo 'experiments\038_depth6_collapse'
$out  = Join-Path $exp 'outputs'
$log  = Join-Path $out 'run.log'

# THE SYNC IS NOT OPTIONAL AND ITS EXIT CODE IS NOT ADVISORY. A bare `git pull` here has
# silently failed twice - 12 commits behind before EXP-037, 48 colliding head checkpoints
# before EXP-038 - and each time the dispatch would have run against stale source.
& powershell -NoProfile -ExecutionPolicy Bypass -File C:\Users\mlgbr\sync_repo.ps1
if ($LASTEXITCODE -ne 0) {
    Write-Output "ABORT: sync_repo.ps1 exited $LASTEXITCODE. Not starting a 21-hour run on an"
    Write-Output "unverified checkout."
    exit 1
}

Set-Location $repo
Write-Output "repo at: $(git log --oneline -1)"

# Verify the SYMBOL, not just the commit. `SYNCED` proves the checkout moved; it does not
# prove the code does what this run needs. B_TOP is new in EXP-038's driver.
$symbol = (Select-String -Path (Join-Path $exp 'run.py') -Pattern 'B_TOP').Count
Write-Output "B_TOP occurrences in run.py: $symbol"
if ($symbol -lt 1) {
    Write-Output "ABORT: the driver on disk does not contain B_TOP. Wrong commit."
    exit 1
}

New-Item -ItemType Directory -Force $out | Out-Null

Write-Output "EXP-038 starting $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Output "48 runs, 10 workers, about 4.68M env steps, estimated ~21 h"
Write-Output "log: $log"

.venv\Scripts\python.exe -u experiments\038_depth6_collapse\run.py `
    --seeds 0 1 2 3 4 5 6 7 8 9 10 11 --workers 10 > $log 2>&1

$code = $LASTEXITCODE
Write-Output "EXP-038 finished $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), exit $code"
exit $code
