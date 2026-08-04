# launch036.ps1 - dispatch EXP-036 on SwizzlesDuo. Copy over with scp, run with
#   ssh -n laptop 'powershell -NoProfile -ExecutionPolicy Bypass -File C:\Users\mlgbr\launch036.ps1'
#
# THREE FAILURE MODES ARE DELIBERATELY AVOIDED HERE. All three were paid for in week 17 and
# are documented in launch3.ps1 / launch032.ps1. Do not "simplify" any of them back in.
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

$ErrorActionPreference = 'Continue'

$repo = 'C:\Users\mlgbr\Desktop\Projects\neuromorphic-nn-snn-research-project'
$exp  = Join-Path $repo 'experiments\036_generalisation_gap'
$out  = Join-Path $exp 'outputs'
$log  = Join-Path $out 'run.log'

Set-Location $repo

git fetch --all --prune
git checkout main
git pull --ff-only origin main
$head = (git log --oneline -1)
Write-Output "repo at: $head"

New-Item -ItemType Directory -Force $out | Out-Null

$stamp = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
Write-Output "EXP-036 starting $stamp"
Write-Output "expect about 11.8 h wall on 16 workers (190 core-hours at the measured 153 ms/step)"
Write-Output "log: $log"

.venv\Scripts\python.exe -u experiments\036_generalisation_gap\run.py `
    --seeds 0 1 2 3 4 5 6 7 8 9 10 11 --workers 16 > $log 2>&1

$code = $LASTEXITCODE
$done = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
Write-Output "EXP-036 finished $done, exit $code"
exit $code
