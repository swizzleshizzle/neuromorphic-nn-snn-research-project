# launch037.ps1 - dispatch EXP-037 on SwizzlesDuo. Copy over with scp, run with
#   ssh -n laptop 'powershell -NoProfile -ExecutionPolicy Bypass -File C:\Users\mlgbr\launch037.ps1'
#
# THREE FAILURE MODES ARE DELIBERATELY AVOIDED HERE. All three were paid for in week 17 and
# are documented in launch3.ps1 / launch032.ps1 / launch036.ps1. Do not "simplify" them back in.
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
# TEN WORKERS, NOT SIXTEEN. EXP-036 ran 16 at 920 MB private each, drove system commit to
# 48.6 of 50.4 GB, and held utilisation at a measured 43.1%: the workers spent most of their
# time paging rather than computing. Ten is 9.2 GB instead of 14.7 GB. Effective throughput
# should be roughly a wash (10 x ~70% vs 16 x 43%) while leaving the laptop usable. That is a
# PREDICTION - check utilisation early against 43.1% and record which way it went.

$ErrorActionPreference = 'Continue'

$repo = 'C:\Users\mlgbr\Desktop\Projects\neuromorphic-nn-snn-research-project'
$exp  = Join-Path $repo 'experiments\037_curriculum_weighting'
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
Write-Output "EXP-037 starting $stamp"
Write-Output "48 runs, 10 workers, about 4.62M env steps"
Write-Output "log: $log"

.venv\Scripts\python.exe -u experiments\037_curriculum_weighting\run.py `
    --seeds 0 1 2 3 4 5 6 7 8 9 10 11 --workers 10 > $log 2>&1

$code = $LASTEXITCODE
$done = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
Write-Output "EXP-037 finished $done, exit $code"
exit $code
