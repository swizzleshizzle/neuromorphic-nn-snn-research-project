# launch040.ps1 - dispatch EXP-040 on SwizzlesDuo. Copy over with scp, run with
#   ssh -n laptop 'powershell -NoProfile -ExecutionPolicy Bypass -File C:\Users\mlgbr\launch040.ps1'
#
# THREE FAILURE MODES ARE DELIBERATELY AVOIDED HERE. All three were paid for in week 17, and
# all three were re-introduced by accident during the EXP-038 design session before being
# caught. Do not "simplify" them back in. Copy this file; do not write a launcher from memory.
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
# TEN WORKERS. Measured 74.2% utilisation against 16 workers' 43.1%; the laptop is MEMORY-bound
# at ~920 MB private per worker, not core-bound despite 22 logical cores. EXP-038 sustained
# 5.8-7.2 effective cores across 22 h at this setting.

$ErrorActionPreference = 'Continue'

$repo = 'C:\Users\mlgbr\Desktop\Projects\neuromorphic-nn-snn-research-project'
$exp  = Join-Path $repo 'experiments\040_pretrained_encoder_policy'
$out  = Join-Path $exp 'outputs'
$log  = Join-Path $out 'run.log'

# THE SYNC IS NOT OPTIONAL AND ITS EXIT CODE IS NOT ADVISORY. A bare `git pull` here has
# silently failed twice, and EXP-038 has just written 48 more untracked *_head.pt files at
# paths the VPS has since committed - so the collision WILL recur on this dispatch.
# sync_repo.ps1 moves exactly those aside into a timestamped attic and verifies the result.
& powershell -NoProfile -ExecutionPolicy Bypass -File C:\Users\mlgbr\sync_repo.ps1
if ($LASTEXITCODE -ne 0) {
    Write-Output "ABORT: sync_repo.ps1 exited $LASTEXITCODE. Not starting a 20-hour run on an"
    Write-Output "unverified checkout."
    exit 1
}

Set-Location $repo
Write-Output "repo at: $(git log --oneline -1)"

# Verify the SYMBOL, not just the commit. `SYNCED` proves the checkout moved; it does not prove
# the code does what this run needs. `encoder_state_path` is the seam EXP-040 depends on, and
# without it every run would silently train on a RANDOM encoder and produce a plausible null.
$symbol = (Select-String -Path (Join-Path $repo 'src\neuromorphic\training\cube_baseline.py') -Pattern 'encoder_state_path').Count
Write-Output "encoder_state_path occurrences in cube_baseline.py: $symbol"
if ($symbol -lt 1) {
    Write-Output "ABORT: the trainer on disk has no encoder_state_path seam. Wrong commit."
    exit 1
}

New-Item -ItemType Directory -Force $out | Out-Null

Write-Output "EXP-040 starting $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Output "phase 1: 12 pretrained encoders (~15 min). phase 2: 36 runs, ~3.24M env steps, ~20 h"
Write-Output "log: $log"

.venv\Scripts\python.exe -u experiments\040_pretrained_encoder_policy\run.py `
    --seeds 0 1 2 3 4 5 6 7 8 9 10 11 --workers 10 --skip-existing > $log 2>&1

$code = $LASTEXITCODE
Write-Output "EXP-040 finished $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), exit $code"
exit $code
