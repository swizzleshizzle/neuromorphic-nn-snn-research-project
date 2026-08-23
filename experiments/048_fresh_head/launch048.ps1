# launch048.ps1 - dispatch EXP-048 on SwizzlesDuo. Copy over with scp, run with
#   ssh -n laptop 'powershell -NoProfile -ExecutionPolicy Bypass -File C:\Users\mlgbr\launch048.ps1'
#
# ONE ARM, 12 cells, ~6.4 h. Arms A (EXP-043) and C (EXP-047) already exist and are NOT re-run.
#
# SIX WORKERS, NOT TEN, and this is the first run to apply the 2026-08-22 estimator correction.
# 12 cells on 10 workers is TWO waves with the second running 8 workers idle (8.8 h); 12 on 6 is
# two CLEAN waves, and 6 workers measured 0.115 s/step against 10 workers' ~0.16, so it is both
# faster and better packed. Prefer a worker count that divides the cell count.
#
# THE THREE FAILURE MODES FROM launch043.ps1 ARE AVOIDED HERE TOO. Do not "simplify" them back in.
#
# 1. NO Start-Process. Windows OpenSSH tears down the session job object when the ssh connection
#    closes, which kills a detached child. The run must be a direct child of this script, and the
#    script must be launched with `ssh -n` so it survives the pipe closing.
#
# 2. NO $ErrorActionPreference = 'Stop' with 2>&1. PowerShell escalates a native command's FIRST
#    stderr line into a terminating error, and reinforce.py emits a UserWarning on episode one.
#
# 3. PLAIN FILE REDIRECTION, never a pipeline. No Tee-Object - orphaned workers inherit the
#    stdout handle and block a Tee reader forever, which reads exactly like a hang.
#
# Exit codes are NOT a health signal here (EXP-046 saw a good dispatch report exit 1 twice).
# Gate on the artifact: 12 records matching exp048_freshhead_d6_*.json.

$ErrorActionPreference = 'Continue'

$repo = 'C:\Users\mlgbr\Desktop\Projects\neuromorphic-nn-snn-research-project'
$exp  = Join-Path $repo 'experiments\048_fresh_head'
$out  = Join-Path $exp 'outputs'
$py   = Join-Path $repo '.venv\Scripts\python.exe'

function Say($msg) {
    Write-Output ("[{0}] {1}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $msg)
}

# THE SYNC IS NOT OPTIONAL. EXP-048 reads EXP-047's fine-tuned encoders, which are TRACKED in
# git - a stale checkout would not have them at all, and the driver would refuse (correctly).
& powershell -NoProfile -ExecutionPolicy Bypass -File C:\Users\mlgbr\sync_repo.ps1
if ($LASTEXITCODE -ne 0) {
    Say "ABORT: sync_repo.ps1 exited $LASTEXITCODE. Not starting on an unknown tree."
    exit 1
}

Set-Location $repo
New-Item -ItemType Directory -Force $out | Out-Null

# THE ARTIFACT CHECK. `SYNCED` proves the commit; it does not prove the encoders arrived.
$enc = @(Get-ChildItem "$repo\experiments\047_encoder_finetuning\outputs\exp047_ft_d6_lr0.0001_*_encoder.pt" -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -match '_s([0-9]|1[01])_sig' })
Say "EXP-047 confirmatory encoders present: $($enc.Count) (expected 12)"
if ($enc.Count -lt 12) {
    Say "ABORT: EXP-048 has nothing to test without EXP-047's fine-tuned encoders."
    exit 1
}

Say "PHASE 1: EXP-048 arm B, 12 cells, 6 workers, ~6.4 h"
& $py -u "$exp\run.py" --workers 6 --skip-existing >> "$out\run.log" 2>&1

$recs = @(Get-ChildItem "$out\exp048_freshhead_d6_*.json" -ErrorAction SilentlyContinue)
Say "done. $($recs.Count) record(s) (expected 12)."
Say "  aggregate with: .venv\Scripts\python.exe experiments\048_fresh_head\aggregate.py"
