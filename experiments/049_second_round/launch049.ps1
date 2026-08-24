# launch049.ps1 - dispatch EXP-049 on SwizzlesDuo. Copy over with scp, run with
#   ssh -n laptop 'powershell -NoProfile -ExecutionPolicy Bypass -File C:\Users\mlgbr\launch049.ps1'
#
# TWO ARMS, STRICTLY SEQUENTIAL, ~15 h. Arm E freezes arm D's OUTPUT, so it cannot start until
# arm D has written all 12 encoders. That is gated on the ARTIFACT, not on an exit code.
#
#   D  fine-tune EXP-047's encoder a SECOND time -> E2, co-adapted head   ~8.5 h
#   E  freeze E2, fresh head                                             ~6.5 h
#
# SIX WORKERS. 12 cells on 6 is two clean waves at the measured 0.115 s/step; 12 on 10 is two
# waves with the second running 8 idle at ~0.16. Per the 2026-08-22 estimator correction.
#
# THE THREE FAILURE MODES FROM launch043.ps1 ARE AVOIDED HERE TOO. Do not "simplify" them back in.
#
# 1. NO Start-Process. Windows OpenSSH tears down the session job object when the ssh connection
#    closes, killing a detached child. The runs must be direct children of this script, and the
#    script must be launched with `ssh -n` so it survives the pipe closing.
#
# 2. NO $ErrorActionPreference = 'Stop' with 2>&1. PowerShell escalates a native command's FIRST
#    stderr line into a terminating error, and reinforce.py emits a UserWarning on episode one.
#
# 3. PLAIN FILE REDIRECTION, never a pipeline. No Tee-Object.
#
# Exit codes are not a health signal (EXP-046 saw a good dispatch report exit 1 twice).

$ErrorActionPreference = 'Continue'

$repo = 'C:\Users\mlgbr\Desktop\Projects\neuromorphic-nn-snn-research-project'
$exp  = Join-Path $repo 'experiments\049_second_round'
$out  = Join-Path $exp 'outputs'
$py   = Join-Path $repo '.venv\Scripts\python.exe'

function Say($msg) {
    Write-Output ("[{0}] {1}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $msg)
}

& powershell -NoProfile -ExecutionPolicy Bypass -File C:\Users\mlgbr\sync_repo.ps1
if ($LASTEXITCODE -ne 0) {
    Say "ABORT: sync_repo.ps1 exited $LASTEXITCODE. Not starting a 15-hour run on an unknown tree."
    exit 1
}

Set-Location $repo
New-Item -ItemType Directory -Force $out | Out-Null

# Round 2 starts from round 1's tracked encoders. A stale checkout would not have them.
$e1 = @(Get-ChildItem "$repo\experiments\047_encoder_finetuning\outputs\exp047_ft_d6_lr0.0001_*_encoder.pt" -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -match '_s([0-9]|1[01])_sig' })
Say "EXP-047 round-1 encoders present: $($e1.Count) (expected 12)"
if ($e1.Count -lt 12) {
    Say "ABORT: round 2 has nothing to start from."
    exit 1
}

# ---------------- arm D: the second fine-tuning round ----------------
Say "ARM D: round-2 fine-tuning, 12 cells, 6 workers, ~8.5 h"
& $py -u "$exp\run.py" --arm D --workers 6 --skip-existing >> "$out\armD.log" 2>&1

$e2 = @(Get-ChildItem "$out\exp049_ft2_d6_*_encoder.pt" -ErrorAction SilentlyContinue)
Say "ARM D done: $($e2.Count) E2 encoder(s) written (expected 12)."
if ($e2.Count -lt 12) {
    Say "ABORT: arm E freezes arm D's output and cannot run without all 12 encoders."
    Say "       Re-run this script; --skip-existing resumes arm D from what is on disk."
    exit 1
}

# ---------------- arm E: freeze E2, fresh head ----------------
Say "ARM E: E2 frozen + fresh head, 12 cells, 6 workers, ~6.5 h"
& $py -u "$exp\run.py" --arm E --workers 6 --skip-existing >> "$out\armE.log" 2>&1

$recs = @(Get-ChildItem "$out\exp049_fresh2_d6_*.json" -ErrorAction SilentlyContinue)
Say "ARM E done: $($recs.Count) record(s) (expected 12)."
Say "CHAIN COMPLETE."
Say "  aggregate with: .venv\Scripts\python.exe experiments\049_second_round\aggregate.py"
Say "  reminder: Claim 1 is PREDICTED TO REFUTE. Claim 2 carries the answer."
