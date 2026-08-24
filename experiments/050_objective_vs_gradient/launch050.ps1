# launch050.ps1 - dispatch EXP-050 on SwizzlesDuo. Copy over with scp, run with
#   ssh -n laptop 'powershell -NoProfile -ExecutionPolicy Bypass -File C:\Users\mlgbr\launch050.ps1'
#
# TWO PHASES, ~6 h. Phase 2 freezes phase 1's output and is gated on the ARTIFACT, not exit codes.
#
#   1  40 more inverse-model epochs from E0, warm-started -> E0+   10 workers  ~1.7 h
#   2  12 RL cells, E0+ frozen + fresh head                         6 workers  ~3.9 h
#
# WORKER COUNTS ARE FROM MEASUREMENT, NOT GUESSWORK. EXP-040 did 12 encoders at 10 workers in 100
# min and this session's phase 0 did 12 at TWELVE workers in 132 min, so 10 wins for pretraining.
# EXP-049's arm E did 12 frozen RL cells at 6 workers in 3.9 h - two clean waves, which beats 10
# workers' ragged two.
#
# The three launch043.ps1 failure modes are avoided here too: no Start-Process, no
# ErrorActionPreference='Stop' with 2>&1, plain file redirection and never a pipeline.

$ErrorActionPreference = 'Continue'

$repo = 'C:\Users\mlgbr\Desktop\Projects\neuromorphic-nn-snn-research-project'
$exp  = Join-Path $repo 'experiments\050_objective_vs_gradient'
$out  = Join-Path $exp 'outputs'
$py   = Join-Path $repo '.venv\Scripts\python.exe'

function Say($msg) {
    Write-Output ("[{0}] {1}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $msg)
}

& powershell -NoProfile -ExecutionPolicy Bypass -File C:\Users\mlgbr\sync_repo.ps1
if ($LASTEXITCODE -ne 0) {
    Say "ABORT: sync_repo.ps1 exited $LASTEXITCODE."
    exit 1
}

Set-Location $repo
New-Item -ItemType Directory -Force $out | Out-Null

Say "PHASE 1: 40 more inverse-model epochs from E0, warm-started, 10 workers"
& $py -u "$exp\extend_pretrain.py" --workers 10 >> "$out\phase1_extend.log" 2>&1

$enc = @(Get-ChildItem "$out\exp050_encoder_plus_s*.pt" -ErrorAction SilentlyContinue)
Say "PHASE 1 done: $($enc.Count) E0+ encoder(s) (expected 12)."
if ($enc.Count -lt 12) {
    Say "ABORT: phase 2 freezes phase 1's output and cannot run without all 12."
    exit 1
}

Say "PHASE 2: arm F, E0+ frozen + fresh head, 12 cells, 6 workers"
& $py -u "$exp\run.py" --workers 6 --skip-existing >> "$out\phase2_armF.log" 2>&1

$recs = @(Get-ChildItem "$out\exp050_pre2_d6_*.json" -ErrorAction SilentlyContinue)
Say "PHASE 2 done: $($recs.Count) record(s) (expected 12)."
Say "CHAIN COMPLETE. aggregate with:"
Say "  .venv\Scripts\python.exe experiments\050_objective_vs_gradient\aggregate.py"
