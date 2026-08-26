# launch052.ps1 - dispatch EXP-052 on SwizzlesDuo. TWO PHASES, ~9.7 h.
#   ssh -n laptop 'powershell -NoProfile -ExecutionPolicy Bypass -File C:\Users\mlgbr\launch052.ps1'
#
#   1  pretrain 12x10 and 12x20 epochs FROM SCRATCH   10 workers  ~1.9 h
#   2  24 RL cells, each frozen + fresh head           6 workers  ~7.8 h
#
# Phase 2 is gated on all 24 encoders existing, not on an exit code. Worker counts from
# measurement: EXP-050 phase 1 (10 workers) and EXP-049 arm E (6 workers).

$ErrorActionPreference = 'Continue'
$repo = 'C:\Users\mlgbr\Desktop\Projects\neuromorphic-nn-snn-research-project'
$exp  = Join-Path $repo 'experiments\052_pretraining_optimum'
$out  = Join-Path $exp 'outputs'
$py   = Join-Path $repo '.venv\Scripts\python.exe'
function Say($m) { Write-Output ("[{0}] {1}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $m) }

$busy = @(Get-Process | Where-Object { $_.Name -match "^python" }).Count
if ($busy -gt 2) { Say "ABORT: $busy python processes already running."; exit 1 }

& powershell -NoProfile -ExecutionPolicy Bypass -File C:\Users\mlgbr\sync_repo.ps1
if ($LASTEXITCODE -ne 0) { Say "ABORT: sync_repo.ps1 exited $LASTEXITCODE."; exit 1 }

Set-Location $repo
New-Item -ItemType Directory -Force $out | Out-Null

Say "PHASE 1: pretrain 10- and 20-epoch encoders FROM SCRATCH, 10 workers"
& $py -u "$exp\pretrain_sweep.py" --workers 10 >> "$out\phase1_sweep.log" 2>&1

$enc = @(Get-ChildItem "$out\exp052_encoder_e*_s*.pt" -ErrorAction SilentlyContinue)
Say "PHASE 1 done: $($enc.Count) encoder(s) (expected 24)."
if ($enc.Count -lt 24) { Say "ABORT: phase 2 needs all 24."; exit 1 }

Say "PHASE 2: 24 RL cells, frozen + fresh head, 6 workers"
& $py -u "$exp\run.py" --workers 6 --skip-existing >> "$out\phase2_rl.log" 2>&1

$recs = @(Get-ChildItem "$out\exp052_e*_d6_*.json" -ErrorAction SilentlyContinue)
Say "PHASE 2 done: $($recs.Count) record(s) (expected 24)."
Say "  aggregate with: .venv\Scripts\python.exe experiments\052_pretraining_optimum\aggregate.py"
Say "  reminder: PRIMARY is 20-vs-40 only. The 10-epoch arm carries no bar."
