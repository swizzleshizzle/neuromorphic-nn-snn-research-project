# launch051.ps1 - dispatch EXP-051 on SwizzlesDuo. ONE ARM, 12 cells, ~7.2 h.
#   ssh -n laptop 'powershell -NoProfile -ExecutionPolicy Bypass -File C:\Users\mlgbr\launch051.ps1'
#
# DO NOT RUN THIS WHILE EXP-050 IS STILL GOING. It refuses if python is already busy.
# Six workers: two clean waves at the measured 0.115 s/step.

$ErrorActionPreference = 'Continue'
$repo = 'C:\Users\mlgbr\Desktop\Projects\neuromorphic-nn-snn-research-project'
$exp  = Join-Path $repo 'experiments\051_depth7_transfer'
$out  = Join-Path $exp 'outputs'
$py   = Join-Path $repo '.venv\Scripts\python.exe'
function Say($m) { Write-Output ("[{0}] {1}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $m) }

$busy = @(Get-Process | Where-Object { $_.Name -match "^python" }).Count
if ($busy -gt 2) {
    Say "ABORT: $busy python processes already running. EXP-050 is probably still going;"
    Say "       two runs sharing 6 workers each would slow both and blur every timing figure."
    exit 1
}

& powershell -NoProfile -ExecutionPolicy Bypass -File C:\Users\mlgbr\sync_repo.ps1
if ($LASTEXITCODE -ne 0) { Say "ABORT: sync_repo.ps1 exited $LASTEXITCODE."; exit 1 }

Set-Location $repo
New-Item -ItemType Directory -Force $out | Out-Null

Say "EXP-051: E1 frozen + fresh head at DEPTH 7, 12 cells, 6 workers, ~7.2 h"
& $py -u "$exp\run.py" --workers 6 --skip-existing >> "$out\run.log" 2>&1

$recs = @(Get-ChildItem "$out\exp051_transfer_d7_*.json" -ErrorAction SilentlyContinue)
Say "done: $($recs.Count) record(s) (expected 12)."
Say "  aggregate with: .venv\Scripts\python.exe experiments\051_depth7_transfer\aggregate.py"
