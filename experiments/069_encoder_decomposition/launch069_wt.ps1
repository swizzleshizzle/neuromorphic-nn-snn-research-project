# EXP-069 launcher for SwizzlesDuo, from the WORKTREE at C:\Users\mlgbr\wt-exp053.
#
#   ... -File C:\Users\mlgbr\launch069_wt.ps1 -Phase check
#   ... -File C:\Users\mlgbr\launch069_wt.ps1 -Phase all -Workers 8
#
# -Phase all runs pretrain THEN rl, both with --skip-existing, so it can run unattended and a
# re-launch after any interruption resumes where it stopped. 64 cells at 8 workers = 8 clean waves.
#
# Same traps as every launcher here: the worktree has no .venv and the main checkout's interpreter
# imports the MAIN checkout's library unless PYTHONPATH wins, which is PROVEN below. No backticks
# in double-quoted strings. No comma-separated arguments. Verify a launch by records and processes,
# never by the ssh exit code.

param(
    [Parameter(Mandatory=$true)][ValidateSet("check","all")][string]$Phase,
    [int]$Workers = 8
)

$wt  = "C:\Users\mlgbr\wt-exp053"
$py  = "C:\Users\mlgbr\Desktop\Projects\neuromorphic-nn-snn-research-project\.venv\Scripts\python.exe"
$src = Join-Path $wt "src"
if (-not (Test-Path $wt)) { Write-Error "worktree missing: $wt"; exit 1 }
if (-not (Test-Path $py)) { Write-Error "interpreter missing: $py"; exit 1 }
Set-Location $wt
$env:PYTHONPATH = $src

$where = & $py -c "import neuromorphic, sys; sys.stdout.write(neuromorphic.__file__)"
if (-not $where.StartsWith($src, [System.StringComparison]::OrdinalIgnoreCase)) {
    Write-Error "WRONG LIBRARY: $where. PYTHONPATH did not win; refusing to run."; exit 1
}

# All three seed roles must resolve independently; EXP-069 crosses encoder_seed against the rest.
$seeds = & $py -c "import sys; from neuromorphic.training.cube_baseline import CubeConfig, resolve_seed; c = CubeConfig(arm='regionalized', depth=5, seed=1, encoder_seed=3, split_seed=6, train_seed=6); sys.stdout.write(str(resolve_seed(c,'encoder') == 3 and resolve_seed(c,'split') == 6 and resolve_seed(c,'train') == 6))"
if ($seeds -ne "True") { Write-Error "seed roles do not resolve independently; stale worktree."; exit 1 }

# The collision trap AND the leak guard, through the real driver on this machine.
$grid = & $py -c "import importlib.util, pathlib, sys; s = importlib.util.spec_from_file_location('e069', 'experiments/069_encoder_decomposition/run.py'); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); from neuromorphic.training.cube_baseline import record_filename; o = pathlib.Path('experiments/069_encoder_decomposition/outputs'); c = m.sweep_configs(o, o); leak = sum(1 for x in c if 'exp040_encoder' in str(x.encoder_state_path)); sys.stdout.write(str(len(c)) + ':' + str(len(set(record_filename(x) for x in c))) + ':' + str(leak))"
if ($grid -ne "64:64:0") { Write-Error "grid check gave $grid, expected 64:64:0 (cells:distinct:leaking)."; exit 1 }

$head = (& git rev-parse --short HEAD).Trim()
"library   = $where"
"worktree  = " + (& git rev-parse --abbrev-ref HEAD).Trim() + " @ $head"
"seeds     = $seeds"
"grid      = $grid  (cells:distinct:leaking)"
$alive = @(Get-Process | Where-Object { $_.ProcessName -match "^python" }).Count
if ($alive -gt 2) { Write-Error "$alive python processes already running; refusing to start."; exit 1 }
"pyprocs   = $alive"
if ($Phase -eq "check") { "CHECK OK"; exit 0 }

$base  = Join-Path $wt "experiments\069_encoder_decomposition"
$stamp = Join-Path $base "run.stamp"
$log   = Join-Path $base "run.log"
$script = "experiments\069_encoder_decomposition\run.py"
"started  = " + (Get-Date -Format o) + "  head $head" | Set-Content -Path $stamp

& $py -u $script --phase pretrain --workers $Workers --skip-existing 2>&1 | Tee-Object -FilePath $log
"pretrain_done = " + (Get-Date -Format o) | Add-Content -Path $stamp
& $py -u $script --phase rl --workers $Workers --skip-existing 2>&1 | Tee-Object -FilePath $log -Append
"finished = " + (Get-Date -Format o) | Add-Content -Path $stamp
