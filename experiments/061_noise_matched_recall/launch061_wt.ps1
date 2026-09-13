# EXP-061 launcher for SwizzlesDuo, running from the WORKTREE at C:\Users\mlgbr\wt-exp053.
#
#   ... -File C:\Users\mlgbr\launch061_wt.ps1 -Phase check
#   ... -File C:\Users\mlgbr\launch061_wt.ps1 -Phase rl -Workers 6 -SkipExisting
#
# ONE arm, 24 cells, 6 workers = 4 clean waves. Six rather than ten deliberately: per-cell time
# is WORSE at 10 workers (0.16 vs 0.115 s/step measured), 24 divides by 6 exactly, and EXP-059
# measured 3.054 h per cell at 6 workers on this same depth and config - so the estimate comes
# from a same-worker-count measurement rather than a cross-worker-count scaling, which has now
# come in low four times running.
#
# THE WORKTREE HAS NO .venv, AND THAT IS A TRAP. The only interpreter belongs to the main
# checkout and installs the project editable with an ABSOLUTE path to the MAIN CHECKOUT's src,
# so a worktree script run with it imports the OLD library and produces a complete, plausible,
# entirely wrong result with no error. PYTHONPATH overrides it and the gate below PROVES it did.
#
# NO BACKTICKS IN DOUBLE-QUOTED OUTPUT STRINGS. PowerShell reads `b as backspace and `f as form
# feed, so "(phase `baseline`)" printed as "(phase aseline)" in EXP-060's launcher.
#
# NO COMMA-SEPARATED ARGUMENTS. cmd.exe eats commas and the failure EXITS ZERO (EXP-055 lost a
# dispatch to -Epochs 1,2,3,5 arriving as 1235). Verify a launch by probing for records and
# worker processes, never by the ssh exit code.

param(
    [Parameter(Mandatory=$true)][ValidateSet("check","rl")][string]$Phase,
    [int]$Workers = 0,
    [switch]$SkipExisting
)

$wt  = "C:\Users\mlgbr\wt-exp053"
$py  = "C:\Users\mlgbr\Desktop\Projects\neuromorphic-nn-snn-research-project\.venv\Scripts\python.exe"
$src = Join-Path $wt "src"

if (-not (Test-Path $wt)) { Write-Error "worktree missing: $wt"; exit 1 }
if (-not (Test-Path $py)) { Write-Error "interpreter missing: $py"; exit 1 }

Set-Location $wt
$env:PYTHONPATH = $src

$where = & $py -c "import neuromorphic, sys; sys.stdout.write(neuromorphic.__file__)"
if ($LASTEXITCODE -ne 0) { Write-Error "could not import neuromorphic"; exit 1 }
if (-not $where.StartsWith($src, [System.StringComparison]::OrdinalIgnoreCase)) {
    Write-Error "WRONG LIBRARY. neuromorphic resolved to $where, expected under $src. PYTHONPATH did not win; refusing to run."
    exit 1
}

# EXP-061 lives or dies on this mode existing in the WORKTREE's library. A stale worktree would
# raise on the unknown readout, which is the loud failure - but the SILENT one to guard against
# is a worktree whose MemoryReadout accepts the name while lacking the substitution, so the
# check below constructs the mode and asserts the noise instrument exists on it.
$hasMode = & $py -c "import random, sys; from neuromorphic.training.cube_baseline import MemoryReadout; ro = MemoryReadout('memory_noise', random.Random(0), None); ro.reset(); sys.stdout.write(str(hasattr(ro, 'noise_cos_n') and ro._noise_gen is not None))"
if ($hasMode -ne "True") {
    Write-Error "MemoryReadout has no working memory_noise mode. The worktree is stale; EXP-061 cannot run."
    exit 1
}

# The validity gate reads this field. Without it the gate would read None and VOID the run.
$hasRatio = & $py -c "import inspect, sys; from neuromorphic.training import cube_baseline as cb; sys.stdout.write(str('recall_concept_norm_ratio' in inspect.getsource(cb)))"
if ($hasRatio -ne "True") {
    Write-Error "no recall_concept_norm_ratio instrument in the worktree library. EXP-061's gate would have no data."
    exit 1
}

# All 24 E0 encoders. NOT tracked in git, so a fresh worktree has none and a synced one has
# only whichever a previous run left behind. Refuse rather than silently run a smaller sweep.
$encDir = Join-Path $wt "experiments\040_pretrained_encoder_policy\outputs"
$e0 = @(Get-ChildItem -Path $encDir -Filter "exp040_encoder_s*.pt" -ErrorAction SilentlyContinue).Count
if ($e0 -lt 24) { Write-Error "found $e0 of 24 E0 encoders in $encDir; copy them from the main checkout."; exit 1 }

$head   = (& git rev-parse --short HEAD).Trim()
$branch = (& git rev-parse --abbrev-ref HEAD).Trim()
"library          = $where"
"worktree         = $branch @ $head"
"memory_noise     = $hasMode"
"norm instrument  = $hasRatio"
"E0 encoders      = $e0 of 24"

$alive = @(Get-Process | Where-Object { $_.ProcessName -match "^python" }).Count
if ($alive -gt 2) { Write-Error "$alive python processes already running; refusing to start."; exit 1 }
"pyprocs          = $alive"

if ($Phase -eq "check") { "CHECK OK"; exit 0 }

if ($Workers -eq 0) { $Workers = 6 }
$outDir = Join-Path $wt "experiments\061_noise_matched_recall\outputs"
New-Item -ItemType Directory -Force $outDir | Out-Null

$script  = "experiments\061_noise_matched_recall\run.py"
$cliArgs = @("--workers", $Workers)
if ($SkipExisting) { $cliArgs += "--skip-existing" }
$log = Join-Path $wt "experiments\061_noise_matched_recall\phase_rl.log"

"script           = $script $cliArgs"
"log              = $log"
""

# -u so the log is not fully buffered. Tee-Object so the record survives an ssh drop: Windows
# has no SIGHUP semantics, and an ssh client dying with "Broken pipe" leaves the job running.
# Do NOT wrap in Start-Process over ssh; that dies with the session.
& $py -u $script @cliArgs 2>&1 | Tee-Object -FilePath $log
