# EXP-058 launcher for SwizzlesDuo, running from the WORKTREE at C:\Users\mlgbr\wt-exp053.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File C:\Users\mlgbr\launch058_wt.ps1 -Phase check
#   powershell -NoProfile -ExecutionPolicy Bypass -File C:\Users\mlgbr\launch058_wt.ps1 -Phase rl -Workers 6
#
# THE WORKTREE HAS NO .venv, AND THAT IS A TRAP. The only interpreter on the machine belongs to
# the main checkout, and it installs the project editable with an ABSOLUTE path to the MAIN
# CHECKOUT's src. Running a worktree script with it imports the OLD library while producing a
# complete, plausible, entirely wrong result: no error, no warning. PYTHONPATH overrides it, and
# the gate below PROVES it did rather than assuming, by resolving `neuromorphic.__file__` and
# refusing to start if it is not under the worktree. Never remove that gate.
#
# NO COMMA-SEPARATED ARGUMENTS. The remote shell over ssh is cmd.exe, which treats commas as
# argument separators; EXP-055 lost a dispatch to `-Epochs 1,2,3,5` arriving as `1235`, and the
# resulting ValidateSet failure EXITED ZERO because PowerShell parameter binding fails before the
# script body runs. Verify a launch by probing for records and worker processes, never by the ssh
# exit code.

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

# EXP-056 lives or dies on this symbol existing in the worktree's library, not just in the repo
# on the controller. A stale worktree would run arm B again under EXP-056's tag and the records
# would be indistinguishable from a real result.
# EXP-058 needs two things present in the WORKTREE, not merely in the repo on the controller:
# the three memory readout modes, and EXP-049's E2 encoders. A stale worktree missing either
# would fail late or, worse, run something that is not this experiment.
$hasFlag = & $py -c "import inspect,sys; from neuromorphic.training.cube_baseline import MemoryReadout as M; src=inspect.getsource(M); sys.stdout.write(str(all(m in src for m in ('memory_shuffled','memory_amnesic'))))"
if ($hasFlag -ne "True") {
    Write-Error "MemoryReadout is not importable from the worktree library. The worktree is stale."
    exit 1
}
$encDir = Join-Path $wt "experiments\049_second_round\outputs"
$encs = @(Get-ChildItem -Path $encDir -Filter "exp049_ft2_d6_*_encoder.pt" -ErrorAction SilentlyContinue).Count
if ($encs -lt 12) {
    Write-Error "found $encs of 12 E2 encoders in $encDir. They are tracked in git; the worktree is stale or incomplete."
    exit 1
}
"E2 encoders  = $encs of 12"

$head   = (& git rev-parse --short HEAD).Trim()
$branch = (& git rev-parse --abbrev-ref HEAD).Trim()
"library      = $where"
"worktree     = $branch @ $head"
"MemoryReadout   = $hasFlag"

# Worker processes appear as python3.13.exe, not python.exe, so match on ^python.
$alive = @(Get-Process | Where-Object { $_.ProcessName -match "^python" }).Count
if ($alive -gt 2) { Write-Error "$alive python processes already running; refusing to start."; exit 1 }
"pyprocs      = $alive"

if ($Phase -eq "check") { "CHECK OK"; exit 0 }

$outDir = Join-Path $wt "experiments\058_memory_reask\outputs"
New-Item -ItemType Directory -Force $outDir | Out-Null

# 36 cells on 6 workers is 6 clean waves. Worker count is chosen from memory headroom, not wall
# clock: EXP-055 measured that more workers buy nothing on RL cells. NOTE the per-cell cost of a
# memory arm is NOT known - read wave 1's recorded `seconds` before assuming the total.
if ($Workers -eq 0) { $Workers = 6 }
$script  = "experiments\058_memory_reask\run.py"
$cliArgs = @("--workers", $Workers)
if ($SkipExisting) { $cliArgs += "--skip-existing" }
$log = Join-Path $wt "experiments\058_memory_reask\phase_rl.log"

"script       = $script $cliArgs"
"log          = $log"
""

# -u so the log is not fully buffered. Tee-Object so the record survives an ssh drop: Windows has
# no SIGHUP semantics. Do NOT wrap in Start-Process -WindowStyle Hidden over ssh; that dies with
# the session. Run in the ssh FOREGROUND and background the call on the controller side.
& $py -u $script @cliArgs 2>&1 | Tee-Object -FilePath $log
