# EXP-057 launcher for SwizzlesDuo, running from the WORKTREE at C:\Users\mlgbr\wt-exp053.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File C:\Users\mlgbr\launch057_wt.ps1 -Phase check
#   powershell -NoProfile -ExecutionPolicy Bypass -File C:\Users\mlgbr\launch057_wt.ps1 -Phase rl -Workers 6
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
$hasFlag = & $py -c "from neuromorphic.training.cube_baseline import CubeConfig; import sys; sys.stdout.write(str(hasattr(CubeConfig, '__dataclass_fields__') and 'constant_critic' in CubeConfig.__dataclass_fields__))"
if ($hasFlag -ne "True") {
    Write-Error "CubeConfig has no constant_critic field. The worktree is stale; this would silently re-run arm B under EXP-057's tag."
    exit 1
}

$head   = (& git rev-parse --short HEAD).Trim()
$branch = (& git rev-parse --abbrev-ref HEAD).Trim()
"library      = $where"
"worktree     = $branch @ $head"
"constant_critic present = $hasFlag"

# Worker processes appear as python3.13.exe, not python.exe, so match on ^python.
$alive = @(Get-Process | Where-Object { $_.ProcessName -match "^python" }).Count
if ($alive -gt 2) { Write-Error "$alive python processes already running; refusing to start."; exit 1 }
"pyprocs      = $alive"

if ($Phase -eq "check") { "CHECK OK"; exit 0 }

$outDir = Join-Path $wt "experiments\057_constant_critic\outputs"
New-Item -ItemType Directory -Force $outDir | Out-Null

# 12 cells on 6 workers is 2 clean waves. EXP-055 measured that more workers buy no wall clock
# on RL cells, so this is chosen for memory headroom: RL workers run about 1.05 GB private each.
if ($Workers -eq 0) { $Workers = 6 }
$script  = "experiments\057_constant_critic\run.py"
$cliArgs = @("--workers", $Workers)
if ($SkipExisting) { $cliArgs += "--skip-existing" }
$log = Join-Path $wt "experiments\057_constant_critic\phase_rl.log"

"script       = $script $cliArgs"
"log          = $log"
""

# -u so the log is not fully buffered. Tee-Object so the record survives an ssh drop: Windows has
# no SIGHUP semantics. Do NOT wrap in Start-Process -WindowStyle Hidden over ssh; that dies with
# the session. Run in the ssh FOREGROUND and background the call on the controller side.
& $py -u $script @cliArgs 2>&1 | Tee-Object -FilePath $log
