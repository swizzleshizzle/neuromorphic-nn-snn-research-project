# EXP-068 launcher for SwizzlesDuo, running from the WORKTREE at C:\Users\mlgbr\wt-exp053.
#
#   ... -File C:\Users\mlgbr\launch068_wt.ps1 -Phase check
#   ... -File C:\Users\mlgbr\launch068_wt.ps1 -Phase calibrate -Workers 10
#   ... -File C:\Users\mlgbr\launch068_wt.ps1 -Phase full -Workers 10 -SkipExisting
#
# 100 cells, 10 workers = 10 clean waves. Worker count DIVIDES the cell count deliberately: a
# partial final wave is the part left exposed to an interruption, and the two runs lost to a
# restart differed 19 CPU-hours against 4 purely on how much had banked. Every cell writes its own
# record and -SkipExisting is always passed, so an interruption costs one wave at most.
#
# THE WORKTREE HAS NO .venv, AND THAT IS A TRAP. The only interpreter belongs to the main checkout
# and installs the project editable with an ABSOLUTE path to the MAIN CHECKOUT's src, so a
# worktree script run with it imports the OLD library and produces a complete, plausible, entirely
# wrong result with no error. PYTHONPATH overrides it and the gate below PROVES it did.
#
# NO BACKTICKS IN DOUBLE-QUOTED OUTPUT STRINGS. PowerShell reads `b as backspace and `f as form
# feed, so "(phase `baseline`)" printed as "(phase aseline)" in EXP-060's launcher.
#
# NO COMMA-SEPARATED ARGUMENTS. cmd.exe eats commas and the failure EXITS ZERO (EXP-055 lost a
# dispatch to -Epochs 1,2,3,5 arriving as 1235). Verify a launch by probing for records and
# worker processes, never by the ssh exit code.

param(
    [Parameter(Mandatory=$true)][ValidateSet("check","calibrate","full")][string]$Phase,
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

# EXP-068 lives or dies on these two fields existing and being SEPARATELY resolvable. A worktree
# whose CubeConfig lacks them would raise; the silent failure to guard against is one where the
# fields exist but resolve_seed ignores them, so this asserts the resolution itself.
$seedFields = & $py -c "import sys; from dataclasses import fields; from neuromorphic.training.cube_baseline import CubeConfig, resolve_seed; c = CubeConfig(arm='regionalized', depth=3, seed=1, split_seed=4, train_seed=9); ok = {'split_seed','train_seed'} <= {f.name for f in fields(CubeConfig)} and resolve_seed(c,'split') == 4 and resolve_seed(c,'train') == 9; sys.stdout.write(str(ok))"
if ($seedFields -ne "True") {
    Write-Error "CubeConfig does not resolve split_seed and train_seed independently. The worktree is stale; EXP-068 cannot run."
    exit 1
}

# THE COLLISION TRAP, TESTED ON THE MACHINE THAT WILL RUN IT. record_filename encodes neither
# seed, so 100 cells could land in one file. Its own docstring names a seed-decomposition sweep
# as the case. This builds the real grid through the real driver and counts distinct names.
$grid = & $py -c "import importlib.util, pathlib, sys; s = importlib.util.spec_from_file_location('e068', 'experiments/068_seed_decomposition/run.py'); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); from neuromorphic.training.cube_baseline import record_filename; c = m.sweep_configs(pathlib.Path('.')); n = len(set(record_filename(x) for x in c)); sys.stdout.write(str(len(c)) + ':' + str(n))"
if ($grid -ne "100:100") {
    Write-Error "grid built $grid, expected 100:100 cells:distinct-filenames. Refusing to run a sweep that would overwrite its own records."
    exit 1
}

$head   = (& git rev-parse --short HEAD).Trim()
$branch = (& git rev-parse --abbrev-ref HEAD).Trim()
"library          = $where"
"worktree         = $branch @ $head"
"seed resolution  = $seedFields"
"grid:distinct    = $grid"

$alive = @(Get-Process | Where-Object { $_.ProcessName -match "^python" }).Count
if ($alive -gt 2) { Write-Error "$alive python processes already running; refusing to start."; exit 1 }
"pyprocs          = $alive"

if ($Phase -eq "check") { "CHECK OK"; exit 0 }

if ($Workers -eq 0) { $Workers = 10 }
$outDir = Join-Path $wt "experiments\068_seed_decomposition\outputs"
New-Item -ItemType Directory -Force $outDir | Out-Null

$script  = "experiments\068_seed_decomposition\run.py"
$cliArgs = @("--phase", $Phase, "--workers", $Workers)
if ($SkipExisting) { $cliArgs += "--skip-existing" }
$log = Join-Path $wt ("experiments\068_seed_decomposition\phase_" + $Phase + ".log")

"script           = $script $cliArgs"
"log              = $log"
""

# The header above goes to STDOUT, which dies with the ssh client. The calibration wave proved
# that: the client was killed at 590 s and the start time went with it, leaving no way to price
# a cell except by record mtimes. A stamp file survives independently of any console.
$stamp = Join-Path $wt ("experiments\068_seed_decomposition\phase_" + $Phase + ".stamp")
"started  = " + (Get-Date -Format o) | Set-Content -Path $stamp

# -u so the log is not fully buffered. Tee-Object so the record survives an ssh drop: Windows has
# no SIGHUP semantics, and an ssh client dying with "Broken pipe" leaves the job running. Do NOT
# wrap in Start-Process over ssh; that dies with the session.
& $py -u $script @cliArgs 2>&1 | Tee-Object -FilePath $log
"finished = " + (Get-Date -Format o) | Add-Content -Path $stamp
"finished         = " + (Get-Date -Format o)
