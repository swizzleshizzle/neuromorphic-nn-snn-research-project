# EXP-071 launcher for SwizzlesDuo, running from the laptop's MAIN checkout.
#
#   ... -File C:\Users\mlgbr\launch071.ps1 -Phase check
#   ... -File C:\Users\mlgbr\launch071.ps1 -Phase rank   -Workers 12
#   ... -File C:\Users\mlgbr\launch071.ps1 -Phase base   -Workers 20 -SkipExisting
#   ... -File C:\Users\mlgbr\launch071.ps1 -Phase det
#   ... -File C:\Users\mlgbr\launch071.ps1 -Phase calib  -Workers 2
#   ... -File C:\Users\mlgbr\launch071.ps1 -Phase full   -Workers 20 -SkipExisting
#
# Phases follow the plan's pre-flight order (Task 5): rank is step 0 (the critic ranking check,
# no C cell yet), base re-evaluates the four plain arms for Gate 0(b), det re-runs three arms a
# second time into a separate folder for the determinism check (Gate 0(a)), calib prices one C3
# and one C3V cell for wall-clock only, and full is the remaining eight arms.
#
# Re-evaluation only: nothing trains. Records are one JSON per cell, so a Windows Update restart
# loses at most the in-flight wave; re-run the same phase with -SkipExisting.
#
# NO BACKTICKS IN DOUBLE-QUOTED OUTPUT STRINGS, and NO COMMA-SEPARATED ARGUMENTS across ssh
# (cmd.exe eats commas and exits zero). Arm lists therefore live in this file, not on the
# command line.

param(
    [Parameter(Mandatory=$true)][ValidateSet("check","rank","base","det","calib","full")][string]$Phase,
    [int]$Workers = 0,
    [switch]$SkipExisting
)

$repo = "C:\Users\mlgbr\Desktop\Projects\neuromorphic-nn-snn-research-project"
$py   = Join-Path $repo ".venv\Scripts\python.exe"
$src  = Join-Path $repo "src"

if (-not (Test-Path $py)) { Write-Error "interpreter missing: $py"; exit 1 }
Set-Location $repo
$env:PYTHONPATH = $src

$where = & $py -c "import neuromorphic, sys; sys.stdout.write(neuromorphic.__file__)"
if ($LASTEXITCODE -ne 0) { Write-Error "could not import neuromorphic"; exit 1 }
if (-not $where.StartsWith($src, [System.StringComparison]::OrdinalIgnoreCase)) {
    Write-Error "WRONG LIBRARY. neuromorphic resolved to $where, expected under $src."
    exit 1
}

# The new module must be present: a stale checkout would fail later, but fail here first.
$hasLa = & $py -c "import sys; from neuromorphic.training.lookahead import evaluate_lookahead, MODES; sys.stdout.write(','.join(MODES))"
if ($hasLa -ne "G,E,P,R,C") { Write-Error "lookahead module missing or wrong (got '$hasLa'). Sync the repo."; exit 1 }

# 12 depth-7 heads, 12 E1 encoders and 12 critics, checked by this experiment's own cells.py.
$present = & $py -c "import importlib.util, sys; s=importlib.util.spec_from_file_location('c','experiments/071_critic_and_no_revisit/cells.py'); c=importlib.util.module_from_spec(s); s.loader.exec_module(c); from pathlib import Path; h=sum(c.c70.head_path(c.DEPTH, x).exists() for x in c.SEEDS); e=sum(Path(c.c70.published_config(c.DEPTH, x).encoder_state_path).exists() for x in c.SEEDS); cr=sum(c.critic_path(x).exists() for x in c.SEEDS); sys.stdout.write(str(h)+' '+str(e)+' '+str(cr))"
if ($present -ne "12 12 12") { Write-Error "expected 12 heads, 12 E1 encoders and 12 critics, found '$present'"; exit 1 }

$head = (& git rev-parse --short HEAD).Trim()
"library            = $where"
"checkout           = $head"
"heads, encoders, critics = $present"

$alive = @(Get-Process | Where-Object { $_.ProcessName -match "^python" }).Count
if ($alive -gt 2) { Write-Error "$alive python processes already running; refusing to start."; exit 1 }
"pyprocs            = $alive"

if ($Phase -eq "check") { "CHECK OK"; exit 0 }

# calib and full both run a committed C cell, which decides a claim: no C cell may run before
# the dated gate amendment lands. That commit sets GATE_R1_PASSED and GATE_R3_PASSED together,
# so checking both is one check, not two. det also touches C3, but into a throwaway folder
# outside the repo's committed outputs (exp071-det), purely for the determinism check, so it is
# not gated here.
if ($Phase -eq "calib" -or $Phase -eq "full") {
    $gates = & $py -c "import importlib.util, sys; s=importlib.util.spec_from_file_location('a','experiments/071_critic_and_no_revisit/aggregate.py'); a=importlib.util.module_from_spec(s); s.loader.exec_module(a); sys.stdout.write(str(a.GATE_R1_PASSED)+' '+str(a.GATE_R3_PASSED))"
    if ($gates -match "None") { Write-Error "GATE_R1_PASSED or GATE_R3_PASSED is unset ($gates): commit the dated gate amendment before any C cell beyond det."; exit 1 }
    "gates              = $gates"
}

$arms = switch ($Phase) {
    "rank"  { @() }
    "base"  { @("G0", "E3", "P3", "R3") }
    "det"   { @("G0V", "C3", "R3V") }
    "calib" { @("C3", "C3V") }
    "full"  { @("G0V", "E3V", "P3V", "C1", "C1V", "C3", "C3V", "R3V") }
}
if ($Workers -eq 0) { $Workers = 20 }

if ($Phase -eq "rank") {
    $w = if ($Workers -eq 0) { 12 } else { $Workers }
    $script = "experiments\071_critic_and_no_revisit\rank.py"
    $cliArgs = @("--workers", $w)
    $log = Join-Path $repo "experiments\071_critic_and_no_revisit\phase_rank.log"
    "script             = $script $cliArgs"
    "log                = $log"
    ""
    & $py -u $script @cliArgs 2>&1 | Tee-Object -FilePath $log
    exit 0
}

$script  = "experiments\071_critic_and_no_revisit\run.py"
$cliArgs = @("--arms") + $arms + @("--workers", $Workers)
if ($Phase -eq "det") {
    $cliArgs += @("--seeds", "0", "--out-dir", "C:\Users\mlgbr\exp071-det")
}
if ($Phase -eq "calib") { $cliArgs += @("--seeds", "0") }
if ($SkipExisting) { $cliArgs += "--skip-existing" }
$log = Join-Path $repo ("experiments\071_critic_and_no_revisit\phase_" + $Phase + ".log")

"script             = $script $cliArgs"
"log                = $log"
""

& $py -u $script @cliArgs 2>&1 | Tee-Object -FilePath $log
