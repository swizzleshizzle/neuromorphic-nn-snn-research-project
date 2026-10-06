# EXP-070 launcher for SwizzlesDuo, running from the laptop's MAIN checkout.
#
#   ... -File C:\Users\mlgbr\launch070.ps1 -Phase check
#   ... -File C:\Users\mlgbr\launch070.ps1 -Phase g0     -Workers 20 -SkipExisting
#   ... -File C:\Users\mlgbr\launch070.ps1 -Phase er     -Workers 20 -SkipExisting
#   ... -File C:\Users\mlgbr\launch070.ps1 -Phase calib  -Workers 2
#   ... -File C:\Users\mlgbr\launch070.ps1 -Phase p      -Workers 20 -SkipExisting
#
# Phases follow the spec's pre-flight order (section 2.6): g0 decides Gate 0's form, er
# calibrates Gate 1, calib prices P on one seed at depth 8, and p is the full P launch, which
# must not start before the dated spec amendment is committed.
#
# Re-evaluation only: nothing trains. Records are one JSON per cell, so a Windows Update restart
# loses at most the in-flight wave; re-run the same phase with -SkipExisting.
#
# NO BACKTICKS IN DOUBLE-QUOTED OUTPUT STRINGS, and NO COMMA-SEPARATED ARGUMENTS across ssh
# (cmd.exe eats commas and exits zero). Arm lists therefore live in this file, not on the
# command line.

param(
    [Parameter(Mandatory=$true)][ValidateSet("check","g0","er","calib","p")][string]$Phase,
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
if ($hasLa -ne "G,E,P,R") { Write-Error "lookahead module missing or wrong (got '$hasLa'). Sync the repo."; exit 1 }

# All 36 heads and 12 E1 encoders, checked by the experiment's own path functions.
$present = & $py -c "import importlib.util, sys; s=importlib.util.spec_from_file_location('c','experiments/070_lookahead_existing/cells.py'); c=importlib.util.module_from_spec(s); s.loader.exec_module(c); from pathlib import Path; h=sum(c.head_path(d,x).exists() for d in c.DEPTHS for x in c.SEEDS); e=sum(Path(c.published_config(7,x).encoder_state_path).exists() for x in c.SEEDS); sys.stdout.write(str(h)+' '+str(e))"
if ($present -ne "36 12") { Write-Error "expected 36 heads and 12 E1 encoders, found '$present'"; exit 1 }

$head = (& git rev-parse --short HEAD).Trim()
"library            = $where"
"checkout           = $head"
"heads, E1 encoders = $present"

$alive = @(Get-Process | Where-Object { $_.ProcessName -match "^python" }).Count
if ($alive -gt 2) { Write-Error "$alive python processes already running; refusing to start."; exit 1 }
"pyprocs            = $alive"

if ($Phase -eq "check") { "CHECK OK"; exit 0 }

# The full P launch is gated on the pre-launch spec amendment, which is the same commit that
# sets GATE0_FORM. Unset means the amendment has not landed, so no P cell beyond calib may run.
if ($Phase -eq "p") {
    $form = & $py -c "import importlib.util, sys; s=importlib.util.spec_from_file_location('a','experiments/070_lookahead_existing/aggregate.py'); a=importlib.util.module_from_spec(s); s.loader.exec_module(a); sys.stdout.write(str(a.GATE0_FORM))"
    if ($form -eq "None") { Write-Error "GATE0_FORM is unset: commit the dated spec amendment before the full P launch."; exit 1 }
    "GATE0_FORM         = $form"
}

$arms = switch ($Phase) {
    "g0"    { @("G0") }
    "er"    { @("E1", "E2", "E3", "R1", "R2", "R3") }
    "calib" { @("P2", "P3") }
    "p"     { @("P2", "P3") }
}
if ($Workers -eq 0) { $Workers = 20 }

$script  = "experiments\070_lookahead_existing\run.py"
$cliArgs = @("--arms") + $arms + @("--workers", $Workers)
if ($Phase -eq "calib") { $cliArgs += @("--depths", "8", "--seeds", "0") }
if ($SkipExisting) { $cliArgs += "--skip-existing" }
$log = Join-Path $repo ("experiments\070_lookahead_existing\phase_" + $Phase + ".log")

"script             = $script $cliArgs"
"log                = $log"
""

& $py -u $script @cliArgs 2>&1 | Tee-Object -FilePath $log
