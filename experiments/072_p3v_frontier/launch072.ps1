# EXP-072 launcher for SwizzlesDuo, running from the laptop's MAIN checkout.
#
#   ... -File C:\Users\mlgbr\launch072.ps1 -Phase check
#   ... -File C:\Users\mlgbr\launch072.ps1 -Phase cont  -Workers 4
#   ... -File C:\Users\mlgbr\launch072.ps1 -Phase det   -Workers 4
#   ... -File C:\Users\mlgbr\launch072.ps1 -Phase full  -Workers 20 -SkipExisting
#
# Order (spec section 5): cont re-runs G0 and P3 at seed 0, depths 8 and 9, for Gate 0(b); stop if
# they do not equal EXP-070's records. det re-runs P3V and G0V at seed 0 into a SEPARATE directory
# for Gate 0(a). full runs the 72 new cells (G0V, P3V, R3V x depths 8, 9 x 12 seeds).
#
# Re-evaluation only: nothing trains. Records are one JSON per cell, so a Windows Update restart
# loses at most the in-flight wave; re-run the same phase with -SkipExisting.
#
# NO BACKTICKS IN DOUBLE-QUOTED OUTPUT STRINGS, and NO COMMA-SEPARATED ARGUMENTS across ssh
# (cmd.exe eats commas and exits zero). Arm lists therefore live in this file.

param(
    [Parameter(Mandatory=$true)][ValidateSet("check","cont","det","full")][string]$Phase,
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

# The V modifier must be present: no_revisit is what every new arm depends on.
$hasV = & $py -c "import inspect, sys; from neuromorphic.training.lookahead import evaluate_lookahead; sys.stdout.write(str('no_revisit' in inspect.signature(evaluate_lookahead).parameters))"
if ($LASTEXITCODE -ne 0 -or $hasV -ne "True") { Write-Error "evaluate_lookahead has no no_revisit (got '$hasV'). Sync the repo."; exit 1 }

# 24 heads (depths 8 and 9) and 12 E1 encoders, via EXP-070's own path functions.
$present = & $py -c "import importlib.util, sys; s=importlib.util.spec_from_file_location('c','experiments/070_lookahead_existing/cells.py'); c=importlib.util.module_from_spec(s); s.loader.exec_module(c); from pathlib import Path; h=sum(c.head_path(d,x).exists() for d in (8,9) for x in c.SEEDS); e=sum(Path(c.published_config(8,x).encoder_state_path).exists() for x in c.SEEDS); sys.stdout.write(str(h)+' '+str(e))"
if ($present -ne "24 12") { Write-Error "expected 24 heads and 12 E1 encoders, found '$present'"; exit 1 }

$head = (& git rev-parse --short HEAD).Trim()
"library            = $where"
"checkout           = $head"
"heads, E1 encoders = $present"

$alive = @(Get-Process | Where-Object { $_.ProcessName -match "^python" }).Count
if ($alive -gt 2) { Write-Error "$alive python processes already running; refusing to start."; exit 1 }
"pyprocs            = $alive"

if ($Phase -eq "check") { "CHECK OK"; exit 0 }

$arms = switch ($Phase) {
    "cont" { @("G0", "P3") }
    "det"  { @("P3V", "G0V") }
    "full" { @("G0V", "P3V", "R3V") }
}
if ($Workers -eq 0) { $Workers = 20 }

$script  = "experiments\072_p3v_frontier\run.py"
$cliArgs = @("--arms") + $arms + @("--workers", $Workers)
if ($Phase -eq "cont" -or $Phase -eq "det") { $cliArgs += @("--seeds", "0") }
if ($Phase -eq "det") { $cliArgs += @("--out-dir", "C:\Users\mlgbr\exp072-det") }
if ($SkipExisting) { $cliArgs += "--skip-existing" }
$log = Join-Path $repo ("experiments\072_p3v_frontier\phase_" + $Phase + ".log")

"script             = $script $cliArgs"
"log                = $log"
""

& $py -u $script @cliArgs 2>&1 | Tee-Object -FilePath $log
