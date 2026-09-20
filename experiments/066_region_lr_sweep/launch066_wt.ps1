# EXP-066 launcher for SwizzlesDuo, running from the WORKTREE at C:\Users\mlgbr\wt-exp053.
#
#   ... -File C:\Users\mlgbr\launch066_wt.ps1 -Phase check
#   ... -File C:\Users\mlgbr\launch066_wt.ps1 -Phase rl -Workers 6 -SkipExisting
#
# 24 cells, ALL of them motor arms, 6 workers = 4 clean waves. Unlike EXP-064 there is no
# per-arm cost asymmetry to price separately: EXP-064's motor arm measured ~4.50 h per cell at
# 6 workers (its 14.0 h actual against a 13.7 h estimate), so ~108 CPU-h and ~18 h wall.
#
# THIS EXPERIMENT ADDS NO CODE. readout="motor" and region_lr shipped in EXP-064, so the probe
# below is EXP-064's, reused unchanged. Only a configuration differs.
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

# THE DECISIVE CHECK, reused from EXP-064 because it validates exactly what this sweep needs.
# A worktree carrying the readout but NOT the grad_brain wiring would leave prefrontal and motor
# frozen at random init for the whole run and produce a plausible result for a different
# experiment. This runs real episodes and asserts the trainable surface is 15,540, that the
# regions actually MOVED, and that the hippocampus is off.
$tmpOut = Join-Path $env:TEMP "exp066_launch_check"
$probe = & $py -u "experiments\064_motor_policy_path\launch_probe.py" $tmpOut
if ($LASTEXITCODE -ne 0) { Write-Error "the pre-flight probe failed to run: $probe"; exit 1 }
$parts = ($probe.Trim()) -split '\s+'
if ($parts.Count -ne 3) { Write-Error "probe returned $($parts.Count) values, expected 3: $probe"; exit 1 }
if ($parts[0] -ne "15540") {
    Write-Error "trainable surface is $($parts[0]), expected 15540 (15456 prefrontal + 42 motor + 42 head)."
    exit 1
}
if ([double]$parts[1] -le 0.0) {
    Write-Error "region_drift is $($parts[1]): prefrontal and motor did NOT move. grad_brain is not wired."
    exit 1
}
if ([double]$parts[2] -ne 0.0) {
    Write-Error "mean_n_stored is $($parts[2]): the hippocampus is engaged. Memory hurts on this task."
    exit 1
}
Remove-Item -Recurse -Force $tmpOut -ErrorAction SilentlyContinue

# 12 E0 encoders, seeds 0-11. NOT tracked in git.
$encDir = Join-Path $wt "experiments\040_pretrained_encoder_policy\outputs"
$e0 = 0
foreach ($s in 0..11) { if (Test-Path (Join-Path $encDir "exp040_encoder_s$s.pt")) { $e0++ } }
if ($e0 -lt 12) { Write-Error "found $e0 of 12 E0 encoders for seeds 0-11 in $encDir."; exit 1 }

# EXP-064's motor arm is REUSED as the third point of the sweep. Without it the sweep has two
# points instead of three and the spec's reading changes.
$prior = @(Get-ChildItem -Path (Join-Path $wt "experiments\064_motor_policy_path\outputs") -Filter "exp064_motor_d5*.json" -ErrorAction SilentlyContinue).Count
if ($prior -lt 12) { Write-Error "found $prior of 12 EXP-064 motor records; they are the sweep's 1e-2 point and must be present."; exit 1 }

$head   = (& git rev-parse --short HEAD).Trim()
$branch = (& git rev-parse --abbrev-ref HEAD).Trim()
"library           = $where"
"worktree          = $branch @ $head"
"trainable surface = $($parts[0])"
"region drift      = $($parts[1])   (must be > 0: the regions really train)"
"hippocampus       = off (mean_n_stored $($parts[2]))"
"E0 encoders       = $e0 of 12   (seeds 0-11)"
"EXP-064 records   = $prior of 12   (the sweep's 1e-2 point, reused)"

$alive = @(Get-Process | Where-Object { $_.ProcessName -match "^python" }).Count
if ($alive -gt 2) { Write-Error "$alive python processes already running; refusing to start."; exit 1 }
"pyprocs           = $alive"

if ($Phase -eq "check") { "CHECK OK"; exit 0 }

if ($Workers -eq 0) { $Workers = 6 }
$outDir = Join-Path $wt "experiments\066_region_lr_sweep\outputs"
New-Item -ItemType Directory -Force $outDir | Out-Null

$script  = "experiments\066_region_lr_sweep\run.py"
$cliArgs = @("--workers", $Workers)
if ($SkipExisting) { $cliArgs += "--skip-existing" }
$log = Join-Path $wt "experiments\066_region_lr_sweep\phase_rl.log"

"script            = $script $cliArgs"
"log               = $log"
""

# -u so the log is not fully buffered. Tee-Object so the record survives an ssh drop.
& $py -u $script @cliArgs 2>&1 | Tee-Object -FilePath $log
