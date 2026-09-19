# EXP-064 launcher for SwizzlesDuo, running from the WORKTREE at C:\Users\mlgbr\wt-exp053.
#
#   ... -File C:\Users\mlgbr\launch064_wt.ps1 -Phase check
#   ... -File C:\Users\mlgbr\launch064_wt.ps1 -Phase rl -Workers 6 -SkipExisting
#
# TWO arms, 24 cells, 6 workers. The arms do NOT cost the same per step: measured 104.69 ms for
# the motor arm (gradient through the spiking unroll) against 57.66 ms for the frozen-brain
# control, so ~4.40 h and ~2.43 h per cell respectively, about 82 CPU-h and ~13.7 h wall.
# Priced by RATIO against the memory arm, whose laptop cost is known (3.05 h/cell at 6 workers,
# EXP-059 and EXP-061). Pricing both arms with one figure is what made EXP-063's estimate 24% high.
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

# A stale worktree without the readout raises on the unknown mode, which is the LOUD failure.
# The silent one to guard against is a library that accepts the name while lacking the config
# fields, so this checks the mode AND both new fields AND the feature width together.
$hasMotor = & $py -c "import sys; from neuromorphic.training.cube_baseline import MOTOR_MODES, CubeConfig, feature_width; c = CubeConfig(readout='motor'); ok = ('motor' in MOTOR_MODES) and feature_width(c) == 6 and hasattr(c,'region_lr') and hasattr(c,'head_hidden'); sys.stdout.write(str(ok))"
if ($hasMotor -ne "True") {
    Write-Error "the worktree library has no working motor readout. It is stale; EXP-064 cannot run."
    exit 1
}

# THE DECISIVE CHECK, and the one a source grep cannot make. A worktree carrying the readout but
# NOT the grad_brain wiring would leave prefrontal and motor frozen at random init for the whole
# run, and produce a perfectly plausible result for a completely different experiment. This runs
# real episodes and asserts BOTH that the trainable surface is 15,540 AND that the regions
# actually MOVED. EXP-047 reported a 70x surface while the parameter moved by exactly 0.0.
$tmpOut = Join-Path $env:TEMP "exp064_launch_check"
$probe = & $py -c "import sys, pathlib; from neuromorphic.training.cube_baseline import CubeConfig, run_cube_baseline; d = pathlib.Path(sys.argv[1]); r = run_cube_baseline(CubeConfig(arm='regionalized', readout='motor', region_lr=0.01, tag='launchcheck', depth=1, seed=3, sigma=0.0, episodes=6, entropy_beta=0.0, max_depth=1, out_dir=d)); drift = r['region_drift'] or 0.0; sys.stdout.write(f\"{r['trainable_params']}|{drift:.6f}|{r['mean_n_stored']}\")" $tmpOut
$parts = $probe -split '\|'
if ($parts[0] -ne "15540") {
    Write-Error "trainable surface is $($parts[0]), expected 15540 (15456 prefrontal + 42 motor + 42 head). The regions are not in the optimizer."
    exit 1
}
if ([double]$parts[1] -le 0.0) {
    Write-Error "region_drift is $($parts[1]): prefrontal and motor did NOT move. grad_brain is not wired, and a run would train nothing while looking normal."
    exit 1
}
if ([double]$parts[2] -ne 0.0) {
    Write-Error "mean_n_stored is $($parts[2]): the hippocampus is engaged for the motor arm. Memory hurts on this task and this arm must not use it."
    exit 1
}
Remove-Item -Recurse -Force $tmpOut -ErrorAction SilentlyContinue

# 12 E0 encoders, seeds 0-11. NOT tracked in git.
$encDir = Join-Path $wt "experiments\040_pretrained_encoder_policy\outputs"
$e0 = 0
foreach ($s in 0..11) { if (Test-Path (Join-Path $encDir "exp040_encoder_s$s.pt")) { $e0++ } }
if ($e0 -lt 12) { Write-Error "found $e0 of 12 E0 encoders for seeds 0-11 in $encDir."; exit 1 }

$head   = (& git rev-parse --short HEAD).Trim()
$branch = (& git rev-parse --abbrev-ref HEAD).Trim()
"library           = $where"
"worktree          = $branch @ $head"
"motor readout     = $hasMotor"
"trainable surface = $($parts[0])"
"region drift      = $($parts[1])   (must be > 0: the regions really train)"
"hippocampus       = off (mean_n_stored $($parts[2]))"
"E0 encoders       = $e0 of 12   (seeds 0-11)"

$alive = @(Get-Process | Where-Object { $_.ProcessName -match "^python" }).Count
if ($alive -gt 2) { Write-Error "$alive python processes already running; refusing to start."; exit 1 }
"pyprocs           = $alive"

if ($Phase -eq "check") { "CHECK OK"; exit 0 }

if ($Workers -eq 0) { $Workers = 6 }
$outDir = Join-Path $wt "experiments\064_motor_policy_path\outputs"
New-Item -ItemType Directory -Force $outDir | Out-Null

$script  = "experiments\064_motor_policy_path\run.py"
$cliArgs = @("--workers", $Workers)
if ($SkipExisting) { $cliArgs += "--skip-existing" }
$log = Join-Path $wt "experiments\064_motor_policy_path\phase_rl.log"

"script            = $script $cliArgs"
"log               = $log"
""

# -u so the log is not fully buffered. Tee-Object so the record survives an ssh drop: Windows
# has no SIGHUP semantics, and an ssh client dying with "Broken pipe" leaves the job running.
& $py -u $script @cliArgs 2>&1 | Tee-Object -FilePath $log
