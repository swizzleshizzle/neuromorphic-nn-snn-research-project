# EXP-063 launcher for SwizzlesDuo, running from the WORKTREE at C:\Users\mlgbr\wt-exp053.
#
#   ... -File C:\Users\mlgbr\launch063_wt.ps1 -Phase check
#   ... -File C:\Users\mlgbr\launch063_wt.ps1 -Phase rl -Workers 6 -SkipExisting
#
# TWO arms, 48 cells, 6 workers = 8 clean waves. Six rather than ten deliberately: per-cell time
# is WORSE at 10 workers (0.16 vs 0.115 s/step measured), 48 divides by 6 exactly, and EXP-059
# and EXP-061 both measured ~3.05 h per cell at 6 workers on this same depth and config - so the
# estimate comes from a same-worker-count measurement rather than a cross-worker-count scaling,
# which has now come in low five times running.
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

# A stale worktree without the modes raises on the unknown readout, which is the LOUD failure.
# The silent one to guard against is a worktree that accepts the name while lacking the
# machinery, so this constructs the attention directly and checks its parameter count.
$hasAttn = & $py -c "import sys; from neuromorphic.training.cube_baseline import ATTN_MODES, AttentionReadout; a = AttentionReadout(64); n = sum(p.numel() for p in a.parameters()); sys.stdout.write(str(sorted(ATTN_MODES) == ['memory_attn','memory_attn_noise'] and n == 2048))"
if ($hasAttn -ne "True") {
    Write-Error "the worktree library has no working AttentionReadout. It is stale; EXP-063 cannot run."
    exit 1
}

# Both validity gates read these fields. Without them the gates read None and VOID the run.
$hasInstr = & $py -c "import inspect, sys; from neuromorphic.training import cube_baseline as cb; s = inspect.getsource(cb); sys.stdout.write(str(all(k in s for k in ('attn_choice_steps','train_steps','recall_concept_norm_ratio','attn_entropy_norm'))))"
if ($hasInstr -ne "True") {
    Write-Error "missing gate instruments in the worktree library. EXP-063's gates would have no data."
    exit 1
}

# THE DECISIVE CHECK, and the one a source grep cannot make. A worktree carrying the readout but
# NOT the optimizer parameter group would leave the attention frozen at its random init for the
# whole run and produce a perfectly plausible null. This runs two real episodes and asserts the
# trainable surface is the 780-parameter head PLUS the 2,048-parameter attention.
$tmpOut = Join-Path $env:TEMP "exp063_launch_check"
$surface = & $py -c "import sys, pathlib; from neuromorphic.training.cube_baseline import CubeConfig, run_cube_baseline; d = pathlib.Path(sys.argv[1]); r = run_cube_baseline(CubeConfig(arm='regionalized', readout='memory_attn', tag='launchcheck', depth=1, seed=3, sigma=0.0, episodes=2, entropy_beta=0.0, max_depth=1, out_dir=d)); sys.stdout.write(str(r['trainable_params']))" $tmpOut
if ($surface -ne "2828") {
    Write-Error "trainable surface is $surface, expected 2828 (780 head + 2048 attention). The attention is NOT in the optimizer; a run would train nothing and return a plausible null."
    exit 1
}
Remove-Item -Recurse -Force $tmpOut -ErrorAction SilentlyContinue

# All 24 E0 encoders. NOT tracked in git, so a fresh worktree has none and a synced one has
# only whichever a previous run left behind. Refuse rather than silently run a smaller sweep.
$encDir = Join-Path $wt "experiments\040_pretrained_encoder_policy\outputs"
$e0 = @(Get-ChildItem -Path $encDir -Filter "exp040_encoder_s*.pt" -ErrorAction SilentlyContinue).Count
if ($e0 -lt 24) { Write-Error "found $e0 of 24 E0 encoders in $encDir; copy them from the main checkout."; exit 1 }

# EXP-059's and EXP-061's arms are REUSED, not re-run. If their records are absent the
# aggregator has nothing to contrast against and the experiment answers nothing.
$prior059 = @(Get-ChildItem -Path (Join-Path $wt "experiments\059_memory_depth5\outputs") -Filter "*.json" -ErrorAction SilentlyContinue).Count
$prior061 = @(Get-ChildItem -Path (Join-Path $wt "experiments\061_noise_matched_recall\outputs") -Filter "*.json" -ErrorAction SilentlyContinue).Count
if ($prior059 -lt 24) { Write-Error "found $prior059 EXP-059 records; arms A and M are reused and must be present."; exit 1 }
if ($prior061 -lt 24) { Write-Error "found $prior061 EXP-061 records; arm N is reused and must be present."; exit 1 }

$head   = (& git rev-parse --short HEAD).Trim()
$branch = (& git rev-parse --abbrev-ref HEAD).Trim()
"library           = $where"
"worktree          = $branch @ $head"
"attention modes   = $hasAttn"
"gate instruments  = $hasInstr"
"trainable surface = $surface"
"E0 encoders       = $e0 of 24"
"EXP-059 records   = $prior059   (arms A and M, reused)"
"EXP-061 records   = $prior061   (arm N, reused)"

$alive = @(Get-Process | Where-Object { $_.ProcessName -match "^python" }).Count
if ($alive -gt 2) { Write-Error "$alive python processes already running; refusing to start."; exit 1 }
"pyprocs           = $alive"

if ($Phase -eq "check") { "CHECK OK"; exit 0 }

if ($Workers -eq 0) { $Workers = 6 }
$outDir = Join-Path $wt "experiments\063_learned_readout\outputs"
New-Item -ItemType Directory -Force $outDir | Out-Null

$script  = "experiments\063_learned_readout\run.py"
$cliArgs = @("--workers", $Workers)
if ($SkipExisting) { $cliArgs += "--skip-existing" }
$log = Join-Path $wt "experiments\063_learned_readout\phase_rl.log"

"script            = $script $cliArgs"
"log               = $log"
""

# -u so the log is not fully buffered. Tee-Object so the record survives an ssh drop: Windows
# has no SIGHUP semantics, and an ssh client dying with "Broken pipe" leaves the job running.
# Do NOT wrap in Start-Process over ssh; that dies with the session.
& $py -u $script @cliArgs 2>&1 | Tee-Object -FilePath $log
