# EXP-062 launcher for SwizzlesDuo, running from the WORKTREE at C:\Users\mlgbr\wt-exp053.
#
#   ... -File C:\Users\mlgbr\launch062_wt.ps1 -Phase check
#   ... -File C:\Users\mlgbr\launch062_wt.ps1 -Phase rl -Workers 6 -SkipExisting
#
# 24 cells (depths 8 and 9 x 12 seeds), 6 workers = 4 clean waves. Six rather than ten because
# per-cell time is better there (0.115 vs 0.16 s/step measured) and 24 divides by 6 exactly.
# Expect ~17 h and treat it as a FLOOR: five of the last six estimates came in low, and the one
# that held was priced from a same-worker-count measurement exactly like this one.
#
# THE WORKTREE HAS NO .venv, AND THAT IS A TRAP. The only interpreter belongs to the main
# checkout and installs the project editable with an ABSOLUTE path to the MAIN CHECKOUT's src,
# so a worktree script run with it imports the OLD library and produces a complete, plausible,
# entirely wrong result with no error. PYTHONPATH overrides it and the gate below PROVES it did.
#
# NO BACKTICKS IN DOUBLE-QUOTED OUTPUT STRINGS. PowerShell reads `b as backspace and `f as form
# feed, which silently ate a character in EXP-060's banner.
#
# NO COMMA-SEPARATED ARGUMENTS. cmd.exe eats commas and the failure EXITS ZERO. Depth and seed
# lists are therefore left to the driver's defaults and never cross the ssh boundary. Verify a
# launch by probing for records and worker processes, never by the ssh exit code.

param(
    [Parameter(Mandatory=$true)][ValidateSet("check","rl")][string]$Phase,
    [int]$Workers = 0,
    [switch]$SkipExisting
)

$wt  = "C:\Users\mlgbr\wt-exp053"
$py  = "C:\Users\mlgbr\Desktop\Projects\neuromorphic-nn-snn-research-project\.venv\Scripts\python.exe"
$src = Join-Path $wt "src"
$seeds = 0..11

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

# The learned critic IS the recipe component under test here. A worktree without it would run
# arm B's config minus its critic and score near zero at depth 8, which would look exactly like
# a real frontier result.
$hasCritic = & $py -c "from neuromorphic.training.cube_baseline import CubeConfig; import sys; sys.stdout.write(str('critic_lr' in CubeConfig.__dataclass_fields__))"
if ($hasCritic -ne "True") {
    Write-Error "CubeConfig has no critic_lr field. The worktree is stale; this would silently run arm B WITHOUT its critic."
    exit 1
}

# The validity gate reads these. Without them the gate has no data and the run is unreadable.
$hasGate = & $py -c "import inspect, sys; from neuromorphic.training import reinforce; s=inspect.getsource(reinforce); sys.stdout.write(str('critic_within_ss' in s and 'return_within_ss' in s))"
if ($hasGate -ne "True") {
    Write-Error "no critic_within_ss/return_within_ss terms in the worktree library. EXP-062's validity gate would have no data."
    exit 1
}

# Depth 9 needs a BFS table two shells deeper than anything run before. Cheap (3.3 s measured)
# but it must actually build, and the depth-9 shell must be non-empty.
$bfs = & $py -c "import sys; from neuromorphic.envs.cube_distance import ExactBFSDistance; from neuromorphic.training.cube_baseline import shell_states; p=ExactBFSDistance(max_depth=9); sys.stdout.write(str(len(shell_states(p,8))) + ',' + str(len(shell_states(p,9))))"
if ($LASTEXITCODE -ne 0) { Write-Error "BFS table failed to build to depth 9"; exit 1 }

# E1 encoders, seeds 0-11. TRACKED in git (the *_encoder.pt un-ignore), so a synced worktree has
# them - but verify rather than assume, because a smaller sweep would run silently.
$ftDir = Join-Path $wt "experiments\047_encoder_finetuning\outputs"
$nE1 = @($seeds | Where-Object { Test-Path (Join-Path $ftDir ("exp047_ft_d6_lr0.0001_regionalized_d6_s" + $_ + "_sig0.0_encoder.pt")) }).Count
if ($nE1 -lt 12) { Write-Error "found $nE1 of 12 E1 encoders in $ftDir; EXP-062 needs seeds 0-11."; exit 1 }

$head   = (& git rev-parse --short HEAD).Trim()
$branch = (& git rev-parse --abbrev-ref HEAD).Trim()
"library            = $where"
"worktree           = $branch @ $head"
"critic_lr field    = $hasCritic"
"gate instruments   = $hasGate"
"shell sizes d8,d9  = $bfs"
"E1 encoders 0-11   = $nE1 of 12"

$alive = @(Get-Process | Where-Object { $_.ProcessName -match "^python" }).Count
if ($alive -gt 2) { Write-Error "$alive python processes already running; refusing to start."; exit 1 }
"pyprocs            = $alive"

if ($Phase -eq "check") { "CHECK OK"; exit 0 }

if ($Workers -eq 0) { $Workers = 6 }
$outDir = Join-Path $wt "experiments\062_depth_frontier\outputs"
New-Item -ItemType Directory -Force $outDir | Out-Null

$script  = "experiments\062_depth_frontier\run.py"
$cliArgs = @("--workers", $Workers)
if ($SkipExisting) { $cliArgs += "--skip-existing" }
$log = Join-Path $wt "experiments\062_depth_frontier\phase_rl.log"

"script             = $script $cliArgs"
"log                = $log"
""

# -u so the log is not fully buffered. Tee-Object so the record survives an ssh drop: Windows has
# no SIGHUP semantics, and an ssh client dying with "Broken pipe" leaves the job running.
& $py -u $script @cliArgs 2>&1 | Tee-Object -FilePath $log
