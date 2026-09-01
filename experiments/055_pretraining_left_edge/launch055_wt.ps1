# EXP-055 launcher for SwizzlesDuo, running from the WORKTREE at C:\Users\mlgbr\wt-exp053.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File C:\Users\mlgbr\launch055_wt.ps1 -Phase check
#   powershell -NoProfile -ExecutionPolicy Bypass -File C:\Users\mlgbr\launch055_wt.ps1 -Phase pretrain -Workers 8
#   powershell -NoProfile -ExecutionPolicy Bypass -File C:\Users\mlgbr\launch055_wt.ps1 -Phase rl -AllArms -Workers 8
#
# USE -AllArms, NOT -Epochs 1,2,3,5. Over ssh the remote shell is cmd.exe, which treats commas
# as argument separators, so `-Epochs 1,2,3,5` arrives as the single token `1235` and
# ValidateSet rejects it. That failure exits ZERO, so the dispatching ssh looks successful
# while nothing started. A switch has no comma to eat. Verify a launch by probing for records
# and worker processes, never by the ssh exit code.
#
# All four arms go in ONE call. run.py submits every cell to a single pool, so the pool keeps
# refilling across arm boundaries instead of draining to a two-wave tail four separate times,
# and a dropped ssh cannot leave the sequence half-dispatched. Add -SkipExisting to resume:
# records already on disk are skipped, and seeded runs are byte-identical, so a resumed sweep
# is indistinguishable from an uninterrupted one.
#
# WHY A WORKTREE-SPECIFIC LAUNCHER EXISTS, AND WHY IT IS NOT launch055.ps1
#
# The laptop's main checkout could not switch branches: 48 EXP-052 encoder .pt files are
# untracked there but tracked in the branch, and 24 differ by hash, so a forced checkout would
# have destroyed them. The work therefore runs in a worktree, and THE WORKTREE HAS NO .venv.
#
# The only interpreter on the machine is the main checkout's, and that venv installs the project
# editable with an ABSOLUTE path to the MAIN CHECKOUT's src. Running a worktree script with it
# imports the OLD library while producing a complete, plausible, entirely wrong result. There is
# no error and no warning. PYTHONPATH overrides it, and the check below PROVES it did rather
# than assuming it: `neuromorphic.__file__` must resolve under the worktree or this refuses to
# start. Never delete that gate to save a few seconds.

param(
    [Parameter(Mandatory=$true)][ValidateSet("check","pretrain","rl")][string]$Phase,
    [ValidateSet(1,2,3,5)][int[]]$Epochs = @(),
    [switch]$AllArms,
    [int]$Workers = 0,
    [switch]$SkipExisting
)

if ($AllArms) { $Epochs = @(1, 2, 3, 5) }

$wt   = "C:\Users\mlgbr\wt-exp053"
$py   = "C:\Users\mlgbr\Desktop\Projects\neuromorphic-nn-snn-research-project\.venv\Scripts\python.exe"
$src  = Join-Path $wt "src"

if (-not (Test-Path $wt)) { Write-Error "worktree missing: $wt"; exit 1 }
if (-not (Test-Path $py)) { Write-Error "interpreter missing: $py"; exit 1 }

Set-Location $wt
$env:PYTHONPATH = $src

# --- the gate. The library must come from the worktree, not the main checkout. ---
$where = & $py -c "import neuromorphic, sys; sys.stdout.write(neuromorphic.__file__)"
if ($LASTEXITCODE -ne 0) { Write-Error "could not import neuromorphic"; exit 1 }
if (-not $where.StartsWith($src, [System.StringComparison]::OrdinalIgnoreCase)) {
    Write-Error "WRONG LIBRARY. neuromorphic resolved to $where, expected under $src. PYTHONPATH did not win; refusing to run."
    exit 1
}
$head   = (& git rev-parse --short HEAD).Trim()
$branch = (& git rev-parse --abbrev-ref HEAD).Trim()
"library   = $where"
"worktree  = $branch @ $head"

# Refuse to share the machine. Worker processes appear as python3.13.exe, not python.exe,
# so match on ^python or this guard misses them completely.
$alive = @(Get-Process | Where-Object { $_.ProcessName -match "^python" }).Count
if ($alive -gt 2) { Write-Error "$alive python processes already running; refusing to start."; exit 1 }
"pyprocs   = $alive"

if ($Phase -eq "check") { "CHECK OK"; exit 0 }

$outDir = Join-Path $wt "experiments\055_pretraining_left_edge\outputs"
New-Item -ItemType Directory -Force $outDir | Out-Null

if ($Phase -eq "pretrain") {
    # 48 encoders. Pretraining is memory-bandwidth-bound, not core-bound: EXP-050 measured
    # 2.86 effective cores from 10 workers, against about 7.4 for RL cells. 8 divides 48 into
    # six clean waves and leaves commit-limit headroom that 10 does not.
    if ($Workers -eq 0) { $Workers = 8 }
    $script = "experiments\055_pretraining_left_edge\pretrain_left_edge.py"
    $cliArgs = @("--workers", $Workers)
    $log = Join-Path $wt "experiments\055_pretraining_left_edge\phase1_pretrain.log"
} else {
    if ($Epochs.Count -eq 0) { Write-Error "-Phase rl requires -AllArms, or -Epochs with a single value"; exit 1 }
    if ($Workers -eq 0) { $Workers = 6 }
    $script = "experiments\055_pretraining_left_edge\run.py"
    $cliArgs = @("--epochs") + $Epochs + @("--workers", $Workers)
    if ($SkipExisting) { $cliArgs += "--skip-existing" }
    $log = Join-Path $wt ("experiments\055_pretraining_left_edge\phase3_e" + ($Epochs -join "_") + ".log")
}

"script    = $script $cliArgs"
"log       = $log"
""

# -u so the log is not fully buffered. Tee-Object so the record survives an ssh drop: Windows
# has no SIGHUP semantics, so a dropped connection does not kill the run. Do NOT wrap this in
# Start-Process -WindowStyle Hidden over ssh; that DIES when the session ends. Run it in the
# ssh foreground and background the call on the controller side instead.
& $py -u $script @cliArgs 2>&1 | Tee-Object -FilePath $log
