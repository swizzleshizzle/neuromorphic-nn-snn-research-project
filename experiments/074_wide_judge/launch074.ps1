# EXP-074 launcher for SwizzlesDuo, running from the laptop's MAIN checkout.
#
#   ... -File C:\Users\mlgbr\launch074.ps1 -Phase check
#   ... -File C:\Users\mlgbr\launch074.ps1 -Phase pilot -Workers 4
#   ... -File C:\Users\mlgbr\launch074.ps1 -Phase train -Workers 20
#   ... -File C:\Users\mlgbr\launch074.ps1 -Phase rank  -Workers 12
#   ... -File C:\Users\mlgbr\launch074.ps1 -Phase cont  -Workers 2
#   ... -File C:\Users\mlgbr\launch074.ps1 -Phase eval  -Workers 20 -SkipExisting
#   ... -File C:\Users\mlgbr\launch074.ps1 -Phase det   -Workers 2
#
# Order (spec section 9): pilot (arms W, A, B, seeds 12 and 13 only) -> dated amendment -> train
# (seeds 0 to 11, banked checkpoints) -> Gates L and E and the policy reference, committed -> rank
# -> cont -> eval -> det. cont re-runs P3V at seed 0, depths 8 and 9, for Gate 0(b); stop if they
# do not equal EXP-072's records. det re-runs J3V-W and J3V-A at seed 0, depth 9, into a SEPARATE
# directory for Gate 0(a). The pilot re-runs A and B for Gate 0(c) against EXP-073 pilot 1.
# P3V and R3V at depths 7 to 9 are NOT re-run: they come from EXP-071 and EXP-072's committed
# records, re-checked by cont.
#
# Training resumes by itself: train_judge reads its banked checkpoint at the start, and train.py
# skips (arm, seed) jobs whose record already exists, so re-running the same train phase after a
# Windows Update restart continues rather than restarts (there is no --resume flag, and none is
# needed). Evaluation records are one JSON per cell; re-run the same phase with -SkipExisting.
#
# NO BACKTICKS IN DOUBLE-QUOTED OUTPUT STRINGS, and NO COMMA-SEPARATED ARGUMENTS across ssh
# (cmd.exe eats commas and exits zero). Arm and seed lists therefore live in this file, not on
# the command line.

param(
    [Parameter(Mandatory=$true)][ValidateSet("check","pilot","train","rank","cont","eval","det")][string]$Phase,
    [int]$Workers = 0,
    [switch]$SkipExisting
)

# Spec section 4, fixed: the EXP-073 pilots ran exactly these for A and B.
$Updates = 4000
$SyncEvery = 100
$ProbeEvery = 250
$PilotOutDir = "experiments\074_wide_judge\outputs_pilot"

$repo = "C:\Users\mlgbr\Desktop\Projects\neuromorphic-nn-snn-research-project"
$py   = Join-Path $repo ".venv\Scripts\python.exe"
$src  = Join-Path $repo "src"
$exp  = "experiments\074_wide_judge"

if (-not (Test-Path $py)) { Write-Error "interpreter missing: $py"; exit 1 }
Set-Location $repo
$env:PYTHONPATH = $src

$where = & $py -c "import neuromorphic, sys; sys.stdout.write(neuromorphic.__file__)"
if ($LASTEXITCODE -ne 0) { Write-Error "could not import neuromorphic"; exit 1 }
if (-not $where.StartsWith($src, [System.StringComparison]::OrdinalIgnoreCase)) {
    Write-Error "WRONG LIBRARY. neuromorphic resolved to $where, expected under $src."
    exit 1
}

# The wide-readout seam and the state-reading lookahead seam must be present: a stale checkout fails here first.
$hasVi = & $py -c "import sys, inspect; from neuromorphic.training import value_iteration as v, lookahead as l; sys.stdout.write(str('readout' in inspect.signature(v.Judge).parameters and 'reads_states' in inspect.getsource(l.imagined_values)))"
if ($LASTEXITCODE -ne 0 -or $hasVi -ne "True") { Write-Error "readout or reads_states seam missing (got '$hasVi'). Sync the repo."; exit 1 }

# 36 heads (depths 7, 8 and 9 x 12 seeds) and 12 E1 encoders, via EXP-070's own path functions.
$present = & $py -c "import importlib.util, sys; s=importlib.util.spec_from_file_location('c','experiments/070_lookahead_existing/cells.py'); c=importlib.util.module_from_spec(s); s.loader.exec_module(c); from pathlib import Path; h=sum(c.head_path(d,x).exists() for d in (7,8,9) for x in c.SEEDS); e=sum(Path(c.published_config(7,x).encoder_state_path).exists() for x in c.SEEDS); sys.stdout.write(str(h)+' '+str(e))"
if ($LASTEXITCODE -ne 0 -or $present -ne "36 12") { Write-Error "expected 36 heads and 12 E1 encoders, found '$present'"; exit 1 }

$head = (& git rev-parse --short HEAD).Trim()
"library            = $where"
"checkout           = $head"
"heads, E1 encoders = $present"

$alive = @(Get-Process | Where-Object { $_.ProcessName -match "^python" }).Count
if ($alive -gt 2) { Write-Error "$alive python processes already running; refusing to start."; exit 1 }
"pyprocs            = $alive"

if ($Phase -eq "check") { "CHECK OK"; exit 0 }

# train, rank, eval and det all read or produce numbers that decide a claim. None may run before
# the dated amendment sets GATE_L_THRESHOLD. Check the exit code too: if aggregate.py fails to
# import, stdout is empty, "" does not match "None", and the phase would otherwise run UNGATED.
if ($Phase -ne "pilot" -and $Phase -ne "cont") {
    $gate = & $py -c "import importlib.util, sys; s=importlib.util.spec_from_file_location('a','$($exp.Replace('\','/'))/aggregate.py'); a=importlib.util.module_from_spec(s); s.loader.exec_module(a); sys.stdout.write(str(a.GATE_L_THRESHOLD))"
    if ($LASTEXITCODE -ne 0 -or $gate -eq "") { Write-Error "could not read GATE_L_THRESHOLD (got '$gate'); refusing to run."; exit 1 }
    if ($gate -match "None") { Write-Error "GATE_L_THRESHOLD is unset ($gate): commit the dated post-pilot amendment before this phase."; exit 1 }
    "gate L threshold   = $gate"
}

if ($Workers -eq 0) { $Workers = 20 }
$log = Join-Path $repo ("$exp\phase_" + $Phase + ".log")

# Training phases: one train.py call, it manages its own worker pool.
if ($Phase -eq "pilot" -or $Phase -eq "train") {
    if ($Phase -eq "pilot") {
        $cliArgs = @("--pilot", "--arms", "W", "A", "B", "--seeds", "12", "13", "--n-updates", $Updates, "--sync-every", $SyncEvery, "--probe-every", $ProbeEvery, "--out-dir", (Join-Path $repo $PilotOutDir), "--workers", $Workers)
        $log = Join-Path $repo "$exp\phase_pilot.log"
    } else {
        $cliArgs = @("--arms", "W", "A", "B", "--seeds", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "--n-updates", $Updates, "--sync-every", $SyncEvery, "--probe-every", $ProbeEvery, "--workers", $Workers)
    }
    "script             = $exp\train.py $cliArgs"
    "log                = $log"
    ""
    & $py -u "$exp\train.py" @cliArgs 2>&1 | Tee-Object -FilePath $log
    exit $LASTEXITCODE
}

# Evaluation phases: evaluate.py runs one cell per call, so this file throttles a pool of
# processes itself. Each job is @(label, expected-output-file, args...).
$seeds = 0..11
$outDir = Join-Path $repo "$exp\outputs"
$jobs = New-Object System.Collections.ArrayList
switch ($Phase) {
    "rank" {
        foreach ($d in 7, 8, 9) { foreach ($k in "P", "J-W", "J-A", "J-B") { foreach ($s in $seeds) {
            [void]$jobs.Add(@("rank $k d$d s$s", "exp074_rank_${k}_d${d}_s${s}.json", "--rank", $k, "--depth", $d, "--seed", $s)) } } }
    }
    "cont" {
        foreach ($d in 8, 9) {
            [void]$jobs.Add(@("P3V d$d s0", "exp074_P3V_d${d}_s0.json", "--arm", "P3V", "--depth", $d, "--seed", 0)) }
    }
    "eval" {
        foreach ($d in 7, 8, 9, 11) { foreach ($a in "J3V-W", "J3V-A", "J3V-B") { foreach ($s in $seeds) {
            [void]$jobs.Add(@("$a d$d s$s", "exp074_${a}_d${d}_s${s}.json", "--arm", $a, "--depth", $d, "--seed", $s)) } } }
        foreach ($a in "P3V", "R3V") { foreach ($s in $seeds) {
            [void]$jobs.Add(@("$a d11 s$s", "exp074_${a}_d11_s${s}.json", "--arm", $a, "--depth", 11, "--seed", $s)) } }
    }
    "det" {
        foreach ($a in "J3V-W", "J3V-A") {
            [void]$jobs.Add(@("$a d9 s0", "exp074_${a}_d9_s0.json", "--arm", $a, "--depth", 9, "--seed", 0, "--out-dir", "C:\Users\mlgbr\exp074-det")) }
    }
}

"phase              = $Phase, $($jobs.Count) cells, $Workers workers"
"log                = $log"
""
$running = @()
$failed = 0
$stamp = { param($m) Add-Content -Path $log -Value $m; $m }
foreach ($j in $jobs) {
    $target = if ($Phase -eq "det") { Join-Path "C:\Users\mlgbr\exp074-det" $j[1] } else { Join-Path $outDir $j[1] }
    if ($SkipExisting -and (Test-Path $target)) { continue }
    while (@($running | Where-Object { -not $_.HasExited }).Count -ge $Workers) { Start-Sleep -Seconds 5 }
    foreach ($p in @($running | Where-Object { $_.HasExited })) { if ($p.ExitCode -ne 0) { $failed++ } }
    $running = @($running | Where-Object { -not $_.HasExited })
    & $stamp "start $($j[0])" | Out-Null
    $argList = @("-u", "$exp\evaluate.py") + @($j[2..($j.Count - 1)] | ForEach-Object { "$_" })
    # Read .Handle at once: without it Windows PowerShell 5.1 reports ExitCode as $null after the
    # process exits, $null -ne 0 is true, and every cell counts as failed whatever it did
    # (measured 2026-10-08: the rank phase reported 144 of 144 failed with 144 good records).
    $proc = Start-Process -FilePath $py -ArgumentList $argList -PassThru -NoNewWindow -WorkingDirectory $repo
    $null = $proc.Handle
    $running += $proc
}
while (@($running | Where-Object { -not $_.HasExited }).Count -gt 0) { Start-Sleep -Seconds 5 }
foreach ($p in $running) { if ($p.ExitCode -ne 0) { $failed++ } }
if ($failed -gt 0) { Write-Error "$failed cells exited non-zero; do not aggregate."; exit 1 }
"PHASE OK"
