# EXP-075 launcher for SwizzlesDuo, running from the laptop's MAIN checkout.
#
#   ... -File C:\Users\mlgbr\launch075.ps1 -Phase check
#   ... -File C:\Users\mlgbr\launch075.ps1 -Phase pilot-pretrain -Workers 4
#   ... -File C:\Users\mlgbr\launch075.ps1 -Phase pilot-train    -Workers 4
#   ... -File C:\Users\mlgbr\launch075.ps1 -Phase pretrain -Workers 12
#   ... -File C:\Users\mlgbr\launch075.ps1 -Phase train    -Workers N   (N from the amendment)
#   ... -File C:\Users\mlgbr\launch075.ps1 -Phase cont     -Workers 2
#   ... -File C:\Users\mlgbr\launch075.ps1 -Phase rank     -Workers 12
#   ... -File C:\Users\mlgbr\launch075.ps1 -Phase eval     -Workers 12 -SkipExisting
#   ... -File C:\Users\mlgbr\launch075.ps1 -Phase det      -Workers 1
#
# Order (spec section 10): pilot-pretrain -> pilot-train -> dated amendment (Gate P and L
# thresholds, worker count) -> pretrain (Gate P committed) -> train (Gates L and E committed) ->
# cont (Gate 0(b): J3V-W at seed 0, depths 9 and 11, must equal EXP-074's records) -> rank ->
# eval -> det (J3V-X d9 s0 into a SEPARATE directory, Gate 0(a)).
#
# Pretraining and training both resume by themselves: their drivers skip cells whose record
# exists, and train_judge resumes from its banked checkpoint. Re-run the same phase after a
# Windows Update restart. Evaluation phases take -SkipExisting.
#
# NO BACKTICKS IN DOUBLE-QUOTED OUTPUT STRINGS, and NO COMMA-SEPARATED ARGUMENTS across ssh
# (cmd.exe eats commas and exits zero). Arm and seed lists therefore live in this file.

param(
    [Parameter(Mandatory=$true)][ValidateSet("check","pilot-pretrain","pilot-train","pretrain","train","cont","rank","eval","det")][string]$Phase,
    [int]$Workers = 0,
    [switch]$SkipExisting
)

# Spec section 5, fixed.
$Updates = 4000
$SyncEvery = 100
$ProbeEvery = 250
$PilotOutDir = "experiments\075_wide_region\outputs_pilot"

$repo = "C:\Users\mlgbr\Desktop\Projects\neuromorphic-nn-snn-research-project"
$py   = Join-Path $repo ".venv\Scripts\python.exe"
$src  = Join-Path $repo "src"
$exp  = "experiments\075_wide_region"

if (-not (Test-Path $py)) { Write-Error "interpreter missing: $py"; exit 1 }
Set-Location $repo
$env:PYTHONPATH = $src

$where = & $py -c "import neuromorphic, sys; sys.stdout.write(neuromorphic.__file__)"
if ($LASTEXITCODE -ne 0) { Write-Error "could not import neuromorphic"; exit 1 }
if (-not $where.StartsWith($src, [System.StringComparison]::OrdinalIgnoreCase)) {
    Write-Error "WRONG LIBRARY. neuromorphic resolved to $where, expected under $src."
    exit 1
}

# The hidden-width seam must be present: a stale checkout fails here first.
$hasHidden = & $py -c "import sys, inspect; from neuromorphic.training import encoder_pretrain as e; sys.stdout.write(str('hidden' in inspect.signature(e.make_sensory).parameters))"
if ($LASTEXITCODE -ne 0 -or $hasHidden -ne "True") { Write-Error "hidden seam missing (got '$hasHidden'). Sync the repo."; exit 1 }

# 36 policy heads and 12 E1 encoders (EXP-070's path functions; W's judges rebuild through E1),
# and EXP-074's 12 committed W judges.
$present = & $py -c "import importlib.util, sys; s=importlib.util.spec_from_file_location('c','experiments/070_lookahead_existing/cells.py'); c=importlib.util.module_from_spec(s); s.loader.exec_module(c); from pathlib import Path; h=sum(c.head_path(d,x).exists() for d in (7,8,9) for x in c.SEEDS); e=sum(Path(c.published_config(7,x).encoder_state_path).exists() for x in c.SEEDS); w=sum(Path('experiments/074_wide_judge/outputs/judge_W_s'+str(x)+'/judge.pt').exists() for x in c.SEEDS); sys.stdout.write(str(h)+' '+str(e)+' '+str(w))"
if ($LASTEXITCODE -ne 0 -or $present -ne "36 12 12") { Write-Error "expected 36 heads, 12 E1 encoders and 12 W judges, found '$present'"; exit 1 }

$head = (& git rev-parse --short HEAD).Trim()
"library                 = $where"
"checkout                = $head"
"heads, E1, W judges     = $present"

$alive = @(Get-Process | Where-Object { $_.ProcessName -match "^python" }).Count
if ($alive -gt 2) { Write-Error "$alive python processes already running; refusing to start."; exit 1 }
"pyprocs                 = $alive"

if ($Phase -eq "check") { "CHECK OK"; exit 0 }

# Every phase except the pilot and cont reads or produces numbers that decide a claim, and none
# may run before the dated amendment sets BOTH thresholds. Check the exit code too: an import
# failure prints nothing, and "" would not match "None".
if ($Phase -ne "pilot-pretrain" -and $Phase -ne "pilot-train" -and $Phase -ne "cont") {
    $gate = & $py -c "import importlib.util, sys; s=importlib.util.spec_from_file_location('a','$($exp.Replace('\','/'))/aggregate.py'); a=importlib.util.module_from_spec(s); s.loader.exec_module(a); sys.stdout.write(str(a.GATE_P_THRESHOLD)+' | '+str(a.GATE_L_THRESHOLD))"
    if ($LASTEXITCODE -ne 0 -or $gate -eq "") { Write-Error "could not read GATE_P_THRESHOLD and GATE_L_THRESHOLD (got '$gate'); refusing to run."; exit 1 }
    if ($gate -match "None") { Write-Error "a threshold is unset ($gate): commit the dated post-pilot amendment before this phase."; exit 1 }
    "thresholds P | L        = $gate"
}

if ($Workers -eq 0) { $Workers = 4 }
$log = Join-Path $repo ("$exp\phase_" + $Phase + ".log")
$evalSeeds = @("0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11")

# Pretraining and training phases: one driver call, it manages its own worker pool.
$driver = $null
switch ($Phase) {
    "pilot-pretrain" { $driver = "pretrain.py"; $cliArgs = @("--pilot", "--arms", "X", "Y", "--seeds", "12", "13", "--out-dir", (Join-Path $repo $PilotOutDir), "--workers", $Workers) }
    "pilot-train"    { $driver = "train.py";    $cliArgs = @("--pilot", "--arms", "X", "Y", "--seeds", "12", "13", "--n-updates", $Updates, "--sync-every", $SyncEvery, "--probe-every", $ProbeEvery, "--out-dir", (Join-Path $repo $PilotOutDir), "--workers", $Workers) }
    "pretrain"       { $driver = "pretrain.py"; $cliArgs = @("--arms", "X", "Y", "--seeds") + $evalSeeds + @("--workers", $Workers) }
    "train"          { $driver = "train.py";    $cliArgs = @("--arms", "X", "Y", "--seeds") + $evalSeeds + @("--n-updates", $Updates, "--sync-every", $SyncEvery, "--probe-every", $ProbeEvery, "--workers", $Workers) }
}
if ($driver) {
    "script                  = $exp\$driver $cliArgs"
    "log                     = $log"
    ""
    & $py -u "$exp\$driver" @cliArgs 2>&1 | Tee-Object -FilePath $log
    exit $LASTEXITCODE
}

# Evaluation phases: evaluate.py runs one cell per call, so this file throttles a pool of
# processes itself. Each job is @(label, expected-output-file, args...).
$seeds = 0..11
$outDir = Join-Path $repo "$exp\outputs"
$detDir = "C:\Users\mlgbr\exp075-det"
$jobs = New-Object System.Collections.ArrayList
switch ($Phase) {
    "cont" {
        foreach ($d in 9, 11) {
            [void]$jobs.Add(@("J3V-W d$d s0", "exp075_J3V-W_d${d}_s0.json", "--arm", "J3V-W", "--depth", $d, "--seed", 0)) }
    }
    "rank" {
        foreach ($d in 7, 8, 9, 11) { foreach ($k in "J-X", "J-Y") { foreach ($s in $seeds) {
            [void]$jobs.Add(@("rank $k d$d s$s", "exp075_rank_${k}_d${d}_s${s}.json", "--rank", $k, "--depth", $d, "--seed", $s)) } } }
        foreach ($d in 9, 11) { foreach ($s in $seeds) {
            [void]$jobs.Add(@("rank J-W d$d s$s", "exp075_rank_J-W_d${d}_s${s}.json", "--rank", "J-W", "--depth", $d, "--seed", $s)) } }
    }
    "eval" {
        foreach ($d in 7, 8, 9, 11) { foreach ($a in "J3V-X", "J3V-Y") { foreach ($s in $seeds) {
            [void]$jobs.Add(@("$a d$d s$s", "exp075_${a}_d${d}_s${s}.json", "--arm", $a, "--depth", $d, "--seed", $s)) } } }
    }
    "det" {
        [void]$jobs.Add(@("J3V-X d9 s0", "exp075_J3V-X_d9_s0.json", "--arm", "J3V-X", "--depth", 9, "--seed", 0, "--out-dir", $detDir))
    }
}

"phase                   = $Phase, $($jobs.Count) cells, $Workers workers"
"log                     = $log"
""
$running = @()
$failed = 0
$stamp = { param($m) Add-Content -Path $log -Value $m; $m }
foreach ($j in $jobs) {
    $target = if ($Phase -eq "det") { Join-Path $detDir $j[1] } else { Join-Path $outDir $j[1] }
    if ($SkipExisting -and (Test-Path $target)) { continue }
    while (@($running | Where-Object { -not $_.HasExited }).Count -ge $Workers) { Start-Sleep -Seconds 5 }
    foreach ($p in @($running | Where-Object { $_.HasExited })) { if ($p.ExitCode -ne 0) { $failed++ } }
    $running = @($running | Where-Object { -not $_.HasExited })
    & $stamp "start $($j[0])" | Out-Null
    $argList = @("-u", "$exp\evaluate.py") + @($j[2..($j.Count - 1)] | ForEach-Object { "$_" })
    # Read .Handle at once: without it Windows PowerShell 5.1 reports ExitCode as $null after the
    # process exits, and every cell counts as failed whatever it did (EXP-074, 2026-10-08).
    $proc = Start-Process -FilePath $py -ArgumentList $argList -PassThru -NoNewWindow -WorkingDirectory $repo
    $null = $proc.Handle
    $running += $proc
}
while (@($running | Where-Object { -not $_.HasExited }).Count -gt 0) { Start-Sleep -Seconds 5 }
foreach ($p in $running) { if ($p.ExitCode -ne 0) { $failed++ } }
if ($failed -gt 0) { Write-Error "$failed cells exited non-zero; do not aggregate."; exit 1 }
"PHASE OK"
