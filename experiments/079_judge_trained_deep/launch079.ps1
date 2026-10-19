# EXP-079 launcher for SwizzlesDuo, running from the laptop's MAIN checkout.
#
#   ... -File C:\Users\mlgbr\launch079.ps1 -Phase check
#   ... -File C:\Users\mlgbr\launch079.ps1 -Phase pilot-encoders
#   ... -File C:\Users\mlgbr\launch079.ps1 -Phase pilot-pretrain -Workers 2   (only if pilot-encoders refused)
#   ... -File C:\Users\mlgbr\launch079.ps1 -Phase pilot-train    -Workers 2
#   ... -File C:\Users\mlgbr\launch079.ps1 -Phase cont  -Workers 2
#   ... -File C:\Users\mlgbr\launch079.ps1 -Phase train -Workers N   (N from the amendment)
#   ... -File C:\Users\mlgbr\launch079.ps1 -Phase eval  -Workers N -SkipExisting
#   ... -File C:\Users\mlgbr\launch079.ps1 -Phase det   -Workers 1
#   ... -File C:\Users\mlgbr\launch079.ps1 -Phase rank  -Workers N -SkipExisting
#
# Order (spec section 10): pilot-encoders (or pilot-pretrain) -> pilot-train -> stop rule and
# dated amendment (sets aggregate.AMENDMENT and aggregate.GATE_L_THRESHOLD) -> chain079.ps1:
# cont (Gate 0(b)) -> train -> Gates L and E -> eval (depth 11 first) -> det (Gate 0(a), into
# outputs_det) -> rank. Every phase but check and the pilot phases refuses to run while either
# amendment field is None. Training resumes by itself (the driver skips finished seeds and
# train_judge resumes its checkpoint); evaluation phases take -SkipExisting.
#
# NO BACKTICKS IN DOUBLE-QUOTED OUTPUT STRINGS, and NO COMMA-SEPARATED ARGUMENTS across ssh
# (cmd.exe eats commas and exits zero). Arm and seed lists therefore live in this file.
# Launch long phases with OS-level redirection (playbook section 3), never Start-Process
# from an ssh session.

param(
    [Parameter(Mandatory=$true)][ValidateSet("check","pilot-encoders","pilot-pretrain","pilot-train","train","cont","eval","det","rank")][string]$Phase,
    [int]$Workers = 0,
    [int]$MaxLen = 40,
    [switch]$SkipExisting
)

# Spec sections 3 and 4, fixed. The walk length is pinned: any other value is refused.
if ($MaxLen -ne 40) { Write-Error "MaxLen is pinned at 40 (spec section 4); got $MaxLen."; exit 1 }
$Updates = 4000
$SyncEvery = 100
$ProbeEvery = 250

$repo = "C:\Users\mlgbr\Desktop\Projects\neuromorphic-nn-snn-research-project"
$py   = Join-Path $repo ".venv\Scripts\python.exe"
$src  = Join-Path $repo "src"
$exp  = "experiments\079_judge_trained_deep"

if (-not (Test-Path $py)) { Write-Error "interpreter missing: $py"; exit 1 }
Set-Location $repo
$env:PYTHONPATH = $src

$where = & $py -c "import neuromorphic, sys; sys.stdout.write(neuromorphic.__file__)"
if ($LASTEXITCODE -ne 0) { Write-Error "could not import neuromorphic"; exit 1 }
if (-not $where.StartsWith($src, [System.StringComparison]::OrdinalIgnoreCase)) {
    Write-Error "WRONG LIBRARY. neuromorphic resolved to $where, expected under $src."
    exit 1
}

# EXP-075's 12 X encoders and 12 X judges at 4000 updates, EXP-076's 24 J4V-X records, and
# EXP-075's two X pilot training records (the pilot comparison and the encoder proof).
$present = & $py -c "import json, sys; from pathlib import Path; o=Path('experiments/075_wide_region/outputs'); p=Path('experiments/075_wide_region/outputs_pilot'); r6=Path('experiments/076_deeper_tree/outputs'); e=sum((o/'encoders'/('enc_X_s'+str(s)+'.pt')).exists() for s in range(12)); j=sum((o/('judge_X_s'+str(s))/'judge.pt').exists() and json.loads((o/('judge_X_s'+str(s))/'state.json').read_text())['update']==4000 for s in range(12)); r=sum((r6/('exp076_J4V-X_d'+str(d)+'_s'+str(s)+'.json')).exists() for d in (9, 11) for s in range(12)); t=sum((p/('exp075_train_X_s'+str(s)+'.json')).exists() for s in (12, 13)); sys.stdout.write(str(e)+' '+str(j)+' '+str(r)+' '+str(t))"
if ($LASTEXITCODE -ne 0 -or $present -ne "12 12 24 2") { Write-Error "expected 12 X encoders, 12 X judges, 24 J4V-X records and 2 X pilot records, found '$present'"; exit 1 }

$head = (& git rev-parse --short HEAD).Trim()
"library                     = $where"
"checkout                    = $head"
"X enc, X judges, J4V-X, pilot = $present"

$alive = @(Get-Process | Where-Object { $_.ProcessName -match "^python" }).Count
if ($alive -gt 2) { Write-Error "$alive python processes already running; refusing to start."; exit 1 }
"pyprocs                     = $alive"

if ($Phase -eq "check") { "CHECK OK"; exit 0 }

# Every phase after the pilot reads or produces numbers that decide a claim, and none may run
# before the dated amendment sets BOTH fields. Check the exit code too: an import failure
# prints nothing, and "" would not match "None".
$isPilot = $Phase -like "pilot-*"
if (-not $isPilot) {
    $amend = & $py -c "import importlib.util, sys; s=importlib.util.spec_from_file_location('a','$($exp.Replace('\','/'))/aggregate.py'); a=importlib.util.module_from_spec(s); s.loader.exec_module(a); sys.stdout.write(str(a.AMENDMENT)+' | '+str(a.GATE_L_THRESHOLD))"
    if ($LASTEXITCODE -ne 0 -or $amend -eq "") { Write-Error "could not read AMENDMENT and GATE_L_THRESHOLD (got '$amend'); refusing to run."; exit 1 }
    if ($amend -match "None") { Write-Error "an amendment field is unset ($amend): commit the dated post-pilot amendment before this phase."; exit 1 }
    "amendment | Gate L          = $amend"
}

if ($Workers -eq 0) { $Workers = 2 }
if ($Workers -gt 6) { Write-Error "Workers capped at 6; got $Workers."; exit 1 }
$log = Join-Path $repo ("$exp\phase_" + $Phase + ".log")
$outDir   = Join-Path $repo "$exp\outputs"
$pilotDir = Join-Path $repo "$exp\outputs_pilot"
$detDir   = Join-Path $repo "$exp\outputs_det"
$evalSeeds = @("0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11")

if ($Phase -eq "pilot-encoders") {
    & $py "$exp\cells.py" prepare-pilot-encoders 2>&1 | Tee-Object -FilePath $log
    exit $LASTEXITCODE
}

# Driver phases: one python call that runs its own worker pool. Its output goes to the phase
# log, and every python process is sampled for committed memory while it runs, because the
# driver's workers are not this script's children.
$driver = $null
switch ($Phase) {
    "pilot-pretrain" { $driver = "experiments\075_wide_region\pretrain.py"; $cliArgs = @("--pilot", "--arms", "X", "--seeds", "12", "13", "--out-dir", $pilotDir, "--workers", $Workers) }
    "pilot-train"    { $driver = "$exp\train.py"; $cliArgs = @("--pilot", "--seeds", "12", "13", "--n-updates", $Updates, "--max-len", $MaxLen, "--sync-every", $SyncEvery, "--probe-every", $ProbeEvery, "--out-dir", $pilotDir, "--workers", $Workers) }
    "train"          { $driver = "$exp\train.py"; $cliArgs = @("--seeds") + $evalSeeds + @("--n-updates", $Updates, "--max-len", $MaxLen, "--sync-every", $SyncEvery, "--probe-every", $ProbeEvery, "--out-dir", $outDir, "--workers", $Workers) }
}
if ($driver) {
    "script                      = $driver $cliArgs"
    "log                         = $log"
    ""
    $errLog = $log + ".err"
    $argList = @("-u", $driver) + @($cliArgs | ForEach-Object { "$_" })
    $proc = Start-Process -FilePath $py -ArgumentList $argList -PassThru -NoNewWindow -WorkingDirectory $repo -RedirectStandardOutput $log -RedirectStandardError $errLog
    $null = $proc.Handle
    $maxEach = 0
    $maxTotal = 0
    while (-not $proc.HasExited) {
        $ms = @(Get-Process | Where-Object { $_.ProcessName -match "^python" } | ForEach-Object { [int]($_.PrivateMemorySize64 / 1MB) })
        if ($ms.Count -gt 0) {
            $each = ($ms | Measure-Object -Maximum).Maximum
            $tot = ($ms | Measure-Object -Sum).Sum
            if ($each -gt $maxEach) { $maxEach = $each }
            if ($tot -gt $maxTotal) { $maxTotal = $tot }
        }
        Start-Sleep -Seconds 10
    }
    Get-Content $log
    if (Test-Path $errLog) { Get-Content $errLog }
    "max peak commit MB per python process = $maxEach"
    "max total commit MB across python processes = $maxTotal"
    if ($proc.ExitCode -ne 0) { Write-Error "driver exited $($proc.ExitCode); do not aggregate."; exit 1 }
    "PHASE OK"
    exit 0
}

# Evaluation phases: evaluate.py runs one cell per call, so this file throttles a pool of
# processes itself. Each job is @(label, expected-output-file, args...).
$phaseDir = switch ($Phase) { "det" { $detDir } default { $outDir } }
$seeds = 0..11
$jobs = New-Object System.Collections.ArrayList
switch ($Phase) {
    "cont" {
        foreach ($d in 9, 11) {
            [void]$jobs.Add(@("J4V-X d$d s0", "exp079_J4V-X_d${d}_s0.json", "--arm", "J4V-X", "--depth", $d, "--seed", 0)) }
        [void]$jobs.Add(@("gatel X s0", "exp079_gatel_X_s0.json", "--gatel-x", "--seed", 0))
    }
    "eval" {
        foreach ($d in 11, 9) { foreach ($s in $seeds) {
            [void]$jobs.Add(@("J4V-S d$d s$s", "exp079_J4V-S_d${d}_s${s}.json", "--arm", "J4V-S", "--depth", $d, "--seed", $s)) } }
    }
    "det" {
        [void]$jobs.Add(@("J4V-S d9 s0", "exp079_J4V-S_d9_s0.json", "--arm", "J4V-S", "--depth", 9, "--seed", 0))
    }
    "rank" {
        foreach ($d in 11, 9) { foreach ($s in $seeds) {
            [void]$jobs.Add(@("rank d$d s$s", "exp079_rank_J-S4_d${d}_s${s}.json", "--rank", "--depth", $d, "--seed", $s)) } }
    }
}

"phase                       = $Phase, $($jobs.Count) cells, $Workers workers, into $phaseDir"
"log                         = $log"
""
$running = @()
$failed = 0
$peak = @{}
$labels = @{}
$maxPeak = 0
$stamp = { param($m) Add-Content -Path $log -Value $m; $m }
# Sample each live child's committed memory (PrivateMemorySize64) and log the peak at exit.
$sample = {
    foreach ($p in $running) {
        try {
            if (-not $p.HasExited) {
                $p.Refresh()
                $m = [int]($p.PrivateMemorySize64 / 1MB)
                if ($m -gt $peak[$p.Id]) { $peak[$p.Id] = $m }
            }
        } catch { }
    }
}
$reap = {
    foreach ($p in $running) {
        if ($p.HasExited -and $labels.ContainsKey($p.Id)) {
            $v = [int]$peak[$p.Id]
            & $stamp "peak commit MB $($labels[$p.Id]) $v" | Out-Null
            if ($v -gt $script:maxPeak) { $script:maxPeak = $v }
            $labels.Remove($p.Id)
        }
    }
}
foreach ($j in $jobs) {
    $target = Join-Path $phaseDir $j[1]
    if ($SkipExisting -and (Test-Path $target)) { continue }
    while (@($running | Where-Object { -not $_.HasExited }).Count -ge $Workers) { & $sample; & $reap; Start-Sleep -Seconds 5 }
    & $reap
    foreach ($p in @($running | Where-Object { $_.HasExited })) { if ($p.ExitCode -ne 0) { $failed++ } }
    $running = @($running | Where-Object { -not $_.HasExited })
    & $stamp "start $($j[0])" | Out-Null
    $argList = @("-u", "$exp\evaluate.py") + @($j[2..($j.Count - 1)] | ForEach-Object { "$_" }) + @("--out-dir", $phaseDir)
    # Read .Handle at once: without it Windows PowerShell 5.1 reports ExitCode as $null after the
    # process exits, and every cell counts as failed whatever it did (EXP-074, 2026-10-08).
    $proc = Start-Process -FilePath $py -ArgumentList $argList -PassThru -NoNewWindow -WorkingDirectory $repo
    $null = $proc.Handle
    $running += $proc
    $labels[$proc.Id] = $j[0]
    $peak[$proc.Id] = 0
}
while (@($running | Where-Object { -not $_.HasExited }).Count -gt 0) { & $sample; & $reap; Start-Sleep -Seconds 5 }
& $reap
"max peak commit MB = $maxPeak"
foreach ($p in $running) { if ($p.ExitCode -ne 0) { $failed++ } }
if ($failed -gt 0) { Write-Error "$failed cells exited non-zero; do not aggregate."; exit 1 }
"PHASE OK"
