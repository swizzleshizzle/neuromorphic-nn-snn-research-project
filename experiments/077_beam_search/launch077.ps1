# EXP-077 launcher for SwizzlesDuo, running from the laptop's MAIN checkout.
#
#   ... -File C:\Users\mlgbr\launch077.ps1 -Phase check
#   ... -File C:\Users\mlgbr\launch077.ps1 -Phase pilot -Workers 3
#   ... -File C:\Users\mlgbr\launch077.ps1 -Phase cont  -Workers 2
#   ... -File C:\Users\mlgbr\launch077.ps1 -Phase eval  -Workers N -SkipExisting
#   ... -File C:\Users\mlgbr\launch077.ps1 -Phase det   -Workers 1
#
# Order (spec section 9): pilot -> dated amendment (worker count, schedule; sets
# aggregate.AMENDMENT) -> cont (Gate 0(b)) -> eval -> det (Gate 0(a), into outputs_det).
# Every phase but check and pilot refuses to run while aggregate.AMENDMENT is None.
# Evaluation cells take -SkipExisting: re-run the same phase after a Windows Update restart.
#
# Memory: the venv's python.exe is a redirector; the worker is its python3.13 child (EXP-076
# section 12). Peak commit is therefore read from the child process, not the redirector.
#
# NO BACKTICKS IN DOUBLE-QUOTED OUTPUT STRINGS, and NO COMMA-SEPARATED ARGUMENTS across ssh
# (cmd.exe eats commas and exits zero). Arm and seed lists therefore live in this file.

param(
    [Parameter(Mandatory=$true)][ValidateSet("check","pilot","cont","eval","det")][string]$Phase,
    [int]$Workers = 0,
    [switch]$SkipExisting
)

$repo = "C:\Users\mlgbr\Desktop\Projects\neuromorphic-nn-snn-research-project"
$py   = Join-Path $repo ".venv\Scripts\python.exe"
$src  = Join-Path $repo "src"
$exp  = "experiments\077_beam_search"

if (-not (Test-Path $py)) { Write-Error "interpreter missing: $py"; exit 1 }
Set-Location $repo
$env:PYTHONPATH = $src

$where = & $py -c "import neuromorphic, sys; sys.stdout.write(neuromorphic.__file__)"
if ($LASTEXITCODE -ne 0) { Write-Error "could not import neuromorphic"; exit 1 }
if (-not $where.StartsWith($src, [System.StringComparison]::OrdinalIgnoreCase)) {
    Write-Error "WRONG LIBRARY. neuromorphic resolved to $where, expected under $src."
    exit 1
}

# The beam_width seam must be present: a stale checkout fails here first.
$hasSeam = & $py -c "import sys, inspect; from neuromorphic.training import lookahead as l; sys.stdout.write(str('beam_width' in inspect.signature(l.evaluate_lookahead).parameters))"
if ($LASTEXITCODE -ne 0 -or $hasSeam -ne "True") { Write-Error "beam_width seam missing (got '$hasSeam'). Sync the repo."; exit 1 }

# EXP-075's 12 X judges at 4000 updates, and EXP-076's 24 J4V-X records at depths 9 and 11.
$present = & $py -c "import json, sys; from pathlib import Path; o=Path('experiments/075_wide_region/outputs'); o6=Path('experiments/076_deeper_tree/outputs'); j=sum(json.loads((o/('judge_X_s'+str(s))/'state.json').read_text())['update']==4000 and (o/('judge_X_s'+str(s))/'judge.pt').exists() for s in range(12)); r=sum((o6/('exp076_J4V-X_d'+str(d)+'_s'+str(s)+'.json')).exists() for d in (9, 11) for s in range(12)); sys.stdout.write(str(j)+' '+str(r))"
if ($LASTEXITCODE -ne 0 -or $present -ne "12 24") { Write-Error "expected 12 trained X judges and 24 J4V-X records, found '$present'"; exit 1 }

$head = (& git rev-parse --short HEAD).Trim()
"library                 = $where"
"checkout                = $head"
"X judges, J4V-X records = $present"

$alive = @(Get-Process | Where-Object { $_.ProcessName -match "^python" }).Count
if ($alive -gt 2) { Write-Error "$alive python processes already running; refusing to start."; exit 1 }
"pyprocs                 = $alive"

if ($Phase -eq "check") { "CHECK OK"; exit 0 }

# Every phase but the pilot produces numbers that decide a claim, and none may run before the
# dated amendment. Check the exit code too: an import failure prints nothing.
if ($Phase -ne "pilot") {
    $amend = & $py -c "import importlib.util, sys; s=importlib.util.spec_from_file_location('a','$($exp.Replace('\','/'))/aggregate.py'); a=importlib.util.module_from_spec(s); s.loader.exec_module(a); sys.stdout.write(str(a.AMENDMENT))"
    if ($LASTEXITCODE -ne 0 -or $amend -eq "") { Write-Error "could not read AMENDMENT (got '$amend'); refusing to run."; exit 1 }
    if ($amend -eq "None") { Write-Error "AMENDMENT is unset: commit the dated post-pilot amendment before this phase."; exit 1 }
    "amendment               = $amend"
}

if ($Workers -eq 0) { $Workers = 4 }
if ($Workers -gt 6) { Write-Error "Workers capped at 6 (spec section 6); got $Workers."; exit 1 }
$log = Join-Path $repo ("$exp\phase_" + $Phase + ".log")
$outDir   = Join-Path $repo "$exp\outputs"
$pilotDir = Join-Path $repo "$exp\outputs_pilot"
$detDir   = Join-Path $repo "$exp\outputs_det"
$phaseDir = switch ($Phase) { "pilot" { $pilotDir } "det" { $detDir } default { $outDir } }
$seeds = 0..11

# Each job is @(label, expected-output-file, args...). The longest cells go first.
$jobs = New-Object System.Collections.ArrayList
switch ($Phase) {
    "pilot" {
        foreach ($a in "J4VG5-X", "B5-X") {
            [void]$jobs.Add(@("pilot $a d11 s0", "exp077_${a}_d11_s0.json", "--arm", $a, "--depth", 11, "--seed", 0, "--limit-states", 20)) }
    }
    "cont" {
        [void]$jobs.Add(@("J4V-X d11 s0", "exp077_J4V-X_d11_s0.json", "--arm", "J4V-X", "--depth", 11, "--seed", 0))
        [void]$jobs.Add(@("J4V-X d9 s0", "exp077_J4V-X_d9_s0.json", "--arm", "J4V-X", "--depth", 9, "--seed", 0))
    }
    "eval" {
        foreach ($a in "B5-X", "J4VG5-X") { foreach ($d in 11, 9) { foreach ($s in $seeds) {
            [void]$jobs.Add(@("$a d$d s$s", "exp077_${a}_d${d}_s${s}.json", "--arm", $a, "--depth", $d, "--seed", $s)) } } }
    }
    "det" {
        [void]$jobs.Add(@("B5-X d9 s0", "exp077_B5-X_d9_s0.json", "--arm", "B5-X", "--depth", 9, "--seed", 0))
    }
}

"phase                   = $Phase, $($jobs.Count) cells, $Workers workers, into $phaseDir"
"log                     = $log"
""
$running = @()
$failed = 0
$peak = @{}
$labels = @{}
$maxPeak = 0
$stamp = { param($m) Add-Content -Path $log -Value $m; $m }
# Sample committed memory (PrivateMemorySize64) of each live cell AND its python3.13 child, log the peak at exit.
$sample = {
    foreach ($p in $running) {
        try {
            if (-not $p.HasExited) {
                $kids = @(Get-CimInstance Win32_Process -Filter "ParentProcessId=$($p.Id)" -ErrorAction SilentlyContinue)
                $ids = @($p.Id) + @($kids | ForEach-Object { $_.ProcessId })
                foreach ($id in $ids) {
                    $q = Get-Process -Id $id -ErrorAction SilentlyContinue
                    if ($q) {
                        $m = [int]($q.PrivateMemorySize64 / 1MB)
                        if ($m -gt $peak[$p.Id]) { $peak[$p.Id] = $m }
                    }
                }
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
