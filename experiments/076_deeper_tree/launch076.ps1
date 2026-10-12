# EXP-076 launcher for SwizzlesDuo, running from the laptop's MAIN checkout.
#
#   ... -File C:\Users\mlgbr\launch076.ps1 -Phase check
#   ... -File C:\Users\mlgbr\launch076.ps1 -Phase pilot -Workers 3
#   ... -File C:\Users\mlgbr\launch076.ps1 -Phase cont  -Workers 2
#   ... -File C:\Users\mlgbr\launch076.ps1 -Phase rank  -Workers N -SkipExisting
#   ... -File C:\Users\mlgbr\launch076.ps1 -Phase eval  -Workers N -SkipExisting
#   ... -File C:\Users\mlgbr\launch076.ps1 -Phase det   -Workers 1
#
# Order (spec section 9): pilot -> dated amendment (worker count, schedule; sets
# aggregate.AMENDMENT) -> cont (Gate 0(b)) -> rank -> eval -> det (Gate 0(a), separate dir).
# Every phase but check and pilot refuses to run while aggregate.AMENDMENT is None.
# Evaluation cells take -SkipExisting: re-run the same phase after a Windows Update restart.
#
# NO BACKTICKS IN DOUBLE-QUOTED OUTPUT STRINGS, and NO COMMA-SEPARATED ARGUMENTS across ssh
# (cmd.exe eats commas and exits zero). Arm and seed lists therefore live in this file.

param(
    [Parameter(Mandatory=$true)][ValidateSet("check","pilot","cont","rank","eval","det")][string]$Phase,
    [int]$Workers = 0,
    [switch]$SkipExisting
)

$repo = "C:\Users\mlgbr\Desktop\Projects\neuromorphic-nn-snn-research-project"
$py   = Join-Path $repo ".venv\Scripts\python.exe"
$src  = Join-Path $repo "src"
$exp  = "experiments\076_deeper_tree"

if (-not (Test-Path $py)) { Write-Error "interpreter missing: $py"; exit 1 }
Set-Location $repo
$env:PYTHONPATH = $src

$where = & $py -c "import neuromorphic, sys; sys.stdout.write(neuromorphic.__file__)"
if ($LASTEXITCODE -ne 0) { Write-Error "could not import neuromorphic"; exit 1 }
if (-not $where.StartsWith($src, [System.StringComparison]::OrdinalIgnoreCase)) {
    Write-Error "WRONG LIBRARY. neuromorphic resolved to $where, expected under $src."
    exit 1
}

# The k_goal seam must be present: a stale checkout fails here first.
$hasSeam = & $py -c "import sys, inspect; from neuromorphic.training import lookahead as l; sys.stdout.write(str('k_goal' in inspect.signature(l.evaluate_lookahead).parameters))"
if ($LASTEXITCODE -ne 0 -or $hasSeam -ne "True") { Write-Error "k_goal seam missing (got '$hasSeam'). Sync the repo."; exit 1 }

# EXP-075's 12 X judges at 4000 updates, and its 24 J3V-X records at depths 9 and 11.
$present = & $py -c "import json, sys; from pathlib import Path; o=Path('experiments/075_wide_region/outputs'); j=sum(json.loads((o/('judge_X_s'+str(s))/'state.json').read_text())['update']==4000 and (o/('judge_X_s'+str(s))/'judge.pt').exists() for s in range(12)); r=sum((o/('exp075_J3V-X_d'+str(d)+'_s'+str(s)+'.json')).exists() for d in (9, 11) for s in range(12)); sys.stdout.write(str(j)+' '+str(r))"
if ($LASTEXITCODE -ne 0 -or $present -ne "12 24") { Write-Error "expected 12 trained X judges and 24 J3V-X records, found '$present'"; exit 1 }

$head = (& git rev-parse --short HEAD).Trim()
"library                 = $where"
"checkout                = $head"
"X judges, J3V-X records = $present"

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
$log = Join-Path $repo ("$exp\phase_" + $Phase + ".log")
$outDir   = Join-Path $repo "$exp\outputs"
$pilotDir = Join-Path $repo "$exp\outputs_pilot"
$detDir   = "C:\Users\mlgbr\exp076-det"
$phaseDir = switch ($Phase) { "pilot" { $pilotDir } "det" { $detDir } default { $outDir } }
$seeds = 0..11

# Each job is @(label, expected-output-file, args...). The longest cells go first.
$jobs = New-Object System.Collections.ArrayList
switch ($Phase) {
    "pilot" {
        foreach ($a in "J4V-X", "J3VG4-X") {
            [void]$jobs.Add(@("pilot $a d11 s0", "exp076_${a}_d11_s0.json", "--arm", $a, "--depth", 11, "--seed", 0, "--limit-states", 20)) }
        [void]$jobs.Add(@("pilot rank d11 s0", "exp076_rank_J-X4_d11_s0.json", "--rank", "--depth", 11, "--seed", 0, "--limit-states", 20))
    }
    "cont" {
        foreach ($d in 9, 11) {
            [void]$jobs.Add(@("J3V-X d$d s0", "exp076_J3V-X_d${d}_s0.json", "--arm", "J3V-X", "--depth", $d, "--seed", 0)) }
    }
    "rank" {
        foreach ($d in 11, 9) { foreach ($s in $seeds) {
            [void]$jobs.Add(@("rank d$d s$s", "exp076_rank_J-X4_d${d}_s${s}.json", "--rank", "--depth", $d, "--seed", $s)) } }
    }
    "eval" {
        foreach ($a in "J4V-X", "J3VG4-X") { foreach ($d in 11, 9) { foreach ($s in $seeds) {
            [void]$jobs.Add(@("$a d$d s$s", "exp076_${a}_d${d}_s${s}.json", "--arm", $a, "--depth", $d, "--seed", $s)) } } }
    }
    "det" {
        [void]$jobs.Add(@("J4V-X d9 s0", "exp076_J4V-X_d9_s0.json", "--arm", "J4V-X", "--depth", 9, "--seed", 0))
    }
}

"phase                   = $Phase, $($jobs.Count) cells, $Workers workers, into $phaseDir"
"log                     = $log"
""
$running = @()
$failed = 0
$stamp = { param($m) Add-Content -Path $log -Value $m; $m }
foreach ($j in $jobs) {
    $target = Join-Path $phaseDir $j[1]
    if ($SkipExisting -and (Test-Path $target)) { continue }
    while (@($running | Where-Object { -not $_.HasExited }).Count -ge $Workers) { Start-Sleep -Seconds 5 }
    foreach ($p in @($running | Where-Object { $_.HasExited })) { if ($p.ExitCode -ne 0) { $failed++ } }
    $running = @($running | Where-Object { -not $_.HasExited })
    & $stamp "start $($j[0])" | Out-Null
    $argList = @("-u", "$exp\evaluate.py") + @($j[2..($j.Count - 1)] | ForEach-Object { "$_" }) + @("--out-dir", $phaseDir)
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
