# EXP-080 launcher for SwizzlesDuo, running from the laptop's MAIN checkout.
#
#   ... -File <repo>\experiments\080_oracle_ordering\launch080.ps1 -Phase check
#   ... -Phase pilot-mint  -Workers 2
#   ... -Phase pilot-train -Workers 3
#   ... -Phase pilot-time
#   ... -Phase mint   -Workers N   (N from the amendment)
#   ... -Phase gate0
#   ... -Phase train  -Workers N
#   ... -Phase search -Workers N
#
# Order (spec section 11): pilot-mint -> pilot-train -> pilot-time -> stop rule and dated
# amendment (sets aggregate.AMENDMENT) -> chain080.ps1: mint -> gate0 -> Gate 0 -> train ->
# Gates V and T, Claims 1 and 2 -> search (only when Claim 1 is CONFIRMED). Every phase but check
# and the pilot phases refuses to run while AMENDMENT is None. Every driver skips finished work,
# and training resumes its checkpoints, so a phase is re-runnable after the 03:31 restart.
#
# NO BACKTICKS IN DOUBLE-QUOTED OUTPUT STRINGS, and NO COMMA-SEPARATED ARGUMENTS across ssh
# (cmd.exe eats commas and exits zero). Seed and arm lists therefore live in this file.
# Launch long phases with OS-level redirection (playbook section 3), never Start-Process from an
# ssh session.

param(
    [Parameter(Mandatory=$true)][ValidateSet("check","pilot-mint","pilot-train","pilot-time","mint","gate0","train","search")][string]$Phase,
    [int]$Workers = 0
)

$Updates = 4000

$repo = "C:\Users\mlgbr\Desktop\Projects\neuromorphic-nn-snn-research-project"
$py   = Join-Path $repo ".venv\Scripts\python.exe"
$src  = Join-Path $repo "src"
$exp  = "experiments\080_oracle_ordering"

if (-not (Test-Path $py)) { Write-Error "interpreter missing: $py"; exit 1 }
Set-Location $repo
$env:PYTHONPATH = $src

$where = & $py -c "import neuromorphic, sys; sys.stdout.write(neuromorphic.__file__)"
if ($LASTEXITCODE -ne 0) { Write-Error "could not import neuromorphic"; exit 1 }
if (-not $where.StartsWith($src, [System.StringComparison]::OrdinalIgnoreCase)) {
    Write-Error "WRONG LIBRARY. neuromorphic resolved to $where, expected under $src."
    exit 1
}

# EXP-075's 12 X encoders and 12 X judges at 4000 updates, EXP-076's 12 depth-11 J4V-X records,
# EXP-075's 2 X pilot records and 2 X pilot judges at 4000 (laptop only, untracked), EXP-079's 2
# proved pilot encoders (laptop only) and its 12 S training records.
$present = & $py -c "import json, sys; from pathlib import Path; o=Path('experiments/075_wide_region/outputs'); p=Path('experiments/075_wide_region/outputs_pilot'); r6=Path('experiments/076_deeper_tree/outputs'); q=Path('experiments/079_judge_trained_deep'); done=lambda d, s: (d/('judge_X_s'+str(s))/'judge.pt').exists() and json.loads((d/('judge_X_s'+str(s))/'state.json').read_text())['update']==4000; e=sum((o/'encoders'/('enc_X_s'+str(s)+'.pt')).exists() for s in range(12)); j=sum(done(o, s) for s in range(12)); r=sum((r6/('exp076_J4V-X_d11_s'+str(s)+'.json')).exists() for s in range(12)); t=sum((p/('exp075_train_X_s'+str(s)+'.json')).exists() for s in (12, 13)); pj=sum(done(p, s) for s in (12, 13)); pe=sum((q/'outputs_pilot'/'encoders'/('enc_X_s'+str(s)+'.pt')).exists() for s in (12, 13)); sr=sum((q/'outputs'/('exp079_train_S_s'+str(s)+'.json')).exists() for s in range(12)); sys.stdout.write(' '.join(str(x) for x in (e, j, r, t, pj, pe, sr)))"
if ($LASTEXITCODE -ne 0 -or $present -ne "12 12 12 2 2 2 12") { Write-Error "expected '12 12 12 2 2 2 12' (X enc, X judges, J4V-X d11, X pilot records, X pilot judges, pilot encoders, S records), found '$present'"; exit 1 }

$head = (& git rev-parse --short HEAD).Trim()
"library                     = $where"
"checkout                    = $head"
"inputs                      = $present"

$alive = @(Get-Process | Where-Object { $_.ProcessName -match "^python" }).Count
if ($alive -gt 2) { Write-Error "$alive python processes already running; refusing to start."; exit 1 }
"pyprocs                     = $alive"

if ($Phase -eq "check") { "CHECK OK"; exit 0 }

$isPilot = $Phase -like "pilot-*"
if (-not $isPilot) {
    $amend = & $py -c "import importlib.util, sys; s=importlib.util.spec_from_file_location('a','$($exp.Replace('\','/'))/aggregate.py'); a=importlib.util.module_from_spec(s); s.loader.exec_module(a); sys.stdout.write(str(a.AMENDMENT))"
    if ($LASTEXITCODE -ne 0 -or $amend -eq "") { Write-Error "could not read AMENDMENT (got '$amend'); refusing to run."; exit 1 }
    if ($amend -match "None") { Write-Error "AMENDMENT is unset ($amend): commit the dated post-pilot amendment before this phase."; exit 1 }
    "amendment                   = $amend"
}

if ($Workers -eq 0) { $Workers = 2 }
if ($Workers -gt 6) { Write-Error "Workers capped at 6; got $Workers."; exit 1 }
$log = Join-Path $repo ("$exp\phase_" + $Phase + ".log")
$outDir   = Join-Path $repo "$exp\outputs"
$pilotDir = Join-Path $repo "$exp\outputs_pilot"
$evalSeeds = @("0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11")

switch ($Phase) {
    "pilot-mint"  { $driver = "$exp\mint.py"; $cliArgs = @("--pilot", "--seeds", "12", "13", "--out-dir", $pilotDir, "--workers", $Workers) }
    "pilot-train" { $driver = "$exp\train.py"; $cliArgs = @("--pilot", "--seeds", "12", "13", "--n-updates", $Updates, "--out-dir", $pilotDir, "--workers", $Workers) }
    "pilot-time"  { $driver = "$exp\evaluate.py"; $cliArgs = @("--time", "--judge-dir", $pilotDir, "--out-dir", $pilotDir) }
    "mint"        { $driver = "$exp\mint.py"; $cliArgs = @("--seeds") + $evalSeeds + @("--out-dir", $outDir, "--workers", $Workers) }
    "gate0"       { $driver = "$exp\evaluate.py"; $cliArgs = @("--gate0", "--seed", "0", "--out-dir", $outDir) }
    "train"       { $driver = "$exp\train.py"; $cliArgs = @("--seeds") + $evalSeeds + @("--n-updates", $Updates, "--out-dir", $outDir, "--workers", $Workers) }
    "search"      { $driver = "$exp\evaluate.py"; $cliArgs = @("--search", "--seeds") + $evalSeeds + @("--out-dir", $outDir, "--workers", $Workers) }
}

# Every phase is one python call that runs its own pool. The venv's python.exe is a ~2 MB
# redirector that starts python3.13.exe, so sampling the launched process reads the redirector
# (EXP-079's 2 MB disclosure). Sample every python process instead, in every phase.
"script                      = $driver $cliArgs"
"log                         = $log"
""
$errLog = $log + ".err"
$argList = @("-u", $driver) + @($cliArgs | ForEach-Object { "$_" })
$proc = Start-Process -FilePath $py -ArgumentList $argList -PassThru -NoNewWindow -WorkingDirectory $repo -RedirectStandardOutput $log -RedirectStandardError $errLog
# Read .Handle at once: without it Windows PowerShell 5.1 reports ExitCode as $null after exit.
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
