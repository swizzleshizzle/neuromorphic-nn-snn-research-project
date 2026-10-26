# EXP-082 launcher for SwizzlesDuo, running from the laptop's MAIN checkout.
#
#   ... -File <repo>\experiments\082_x_long_budget\launch082.ps1 -Phase check
#   ... -Phase gatec  -Workers 4
#   ... -Phase train  -Workers 4
#   ... -Phase search -Workers 12
#
# Order (spec sections 4 to 8): Gate C -> train (two legs) -> gates stage -> search (depths 9 and
# 11, always) -> aggregate. chain082.ps1 runs them. Every driver skips finished work and training
# resumes its checkpoints, so a phase is re-runnable after the 03:31 restart.
#
# NO BACKTICKS IN DOUBLE-QUOTED OUTPUT STRINGS, and NO COMMA-SEPARATED ARGUMENTS across ssh
# (cmd.exe eats commas and exits zero). Launch long phases with OS-level redirection (playbook
# section 3), never Start-Process from an ssh session.

param(
    [Parameter(Mandatory=$true)][ValidateSet("check","gatec","train","search")][string]$Phase,
    [int]$Workers = 0
)

$repo = "C:\Users\mlgbr\Desktop\Projects\neuromorphic-nn-snn-research-project"
$py   = Join-Path $repo ".venv\Scripts\python.exe"
$src  = Join-Path $repo "src"
$exp  = "experiments\082_x_long_budget"
$caps = @{ "gatec" = 4; "train" = 4; "search" = 12 }

if (-not (Test-Path $py)) { Write-Error "interpreter missing: $py"; exit 1 }
Set-Location $repo
$env:PYTHONPATH = $src

$where = & $py -c "import neuromorphic, sys; sys.stdout.write(neuromorphic.__file__)"
if ($LASTEXITCODE -ne 0) { Write-Error "could not import neuromorphic"; exit 1 }
if (-not $where.StartsWith($src, [System.StringComparison]::OrdinalIgnoreCase)) {
    Write-Error "WRONG LIBRARY. neuromorphic resolved to $where, expected under $src."
    exit 1
}

# EXP-075's 12 X checkpoints at 4000 with all four files, 12 X encoders, 12 X records; EXP-076's
# 12 depth-9 and 12 depth-11 J4V-X records; EXP-081's 12 depth-11 O-Xt16k records.
$present = & $py -c "import json, sys; from pathlib import Path; o=Path('experiments/075_wide_region/outputs'); r6=Path('experiments/076_deeper_tree/outputs'); r81=Path('experiments/081_oxt_long_budget/outputs'); ck=lambda s: o/('judge_X_s'+str(s)); c=sum(all((ck(s)/f).exists() for f in ('judge.pt','jt.pt','opt.pt','state.json')) and json.loads((ck(s)/'state.json').read_text())['update']==4000 for s in range(12)); x=sum((o/'encoders'/('enc_X_s'+str(s)+'.pt')).exists() for s in range(12)); xr=sum((o/('exp075_train_X_s'+str(s)+'.json')).exists() for s in range(12)); j9=sum((r6/('exp076_J4V-X_d9_s'+str(s)+'.json')).exists() for s in range(12)); j11=sum((r6/('exp076_J4V-X_d11_s'+str(s)+'.json')).exists() for s in range(12)); q=sum((r81/('exp081_J4V-O-Xt16k_d11_s'+str(s)+'.json')).exists() for s in range(12)); sys.stdout.write(' '.join(str(v) for v in (c, x, xr, j9, j11, q)))"
if ($LASTEXITCODE -ne 0 -or $present -ne "12 12 12 12 12 12") { Write-Error "expected '12 12 12 12 12 12' (X checkpoints, X encoders, X records, J4V-X d9, J4V-X d11, O-Xt16k d11), found '$present'"; exit 1 }

$head = (& git rev-parse --short HEAD).Trim()
"library                     = $where"
"checkout                    = $head"
"inputs                      = $present"

$alive = @(Get-Process | Where-Object { $_.ProcessName -match "^python" }).Count
if ($alive -gt 2) { Write-Error "$alive python processes already running; refusing to start."; exit 1 }
"pyprocs                     = $alive"

if ($Phase -eq "check") { "CHECK OK"; exit 0 }

if ($Workers -eq 0) { $Workers = 2 }
if ($Workers -gt $caps[$Phase]) { Write-Error "Workers capped at $($caps[$Phase]) for $Phase; got $Workers."; exit 1 }
$log = Join-Path $repo ("$exp\phase_" + $Phase + ".log")
$outDir = Join-Path $repo "$exp\outputs"
$evalSeeds = @("0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11")

switch ($Phase) {
    "gatec"  { $driver = "$exp\gatec.py" }
    "train"  { $driver = "$exp\train.py" }
    "search" { $driver = "$exp\evaluate.py" }
}
$cliArgs = @("--seeds") + $evalSeeds + @("--out-dir", $outDir, "--workers", $Workers)

# Sample every python process: the venv's python.exe is a redirector (EXP-079's disclosure).
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
