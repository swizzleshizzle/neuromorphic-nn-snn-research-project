# EXP-081 launcher for SwizzlesDuo, running from the laptop's MAIN checkout.
#
#   ... -File <repo>\experiments\081_oxt_long_budget\launch081.ps1 -Phase check
#   ... -Phase mint   -Workers 6
#   ... -Phase gatec  -Workers 6
#   ... -Phase train  -Workers 6
#   ... -Phase search -Workers 6
#
# Order (spec sections 4 to 9): mint -> Gate C -> train (two legs) -> gates and claims -> search
# (only when Claim 1 is CONFIRMED). chain081.ps1 runs them. Every driver skips finished work and
# training resumes its checkpoints, so a phase is re-runnable after the 03:31 restart.
#
# NO BACKTICKS IN DOUBLE-QUOTED OUTPUT STRINGS, and NO COMMA-SEPARATED ARGUMENTS across ssh
# (cmd.exe eats commas and exits zero). Launch long phases with OS-level redirection (playbook
# section 3), never Start-Process from an ssh session.

param(
    [Parameter(Mandatory=$true)][ValidateSet("check","mint","gatec","train","search")][string]$Phase,
    [int]$Workers = 0
)

$repo = "C:\Users\mlgbr\Desktop\Projects\neuromorphic-nn-snn-research-project"
$py   = Join-Path $repo ".venv\Scripts\python.exe"
$src  = Join-Path $repo "src"
$exp  = "experiments\081_oxt_long_budget"

if (-not (Test-Path $py)) { Write-Error "interpreter missing: $py"; exit 1 }
Set-Location $repo
$env:PYTHONPATH = $src

$where = & $py -c "import neuromorphic, sys; sys.stdout.write(neuromorphic.__file__)"
if ($LASTEXITCODE -ne 0) { Write-Error "could not import neuromorphic"; exit 1 }
if (-not $where.StartsWith($src, [System.StringComparison]::OrdinalIgnoreCase)) {
    Write-Error "WRONG LIBRARY. neuromorphic resolved to $where, expected under $src."
    exit 1
}

# EXP-080's 12 mints, 12 O-Xt checkpoints at 4000 with opt.pt, 12 O-Xt records; EXP-075's 12 X
# encoders and 12 X records; EXP-076's 12 depth-11 J4V-X records.
$present = & $py -c "import json, sys; from pathlib import Path; e=Path('experiments/080_oracle_ordering/outputs'); o=Path('experiments/075_wide_region/outputs'); r6=Path('experiments/076_deeper_tree/outputs'); ck=lambda s: e/('judge_O-Xt_s'+str(s)); m=sum((e/'mint'/('mint_s'+str(s)+'.npz')).exists() for s in range(12)); c=sum((ck(s)/'opt.pt').exists() and (ck(s)/'judge.pt').exists() and json.loads((ck(s)/'state.json').read_text())['update']==4000 for s in range(12)); r=sum((e/('exp080_train_O-Xt_s'+str(s)+'.json')).exists() for s in range(12)); x=sum((o/'encoders'/('enc_X_s'+str(s)+'.pt')).exists() for s in range(12)); xr=sum((o/('exp075_train_X_s'+str(s)+'.json')).exists() for s in range(12)); j=sum((r6/('exp076_J4V-X_d11_s'+str(s)+'.json')).exists() for s in range(12)); sys.stdout.write(' '.join(str(v) for v in (m, c, r, x, xr, j)))"
if ($LASTEXITCODE -ne 0 -or $present -ne "12 12 12 12 12 12") { Write-Error "expected '12 12 12 12 12 12' (E80 mints, E80 O-Xt checkpoints, E80 O-Xt records, X encoders, X records, J4V-X d11), found '$present'"; exit 1 }

$head = (& git rev-parse --short HEAD).Trim()
"library                     = $where"
"checkout                    = $head"
"inputs                      = $present"

$alive = @(Get-Process | Where-Object { $_.ProcessName -match "^python" }).Count
if ($alive -gt 2) { Write-Error "$alive python processes already running; refusing to start."; exit 1 }
"pyprocs                     = $alive"

if ($Phase -eq "check") { "CHECK OK"; exit 0 }

if ($Workers -eq 0) { $Workers = 2 }
if ($Workers -gt 6) { Write-Error "Workers capped at 6; got $Workers."; exit 1 }
$log = Join-Path $repo ("$exp\phase_" + $Phase + ".log")
$outDir = Join-Path $repo "$exp\outputs"
$evalSeeds = @("0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11")

switch ($Phase) {
    "mint"   { $driver = "$exp\mint.py" }
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
