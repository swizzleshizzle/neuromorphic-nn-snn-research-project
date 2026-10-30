# EXP-083 launcher for SwizzlesDuo, running from the laptop's MAIN checkout.
#
#   ... -File <repo>\experiments\083_unseen_depth9\launch083.ps1 -Phase check
#   ... -Phase sets   -Workers 4
#   ... -Phase search -Workers 6
#   ... -Phase floor  -Workers 4
#
# Order (spec sections 5 and 8): sets (replay, Gate V) -> sets check -> search (Gate R first)
# -> floor -> aggregate. chain083.ps1 runs them. Every driver skips finished records, so a phase
# is re-runnable after the 03:31 restart.
#
# NO BACKTICKS IN DOUBLE-QUOTED OUTPUT STRINGS, and NO COMMA-SEPARATED ARGUMENTS across ssh
# (cmd.exe eats commas and exits zero). Launch the chain with OS-level redirection (playbook
# section 3), never Start-Process from an ssh session.

param(
    [Parameter(Mandatory=$true)][ValidateSet("check","sets","search","floor")][string]$Phase,
    [int]$Workers = 0
)

$repo = "C:\Users\mlgbr\Desktop\Projects\neuromorphic-nn-snn-research-project"
$py   = Join-Path $repo ".venv\Scripts\python.exe"
$src  = Join-Path $repo "src"
$exp  = "experiments\083_unseen_depth9"
$caps = @{ "sets" = 4; "search" = 6; "floor" = 4 }

if (-not (Test-Path $py)) { Write-Error "interpreter missing: $py"; exit 1 }
Set-Location $repo
$env:PYTHONPATH = $src

$where = & $py -c "import neuromorphic, sys; sys.stdout.write(neuromorphic.__file__)"
if ($LASTEXITCODE -ne 0) { Write-Error "could not import neuromorphic"; exit 1 }
if (-not $where.StartsWith($src, [System.StringComparison]::OrdinalIgnoreCase)) {
    Write-Error "WRONG LIBRARY. neuromorphic resolved to $where, expected under $src."
    exit 1
}

# EXP-075's 12 X checkpoints at 4000, EXP-082's 12 X16k checkpoints at 16,000 (judge.pt and
# state.json each), and EXP-082's 12 depth-9 J4V-X16k records (Gate R reads seed 0's).
$present = & $py -c "import json, sys; from pathlib import Path; a=Path('experiments/075_wide_region/outputs'); b=Path('experiments/082_x_long_budget/outputs'); ok=lambda r, s, u: (r/('judge_X_s'+str(s))/'judge.pt').exists() and (r/('judge_X_s'+str(s))/'state.json').exists() and json.loads((r/('judge_X_s'+str(s))/'state.json').read_text())['update']==u; x=sum(ok(a, s, 4000) for s in range(12)); k=sum(ok(b, s, 16000) for s in range(12)); r=sum((b/('exp082_J4V-X16k_d9_s'+str(s)+'.json')).exists() for s in range(12)); sys.stdout.write(' '.join(str(v) for v in (x, k, r)))"
if ($LASTEXITCODE -ne 0 -or $present -ne "12 12 12") { Write-Error "expected '12 12 12' (X at 4000, X16k at 16000, EXP-082 d9 records), found '$present'"; exit 1 }

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
    "sets"   { $driver = "$exp\sets.py" }
    "search" { $driver = "$exp\evaluate.py" }
    "floor"  { $driver = "$exp\floor.py" }
}
$cliArgs = @("--seeds") + $evalSeeds + @("--out-dir", $outDir, "--workers", $Workers)

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
