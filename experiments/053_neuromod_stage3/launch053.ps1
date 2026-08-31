# EXP-053 launcher for SwizzlesDuo. scp this over and run it with:
#   powershell -NoProfile -ExecutionPolicy Bypass -File C:\Users\mlgbr\launch053.ps1 -Arm B
#
# Quoting through cmd.exe eats trailing backslashes and interprets | before PowerShell sees
# it, so this exists as a FILE rather than as an inline command.
param(
    [Parameter(Mandatory=$true)][ValidateSet("pilot","B","G","R")][string]$Arm,
    [int]$Workers = 6
)

$repo = "C:\Users\mlgbr\Desktop\Projects\neuromorphic-nn-snn-research-project"
Set-Location $repo

# Refuse to share the machine. Worker processes appear as python3.13.exe, not python.exe,
# so match on ^python or this guard misses them completely.
$alive = @(Get-Process | Where-Object { $_.ProcessName -match "^python" }).Count
if ($alive -gt 2) {
    Write-Error "$alive python processes already running; refusing to start."
    exit 1
}

$log = Join-Path $repo "experiments\053_neuromod_stage3\phase_$Arm.log"
if ($Arm -eq "pilot") {
    $script = "experiments\053_neuromod_stage3\pilot_critic_lr.py"
    $args = @("--workers", $Workers)
} else {
    $script = "experiments\053_neuromod_stage3\run.py"
    $args = @("--arm", $Arm, "--workers", $Workers)
}

# -u so the log is not fully buffered. Tee-Object so the record survives an ssh drop:
# Windows has no SIGHUP semantics, so a dropped connection does not kill the run.
& ".venv\Scripts\python.exe" -u $script @args 2>&1 | Tee-Object -FilePath $log
