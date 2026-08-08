# probe_run.ps1 - is the remote run HEALTHY, or is it orphaned workers that look healthy?
#
# The playbook's own list of instrument bugs includes "worker count alive: a hung pool looks
# identical to a working one". This is the probe that separates them. Two independent signals,
# because either alone can mislead:
#
#   1. PROCESS TREE. launch037.ps1's failure mode 2 kills the SCRIPT and leaves workers
#      computing forever, with nothing left to collect their results. Workers alone look
#      perfect. What matters is whether the ROOT python still has a live powershell parent.
#      `root_parent=GONE` means the run is producing nothing and should be killed and restarted.
#
#   2. EFFECTIVE CORES over a controlled interval. A tree can be intact and deadlocked. This
#      samples inside a single command rather than across calls, because an agent has no
#      reliable clock between tool calls - inferring wall time that way once produced a false
#      "the run is at 15% and possibly broken" when it was at 73%.
#
# Expect roughly 7 effective cores at 10 workers (measured 74.2% utilisation; 16 workers gave
# 43.1% because the laptop is MEMORY-bound at ~920 MB private per worker, not core-bound).
#
# Usage:
#   scp scripts/laptop/probe_run.ps1 laptop:C:/Users/mlgbr/probe_run.ps1
#   ssh -n laptop 'powershell -NoProfile -ExecutionPolicy Bypass -File C:\Users\mlgbr\probe_run.ps1'
#   ssh -n laptop 'powershell ... -File C:\Users\mlgbr\probe_run.ps1 -OutDir "<path>\outputs"'

param(
    [string]$OutDir = '',
    [int]$SampleSeconds = 45
)

$procs = Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python' }
Write-Output ("python_procs=" + $procs.Count)
if ($procs.Count -eq 0) {
    Write-Output "VERDICT=NO_RUN (no python processes; finished, never started, or killed)"
    exit 0
}

# A root is a python process whose parent is NOT itself a python process, i.e. the one the
# launcher started. Everything else is a pool worker.
$ids = $procs.ProcessId
$roots = @($procs | Where-Object { -not ($ids -contains $_.ParentProcessId) })
Write-Output ("roots=" + $roots.Count + " workers=" + ($procs.Count - $roots.Count))

$orphaned = $false
foreach ($r in $roots) {
    $gp = Get-CimInstance Win32_Process -Filter "ProcessId = $($r.ParentProcessId)" -ErrorAction SilentlyContinue
    if ($gp) {
        Write-Output ("root_parent=" + $gp.Name + " pid=" + $gp.ProcessId)
    } else {
        Write-Output "root_parent=GONE"
        $orphaned = $true
    }
}

# Worker count alone proves nothing; measure that they are actually advancing.
$a = ($procs | Measure-Object -Property WorkingSetSize -Sum).Sum
$c1 = (Get-Process | Where-Object { $_.Name -match '^python' } | Measure-Object -Property CPU -Sum).Sum
Start-Sleep -Seconds $SampleSeconds
$c2 = (Get-Process | Where-Object { $_.Name -match '^python' } | Measure-Object -Property CPU -Sum).Sum
$cores = [math]::Round(($c2 - $c1) / $SampleSeconds, 2)
Write-Output ("effective_cores=" + $cores)
Write-Output ("working_set_gb=" + [math]::Round($a / 1GB, 2))

if ($OutDir -ne '' -and (Test-Path $OutDir)) {
    Write-Output ("records=" + (Get-ChildItem $OutDir -Filter *.json).Count)
    $log = Join-Path $OutDir 'run.log'
    if (Test-Path $log) {
        Write-Output ("tracebacks=" + (Select-String -Path $log -Pattern 'Traceback').Count)
        Write-Output ("log_age_min=" + [math]::Round(((Get-Date) - (Get-Item $log).LastWriteTime).TotalMinutes, 1))
    }
}

if ($orphaned) {
    Write-Output "VERDICT=ORPHANED - workers are computing but nothing will collect them. Kill and restart."
    exit 1
} elseif ($cores -lt 1.0) {
    Write-Output "VERDICT=STALLED - tree is intact but barely consuming CPU. Investigate before waiting longer."
    exit 2
} else {
    Write-Output "VERDICT=HEALTHY"
    exit 0
}
