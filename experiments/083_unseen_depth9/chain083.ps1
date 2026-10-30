# EXP-083 chain: sets (replay, Gate V) -> sets check (Gates V and S; stops on VOID or
# UNRESOLVED and writes that as the verdict) -> search (Gate R first, then depth 10, then 9)
# -> floor -> full aggregate into C:\Users\mlgbr\verdict083.txt. Re-runnable from the top:
# every phase skips finished records. Workers are spec section 8's.
# Launch it with OS-level redirection (playbook section 3):
#   ssh -n laptop 'cmd /c "powershell -NoProfile -ExecutionPolicy Bypass -File <repo>\experiments\083_unseen_depth9\chain083.ps1 > C:\Users\mlgbr\chain083.out 2>&1"'
param([ValidateSet("sets","search","floor")][string]$From = "sets")
$repo = "C:\Users\mlgbr\Desktop\Projects\neuromorphic-nn-snn-research-project"
$py = Join-Path $repo ".venv\Scripts\python.exe"
$l = Join-Path $repo "experiments\083_unseen_depth9\launch083.ps1"
$agg = "experiments\083_unseen_depth9\aggregate.py"
$log = "C:\Users\mlgbr\chain083.log"
$verdictFile = "C:\Users\mlgbr\verdict083.txt"
$workers = @{ "sets" = 4; "search" = 6; "floor" = 4 }
Set-Location $repo
$env:PYTHONPATH = Join-Path $repo "src"
function Say($m) { $line = (Get-Date -Format "yyyy-MM-dd HH:mm:ss") + " " + $m; Add-Content -Path $log -Value $line; $line }
function WriteVerdict($text) {
    Set-Content -Path ($verdictFile + ".tmp") -Value $text
    Move-Item -Force ($verdictFile + ".tmp") $verdictFile
}
Say "chain start from $From at $(git rev-parse --short HEAD)"
$order = @("sets","setscheck","search","floor","aggregate")
$go = $false
foreach ($p in $order) {
    if ($p -eq $From) { $go = $true }
    if (-not $go) { continue }
    if ($p -eq "setscheck") {
        # Fail closed: a crash prints neither line and must stop the chain.
        $g = & $py $agg --stage sets 2>&1 | Out-String
        $code = $LASTEXITCODE
        Say ("sets: " + $g.Trim())
        if ($code -ne 0) { Say "STOP: the sets stage exited $code; fix the records before going on"; exit 1 }
        if ($g -match "VERDICT: (VOID|UNRESOLVED)") { WriteVerdict $g; Say "sets gate failed; verdict written"; exit 3 }
        if (-not ($g -match "SETS: OK")) { Say "STOP: the sets stage printed no SETS line"; exit 2 }
        continue
    }
    if ($p -eq "aggregate") {
        $a = & $py $agg 2>&1 | Out-String
        $code = $LASTEXITCODE
        Say ("aggregate exit " + $code)
        # Written only on success, so the watcher never mistakes an error for a verdict.
        if ($code -ne 0) { Say ("STOP: the aggregate exited " + $code + ": " + $a.Trim()); exit 1 }
        WriteVerdict $a
        continue
    }
    Say "phase $p workers $($workers[$p])"
    & powershell -NoProfile -ExecutionPolicy Bypass -File $l -Phase $p -Workers $workers[$p]
    $code = $LASTEXITCODE
    Say "phase $p exit $code"
    if ($code -ne 0) { Say "STOP: phase $p failed"; exit 1 }
}
Say "chain done"
