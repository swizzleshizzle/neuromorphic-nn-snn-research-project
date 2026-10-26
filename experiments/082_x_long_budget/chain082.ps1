# EXP-082 chain: Gate C -> train (two legs) -> gates stage (record checks, Gate T, Claim 2) ->
# search (depths 9 and 11; always run once the gates stage passes, spec section 8) -> full
# aggregate into C:\Users\mlgbr\verdict082.txt. Re-runnable from the top: every phase skips
# finished work and training resumes its checkpoints. Workers are spec section 10's.
# Launch it with OS-level redirection (playbook section 3):
#   ssh -n laptop 'cmd /c "powershell -NoProfile -ExecutionPolicy Bypass -File <repo>\experiments\082_x_long_budget\chain082.ps1 > C:\Users\mlgbr\chain082.out 2>&1"'
param([ValidateSet("gatec","train","gates")][string]$From = "gatec")
$repo = "C:\Users\mlgbr\Desktop\Projects\neuromorphic-nn-snn-research-project"
$py = Join-Path $repo ".venv\Scripts\python.exe"
$l = Join-Path $repo "experiments\082_x_long_budget\launch082.ps1"
$agg = "experiments\082_x_long_budget\aggregate.py"
$log = "C:\Users\mlgbr\chain082.log"
$verdictFile = "C:\Users\mlgbr\verdict082.txt"
$workers = @{ "gatec" = 4; "train" = 4; "search" = 12 }
Set-Location $repo
$env:PYTHONPATH = Join-Path $repo "src"
function Say($m) { $line = (Get-Date -Format "yyyy-MM-dd HH:mm:ss") + " " + $m; Add-Content -Path $log -Value $line; $line }
Say "chain start from $From at $(git rev-parse --short HEAD)"
$order = @("gatec","train","gates","search","aggregate")
$go = $false
$runSearch = $true
foreach ($p in $order) {
    if ($p -eq $From) { $go = $true }
    if (-not $go) { continue }
    if ($p -eq "gates") {
        # Fail closed: a refused record raises SystemExit, prints neither search line, and must
        # stop the chain, not read as "search not run".
        $g = & $py $agg --stage gates 2>&1 | Out-String
        $code = $LASTEXITCODE
        Say ("gates: " + $g.Trim())
        if ($code -ne 0) { Say "STOP: the gates stage exited $code; fix the records before going on"; exit 1 }
        if ($g -match "SEARCH: NOT RUN") { $runSearch = $false; continue }
        if ($g -notmatch "SEARCH: RUN") { Say "STOP: the gates stage printed no search line"; exit 2 }
        continue
    }
    if ($p -eq "search" -and -not $runSearch) { Say "search not run (see the gates line)"; continue }
    if ($p -eq "aggregate") {
        $a = & $py $agg 2>&1 | Out-String
        $code = $LASTEXITCODE
        Say ("aggregate exit " + $code)
        # Written only on success, so the watcher never mistakes an error for a verdict.
        if ($code -ne 0) { Say ("STOP: the aggregate exited " + $code + ": " + $a.Trim()); exit 1 }
        Set-Content -Path ($verdictFile + ".tmp") -Value $a
        Move-Item -Force ($verdictFile + ".tmp") $verdictFile
        continue
    }
    Say "phase $p workers $($workers[$p])"
    & powershell -NoProfile -ExecutionPolicy Bypass -File $l -Phase $p -Workers $workers[$p]
    Say "phase $p exit $LASTEXITCODE"
    if ($LASTEXITCODE -ne 0) { Say "STOP: phase $p failed"; exit 1 }
}
Say "chain done"
