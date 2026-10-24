# EXP-081 chain: mint -> Gate C -> train (two legs) -> gates and claims -> search (only when the
# aggregator prints SEARCH: RUN, i.e. Claim 1 is CONFIRMED) -> full aggregate into
# C:\Users\mlgbr\verdict081.txt. Re-runnable from the top: every phase skips finished work and
# training resumes its checkpoints. Workers are spec section 10's: 6, 6, 6.
# Launch it with OS-level redirection (playbook section 3):
#   ssh -n laptop 'cmd /c "powershell -NoProfile -ExecutionPolicy Bypass -File <repo>\experiments\081_oxt_long_budget\chain081.ps1 > C:\Users\mlgbr\chain081.out 2>&1"'
param([ValidateSet("mint","gatec","train","gates","search","aggregate")][string]$From = "mint")
$repo = "C:\Users\mlgbr\Desktop\Projects\neuromorphic-nn-snn-research-project"
$py = Join-Path $repo ".venv\Scripts\python.exe"
$l = Join-Path $repo "experiments\081_oxt_long_budget\launch081.ps1"
$agg = "experiments\081_oxt_long_budget\aggregate.py"
$log = "C:\Users\mlgbr\chain081.log"
$verdictFile = "C:\Users\mlgbr\verdict081.txt"
$workers = 6
Set-Location $repo
$env:PYTHONPATH = Join-Path $repo "src"
function Say($m) { $line = (Get-Date -Format "yyyy-MM-dd HH:mm:ss") + " " + $m; Add-Content -Path $log -Value $line; $line }
Say "chain start from $From at $(git rev-parse --short HEAD)"
$order = @("mint","gatec","train","gates","search","aggregate")
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
        Set-Content -Path ($verdictFile + ".tmp") -Value $a
        Move-Item -Force ($verdictFile + ".tmp") $verdictFile
        Say ("aggregate exit " + $code)
        if ($code -ne 0) { Say "STOP: the aggregate exited $code"; exit 1 }
        continue
    }
    Say "phase $p workers $workers"
    & powershell -NoProfile -ExecutionPolicy Bypass -File $l -Phase $p -Workers $workers
    Say "phase $p exit $LASTEXITCODE"
    if ($LASTEXITCODE -ne 0) { Say "STOP: phase $p failed"; exit 1 }
}
Say "chain done"
