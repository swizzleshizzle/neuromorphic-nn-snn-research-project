# EXP-080 full-run chain, after the dated amendment: mint -> gate0 -> Gate 0 -> train -> Gates V
# and T, Claims 1 and 2 -> search (only when the aggregator prints SEARCH: RUN, i.e. Claim 1 is
# CONFIRMED). Re-runnable with -From: every phase skips finished work and training resumes.
# Launch it with OS-level redirection (playbook section 3):
#   ssh -n laptop 'cmd /c "powershell -NoProfile -ExecutionPolicy Bypass -File <repo>\experiments\080_oracle_ordering\chain080.ps1 > C:\Users\mlgbr\chain080.out 2>&1"'
param([string]$From = "mint")
$repo = "C:\Users\mlgbr\Desktop\Projects\neuromorphic-nn-snn-research-project"
$py = Join-Path $repo ".venv\Scripts\python.exe"
$l = Join-Path $repo "experiments\080_oracle_ordering\launch080.ps1"
$agg = "experiments\080_oracle_ordering\aggregate.py"
$log = "C:\Users\mlgbr\chain080.log"
Set-Location $repo
$env:PYTHONPATH = Join-Path $repo "src"
function Say($m) { $line = (Get-Date -Format "yyyy-MM-dd HH:mm:ss") + " " + $m; Add-Content -Path $log -Value $line; $line }
Say "chain start from $From at $(git rev-parse --short HEAD)"
$w = & $py -c "import importlib.util, sys; s=importlib.util.spec_from_file_location('a','experiments/080_oracle_ordering/aggregate.py'); a=importlib.util.module_from_spec(s); s.loader.exec_module(a); m=a.AMENDMENT; sys.stdout.write(str(m['mint_workers'])+' '+str(m['train_workers'])+' '+str(m['eval_workers']))"
if ($LASTEXITCODE -ne 0 -or $w -notmatch "^\d+ \d+ \d+$") { Say "STOP: could not read mint_workers, train_workers and eval_workers from AMENDMENT (got '$w')"; exit 1 }
$mintW, $trainW, $evalW = $w.Split(" ")
foreach ($wk in @($mintW, $trainW, $evalW)) {
    if ([int]$wk -lt 1 -or [int]$wk -gt 6) { Say "STOP: AMENDMENT workers must each be an integer from 1 to 6 (mint $mintW, train $trainW, eval $evalW)"; exit 1 }
}
Say "workers: mint $mintW, train $trainW, eval $evalW"
$order = @("mint","gate0","gate0check","train","gates","search")
$go = $false
foreach ($p in $order) {
    if ($p -eq $From) { $go = $true }
    if (-not $go) { continue }
    if ($p -eq "gate0check") {
        $g = & $py $agg --stage gate0 2>&1 | Out-String
        Say ("gate0: " + $g.Trim())
        if ($g -notmatch "GATE 0 \(X seed 0 re-measured through the mint\): PASS") { Say "STOP: Gate 0 did not pass, the verdict is UNRESOLVED"; exit 2 }
        continue
    }
    if ($p -eq "gates") {
        # Fail closed: a refused record or a missing file raises SystemExit, prints neither
        # search line, and must stop the chain, not read as "search not run".
        $g = & $py $agg --stage gates 2>&1 | Out-String
        $code = $LASTEXITCODE
        Say ("gates: " + $g.Trim())
        if ($code -ne 0) { Say "STOP: the gates stage exited $code; fix the records before going on"; exit 1 }
        if ($g -match "SEARCH: NOT RUN") { Say "chain done: the search is not read (see the gates line); aggregate and report"; exit 0 }
        if ($g -notmatch "SEARCH: RUN") { Say "STOP: the gates stage printed no search line"; exit 2 }
        continue
    }
    $n = $evalW
    if ($p -eq "mint") { $n = $mintW }
    if ($p -eq "train") { $n = $trainW }
    if ($p -eq "gate0") { $n = 1 }
    Say "phase $p workers $n"
    & powershell -NoProfile -ExecutionPolicy Bypass -File $l -Phase $p -Workers $n
    Say "phase $p exit $LASTEXITCODE"
    if ($LASTEXITCODE -ne 0) { Say "STOP: phase $p failed"; exit 1 }
}
Say "chain done"
