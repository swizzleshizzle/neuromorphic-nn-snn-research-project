# EXP-079 full-run chain, after the dated amendment: cont (Gate 0(b) cells and Gate M's X
# continuity cell) -> Gate 0(b) -> train -> Gates L, E and M continuity -> eval -> det -> rank. Re-runnable with -From: training resumes by itself and evaluation
# phases skip existing cells. Launch it with OS-level redirection (playbook section 3):
#   ssh -n laptop 'cmd /c "powershell -NoProfile -ExecutionPolicy Bypass -File <repo>\experiments\079_judge_trained_deep\chain079.ps1 > C:\Users\mlgbr\chain079.out 2>&1"'
param([string]$From = "cont")
$repo = "C:\Users\mlgbr\Desktop\Projects\neuromorphic-nn-snn-research-project"
$py = Join-Path $repo ".venv\Scripts\python.exe"
$l = Join-Path $repo "experiments\079_judge_trained_deep\launch079.ps1"
$agg = "experiments\079_judge_trained_deep\aggregate.py"
$log = "C:\Users\mlgbr\chain079.log"
Set-Location $repo
$env:PYTHONPATH = Join-Path $repo "src"
function Say($m) { $line = (Get-Date -Format "yyyy-MM-dd HH:mm:ss") + " " + $m; Add-Content -Path $log -Value $line; $line }
Say "chain start from $From at $(git rev-parse --short HEAD)"
$w = & $py -c "import importlib.util, sys; s=importlib.util.spec_from_file_location('a','experiments/079_judge_trained_deep/aggregate.py'); a=importlib.util.module_from_spec(s); s.loader.exec_module(a); sys.stdout.write(str(a.AMENDMENT['train_workers'])+' '+str(a.AMENDMENT['eval_workers']))"
if ($LASTEXITCODE -ne 0 -or $w -notmatch "^\d+ \d+$") { Say "STOP: could not read train_workers and eval_workers from AMENDMENT (got '$w')"; exit 1 }
$trainW, $evalW = $w.Split(" ")
foreach ($wk in @($trainW, $evalW)) {
    if ([int]$wk -lt 1 -or [int]$wk -gt 6) { Say "STOP: AMENDMENT workers must each be an integer from 1 to 6 (train $trainW, eval $evalW); the launcher refuses more than 6"; exit 1 }
}
Say "workers: train $trainW, eval $evalW"
$order = @("cont","gate0b","train","gatele","eval","det","rank")
$go = $false
foreach ($p in $order) {
    if ($p -eq $From) { $go = $true }
    if (-not $go) { continue }
    if ($p -eq "gate0b") {
        $g = & $py $agg --stage cont 2>&1 | Out-String
        Say ("gate0b: " + $g.Trim())
        if ($g -notmatch "continuity.*: PASS") { Say "STOP: Gate 0(b) did not pass"; exit 2 }
        continue
    }
    if ($p -eq "gatele") {
        $g = & $py $agg --stage train 2>&1 | Out-String
        Say ("gates L, E, M: " + $g.Trim())
        if ($g -notmatch "GATE L \(VI worked\): PASS") { Say "STOP: Gate L did not pass, every claim is VOID"; exit 2 }
        if ($g -notmatch "GATE E \(encoders trained\): PASS") { Say "STOP: Gate E did not pass, every claim is VOID"; exit 2 }
        if ($g -notmatch "GATE M continuity \(X seed 0 Gate L re-measured here\): PASS") { Say "STOP: X's Gate L does not reproduce, so Gate M cannot be read; re-measure X before going on"; exit 2 }
        continue
    }
    $n = $evalW
    if ($p -eq "cont") { $n = 2 }
    if ($p -eq "train") { $n = $trainW }
    if ($p -eq "det") { $n = 1 }
    Say "phase $p workers $n"
    if ($p -eq "eval" -or $p -eq "rank") { & powershell -NoProfile -ExecutionPolicy Bypass -File $l -Phase $p -Workers $n -SkipExisting }
    else { & powershell -NoProfile -ExecutionPolicy Bypass -File $l -Phase $p -Workers $n }
    Say "phase $p exit $LASTEXITCODE"
    if ($LASTEXITCODE -ne 0) { Say "STOP: phase $p failed"; exit 1 }
}
Say "chain done"
