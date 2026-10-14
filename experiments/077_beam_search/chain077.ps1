# EXP-077 full-run chain: cont -> Gate 0(b) -> eval -> det. Re-runnable: phases skip existing cells.
param([string]$From = "cont")
$repo = "C:\Users\mlgbr\Desktop\Projects\neuromorphic-nn-snn-research-project"
$l = Join-Path $repo "experiments\077_beam_search\launch077.ps1"
$log = "C:\Users\mlgbr\chain077.log"
Set-Location $repo
function Say($m) { $line = (Get-Date -Format "yyyy-MM-dd HH:mm:ss") + " " + $m; Add-Content -Path $log -Value $line; $line }
Say "chain start from $From at $(git rev-parse --short HEAD)"
$order = @("cont","gate","eval","det")
$go = $false
foreach ($p in $order) {
    if ($p -eq $From) { $go = $true }
    if (-not $go) { continue }
    if ($p -eq "gate") {
        $env:PYTHONPATH = Join-Path $repo "src"
        $g = & (Join-Path $repo ".venv\Scripts\python.exe") experiments\077_beam_search\aggregate.py --stage cont 2>&1 | Out-String
        Say ("gate0b: " + $g.Trim())
        if ($g -notmatch "continuity.*: PASS") { Say "STOP: Gate 0(b) did not pass"; exit 2 }
        continue
    }
    if ($p -eq "cont") { $w = 2 }
    elseif ($p -eq "det") { $w = 1 }
    else {
        # rank and eval take the worker count the dated amendment fixed (spec section 6).
        $env:PYTHONPATH = Join-Path $repo "src"
        $wr = & (Join-Path $repo ".venv\Scripts\python.exe") -c "import importlib.util, sys; s=importlib.util.spec_from_file_location('a','experiments/077_beam_search/aggregate.py'); a=importlib.util.module_from_spec(s); s.loader.exec_module(a); sys.stdout.write(str(a.AMENDMENT['workers']))" 2>&1 | Out-String
        $w = 0
        if ($LASTEXITCODE -ne 0 -or -not [int]::TryParse($wr.Trim(), [ref]$w) -or $w -lt 1) { Say "STOP: could not read AMENDMENT workers (got '$($wr.Trim())')"; exit 2 }
    }
    Say "phase $p workers $w"
    if ($p -eq "det") { & powershell -NoProfile -ExecutionPolicy Bypass -File $l -Phase $p -Workers $w }
    else { & powershell -NoProfile -ExecutionPolicy Bypass -File $l -Phase $p -Workers $w -SkipExisting }
    Say "phase $p exit $LASTEXITCODE"
    if ($LASTEXITCODE -ne 0) { Say "STOP: phase $p failed"; exit 1 }
}
Say "chain done"
