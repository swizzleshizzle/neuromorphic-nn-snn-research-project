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
    $w = 6; if ($p -eq "cont") { $w = 2 }; if ($p -eq "det") { $w = 1 }
    Say "phase $p workers $w"
    if ($p -eq "det") { & powershell -NoProfile -ExecutionPolicy Bypass -File $l -Phase $p -Workers $w }
    else { & powershell -NoProfile -ExecutionPolicy Bypass -File $l -Phase $p -Workers $w -SkipExisting }
    Say "phase $p exit $LASTEXITCODE"
    if ($LASTEXITCODE -ne 0) { Say "STOP: phase $p failed"; exit 1 }
}
Say "chain done"
