# EXP-085 chain for the desktop (desktop-ma4deik): check -> rebase -> pre-flight (GPU) -> train
# (both arms, 12 seeds) -> search (48 cells) -> Gate H (a) search replay -> Gate H (b) training
# replay -> aggregate into D:\gpu083\verdict085.txt. One process at a time. Re-runnable from the
# top: every stage skips finished work. Gate C runs on the VPS first (stage_vps.sh), and its
# records and the start checkpoints must be in this checkout.
# Launch it detached from the VPS with launch085.ps1 (Win32_Process.Create), never a bare ssh.
# NO BACKTICKS IN DOUBLE-QUOTED OUTPUT STRINGS, and NO COMMA-SEPARATED ARGUMENTS across ssh.
param([ValidateSet("check","rebase","preflight","train","search","gateha","gatehb","aggregate")][string]$From = "check")
$repo = "D:\gpu083\repo"
$py = "D:\gpu083\venv\Scripts\python.exe"
$exp = "experiments\085_outward_walks"
$out = Join-Path $repo "$exp\outputs"
$log = "D:\gpu083\chain085.log"
$verdictFile = "D:\gpu083\verdict085.txt"
$verdict084 = "D:\gpu083\verdict084.txt"
Set-Location $repo
$src = Join-Path $repo "src"
$env:PYTHONPATH = $src
$env:PYTHONFAULTHANDLER = "1"
function Say($m) { $line = (Get-Date -Format "yyyy-MM-dd HH:mm:ss") + " " + $m; Add-Content -Path $log -Value $line; $line }
function WriteVerdict($text) {
    Set-Content -Path ($verdictFile + ".tmp") -Value $text
    Move-Item -Force ($verdictFile + ".tmp") $verdictFile
}
function Stage($name, $script, $argv) {
    Say "stage $name"
    $stageLog = Join-Path $repo ("$exp\stage_" + $name + ".log")
    & $py -u (Join-Path $exp $script) @argv *>&1 | Tee-Object -FilePath $stageLog -Append | Out-Null
    $code = $LASTEXITCODE
    Say "stage $name exit $code"
    if ($code -ne 0) { Say "STOP: stage $name failed"; exit 1 }
}
Say "chain start from $From at $(git rev-parse --short HEAD)"
$order = @("check","rebase","preflight","train","search","gateha","gatehb","aggregate")
$go = $false
foreach ($s in $order) {
    if ($s -eq $From) { $go = $true }
    if (-not $go) { continue }
    switch ($s) {
        "check" {
            $where = & $py -c "import neuromorphic, sys; sys.stdout.write(neuromorphic.__file__)"
            if (-not $where -or -not $where.StartsWith($src, [System.StringComparison]::OrdinalIgnoreCase)) { Say "STOP: neuromorphic resolved to $where, expected under $src"; exit 1 }
            $gpu = & $py -c "import torch, sys; sys.stdout.write(str(torch.cuda.is_available()))"
            if ($gpu -ne "True") { Say "STOP: torch.cuda.is_available() is $gpu"; exit 1 }
            if (-not (Test-Path $verdict084) -or -not (Select-String -Path $verdict084 -Pattern "VERDICT: GPU PORT REPRODUCES" -Quiet)) { Say "STOP: EXP-084 has not read GPU PORT REPRODUCES (spec section 4)"; exit 1 }
            $gc = & $py -c "import json, sys; from pathlib import Path; o = Path(r'$out'); ok = all((o / f'exp085_gatec_s{s}.json').exists() and json.loads((o / f'exp085_gatec_s{s}.json').read_text())['pass'] and (o / 'start' / f'judge_X_s{s}' / 'state.json').exists() for s in range(12)); sys.stdout.write(str(ok))"
            if ($gc -ne "True") { Say "STOP: Gate C records or start checkpoints missing; run stage_vps.sh on the VPS"; exit 1 }
            $t = & $py -m pytest tests/experiments/test_exp085.py -q -m "not slow" 2>&1 | Out-String
            $code = $LASTEXITCODE
            Add-Content -Path (Join-Path $repo "$exp\stage_check.log") -Value $t
            Say ("check pytest exit " + $code + ": " + ($t.Trim() -split '\r?\n')[-1])
            if ($code -ne 0) { Say "STOP: tests failed on the desktop"; exit 1 }
        }
        "rebase" { Stage "rebase" "rebase.py" @("--device", "cuda") }
        "preflight" { Stage "preflight" "preflight.py" @("--part", "gpu", "--device", "cuda") }
        "train" { Stage "train" "train.py" @("--device", "cuda") }
        "search" { Stage "search" "evaluate.py" @("--device", "cuda") }
        "gateha" { Stage "gateha" "gateh.py" @("--part", "a", "--device", "cuda") }
        "gatehb" { Stage "gatehb" "gateh.py" @("--part", "b", "--device", "cuda") }
        "aggregate" {
            $a = & $py (Join-Path $exp "aggregate.py") 2>&1 | Out-String
            $code = $LASTEXITCODE
            Say ("aggregate exit " + $code)
            # Written only on success, so a watcher never mistakes an error for a verdict.
            if ($code -ne 0) { Say ("STOP: the aggregate exited " + $code + ": " + $a.Trim()); exit 1 }
            WriteVerdict $a
        }
    }
}
Say "chain done"
