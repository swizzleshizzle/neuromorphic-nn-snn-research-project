# EXP-084 chain for the desktop (desktop-ma4deik): check -> Gate P on the GPU -> Gate P check
# (stops on HALT and writes it as the verdict file) -> Gate E on the GPU -> Gate M on the CPU ->
# Gate T on the GPU -> full aggregate into D:\gpu083\verdict084.txt. One process at a time.
# Re-runnable from the top: every stage skips finished records. Gate C runs on the VPS before
# this, and its record must be in the pulled checkout.
# Launch it with OS-level redirection (playbook section 3), from the VPS:
#   ssh -n desktop 'cmd /c "powershell -NoProfile -ExecutionPolicy Bypass -File D:\gpu083\repo\experiments\084_gpu_port\chain084.ps1 > D:\gpu083\chain084.out 2>&1"'
# NO BACKTICKS IN DOUBLE-QUOTED OUTPUT STRINGS, and NO COMMA-SEPARATED ARGUMENTS across ssh.
param([ValidateSet("check","p","e","m","t")][string]$From = "check")
$repo = "D:\gpu083\repo"
$py = "D:\gpu083\venv\Scripts\python.exe"
$exp = "experiments\084_gpu_port"
$out = Join-Path $repo "$exp\outputs"
$log = "D:\gpu083\chain084.log"
$verdictFile = "D:\gpu083\verdict084.txt"
Set-Location $repo
$src = Join-Path $repo "src"
$env:PYTHONPATH = $src
function Say($m) { $line = (Get-Date -Format "yyyy-MM-dd HH:mm:ss") + " " + $m; Add-Content -Path $log -Value $line; $line }
function WriteVerdict($text) {
    Set-Content -Path ($verdictFile + ".tmp") -Value $text
    Move-Item -Force ($verdictFile + ".tmp") $verdictFile
}
function Stage($name, $script, $argv) {
    Say "stage $name"
    $stageLog = Join-Path $repo ("$exp\stage_" + $name + ".log")
    & $py -u (Join-Path $exp $script) @argv *>&1 | Tee-Object -FilePath $stageLog | Out-Null
    $code = $LASTEXITCODE
    Say "stage $name exit $code"
    if ($code -ne 0) { Say "STOP: stage $name failed"; exit 1 }
}
Say "chain start from $From at $(git rev-parse --short HEAD)"
$order = @("check","p","pcheck","e","m","t","aggregate")
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
            $gc = & $py -c "import json, sys; from pathlib import Path; p = Path(r'$out') / 'exp084_gatec.json'; sys.stdout.write(str(p.exists() and json.loads(p.read_text())['pass']))"
            if ($gc -ne "True") { Say "STOP: Gate C record missing or failed ($gc); run gatec.py on the VPS and pull"; exit 1 }
            # The CPU golden pins are bit-exact for the VPS, where the fixture was made; across
            # machines a 1-ULP difference is possible and is Gate M's job, not a reason to stop.
            $t = & $py -m pytest tests/training/test_device_port.py tests/experiments/test_exp084.py -q -k "not cpu_matches_golden" 2>&1 | Out-String
            $code = $LASTEXITCODE
            Add-Content -Path (Join-Path $repo "$exp\stage_check.log") -Value $t
            Say ("check pytest exit " + $code + ": " + ($t.Trim() -split '\r?\n')[-1])
            if ($code -ne 0) { Say "STOP: tests failed on the desktop"; exit 1 }
        }
        "p" { Stage "p" "decisions.py" @("--device", "cuda") }
        "pcheck" {
            $g = & $py (Join-Path $exp "aggregate.py") --stage p 2>&1 | Out-String
            $code = $LASTEXITCODE
            Say ("gate P: " + $g.Trim())
            if ($code -ne 0) { Say "STOP: the Gate P check exited $code"; exit 1 }
            if ($g -match "GATE P: HALT") { WriteVerdict $g; Say "Gate P halted the chain; verdict file written"; exit 3 }
            if (-not ($g -match "GATE P: CONTINUE")) { Say "STOP: the Gate P check printed no CONTINUE line"; exit 2 }
        }
        "e" { Stage "e" "reeval.py" @("--gate", "e", "--device", "cuda") }
        "m" { Stage "m" "reeval.py" @("--gate", "m") }
        "t" { Stage "t" "train.py" @("--device", "cuda") }
        "aggregate" {
            $a = & $py (Join-Path $exp "aggregate.py") --stage all 2>&1 | Out-String
            $code = $LASTEXITCODE
            Say ("aggregate exit " + $code)
            # Written only on success, so a watcher never mistakes an error for a verdict.
            if ($code -ne 0) { Say ("STOP: the aggregate exited " + $code + ": " + $a.Trim()); exit 1 }
            WriteVerdict $a
        }
    }
}
Say "chain done"
