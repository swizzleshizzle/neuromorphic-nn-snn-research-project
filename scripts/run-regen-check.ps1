# Run EXP-040's E0 regeneration check from the worktree, with the library gate the launchers use.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File C:\Users\mlgbr\run-regen-check.ps1 -Workers 10
#
# A FILE and not an ssh one-liner, deliberately: driving PowerShell through `ssh ... -Command`
# eats backslashes and has silently produced no output twice in this project.

param(
    [int]$Workers = 10,
    [string]$Worktree = "C:\Users\mlgbr\wt-exp053"
)

$py  = "C:\Users\mlgbr\Desktop\Projects\neuromorphic-nn-snn-research-project\.venv\Scripts\python.exe"
$src = Join-Path $Worktree "src"
if (-not (Test-Path $py)) { Write-Error "interpreter missing: $py"; exit 1 }
Set-Location $Worktree
$env:PYTHONPATH = $src

# THE WORKTREE HAS NO .venv. The only interpreter belongs to the main checkout and installs the
# project editable with an ABSOLUTE path to the MAIN CHECKOUT's src, so without this gate the
# script would import the OLD library and compare the wrong code's output.
$where = & $py -c "import neuromorphic, sys; sys.stdout.write(neuromorphic.__file__)"
if (-not $where.StartsWith($src, [System.StringComparison]::OrdinalIgnoreCase)) {
    Write-Error "WRONG LIBRARY. neuromorphic resolved to $where, expected under $src."
    exit 1
}
"library = $where"
"branch  = " + (& git rev-parse --abbrev-ref HEAD).Trim() + " @ " + (& git rev-parse --short HEAD).Trim()

$alive = @(Get-Process | Where-Object { $_.ProcessName -match "^python" }).Count
if ($alive -gt 2) { Write-Error "$alive python processes already running; refusing."; exit 1 }
"pyprocs = $alive"
""

$log = Join-Path $Worktree "experiments\040_pretrained_encoder_policy\regen_check.log"
& $py -u "experiments\040_pretrained_encoder_policy\verify_regeneration.py" `
    --seeds 0 1 2 3 4 5 6 7 8 9 10 11 --workers $Workers 2>&1 | Tee-Object -FilePath $log
