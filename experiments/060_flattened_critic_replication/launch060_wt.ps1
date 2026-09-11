# EXP-060 launcher for SwizzlesDuo, running from the WORKTREE at C:\Users\mlgbr\wt-exp053.
#
#   ... -File C:\Users\mlgbr\launch060_wt.ps1 -Phase check
#   ... -File C:\Users\mlgbr\launch060_wt.ps1 -Phase baseline -Workers 10   # EXP-043 d6, seeds 14-23
#   ... -File C:\Users\mlgbr\launch060_wt.ps1 -Phase finetune -Workers 10   # EXP-047 confirm -> E1
#   ... -File C:\Users\mlgbr\launch060_wt.ps1 -Phase rl       -Workers 10   # EXP-060 arms B and F
#
# THREE PHASES, IN THAT ORDER, AND THE ORDER IS NOT OPTIONAL. E1 encoders exist only for seeds
# 0-13. Making them for 14-23 needs EXP-047 --mode confirm, which REFUSES without EXP-043's
# depth-6 records for the same seeds - and those exist only for 0-11. That dependency was
# discovered at dispatch time, not at spec time, and it is why `baseline` exists at all.
#
# THE WORKTREE HAS NO .venv, AND THAT IS A TRAP. The only interpreter belongs to the main
# checkout and installs the project editable with an ABSOLUTE path to the MAIN CHECKOUT's src, so
# a worktree script run with it imports the OLD library and produces a complete, plausible,
# entirely wrong result with no error. PYTHONPATH overrides it and the gate below PROVES it did.
#
# NO COMMA-SEPARATED ARGUMENTS. cmd.exe eats commas and the failure EXITS ZERO (EXP-055 lost a
# dispatch to `-Epochs 1,2,3,5` arriving as `1235`). Seed lists are therefore hardcoded here and
# never cross the ssh boundary. Verify a launch by probing for records and worker processes.

param(
    [Parameter(Mandatory=$true)][ValidateSet("check","baseline","finetune","rl")][string]$Phase,
    [int]$Workers = 0,
    [switch]$SkipExisting
)

$wt  = "C:\Users\mlgbr\wt-exp053"
$py  = "C:\Users\mlgbr\Desktop\Projects\neuromorphic-nn-snn-research-project\.venv\Scripts\python.exe"
$src = Join-Path $wt "src"
$seeds = @(14,15,16,17,18,19,20,21,22,23)

if (-not (Test-Path $wt)) { Write-Error "worktree missing: $wt"; exit 1 }
if (-not (Test-Path $py)) { Write-Error "interpreter missing: $py"; exit 1 }

Set-Location $wt
$env:PYTHONPATH = $src

$where = & $py -c "import neuromorphic, sys; sys.stdout.write(neuromorphic.__file__)"
if ($LASTEXITCODE -ne 0) { Write-Error "could not import neuromorphic"; exit 1 }
if (-not $where.StartsWith($src, [System.StringComparison]::OrdinalIgnoreCase)) {
    Write-Error "WRONG LIBRARY. neuromorphic resolved to $where, expected under $src. PYTHONPATH did not win; refusing to run."
    exit 1
}

# EXP-060 is EXP-056 again, so it lives or dies on this field existing in the WORKTREE's library.
# A stale worktree would run arm B twice under two different tags and the records would look
# perfectly ordinary.
$hasFlag = & $py -c "from neuromorphic.training.cube_baseline import CubeConfig; import sys; sys.stdout.write(str('flatten_critic' in CubeConfig.__dataclass_fields__))"
if ($hasFlag -ne "True") {
    Write-Error "CubeConfig has no flatten_critic field. The worktree is stale; this would silently run arm B twice."
    exit 1
}

$encDir = Join-Path $wt "experiments\040_pretrained_encoder_policy\outputs"
$e0 = @(Get-ChildItem -Path $encDir -Filter "exp040_encoder_s*.pt" -ErrorAction SilentlyContinue).Count
if ($e0 -lt 24) { Write-Error "found $e0 of 24 E0 encoders in $encDir; copy them from the main checkout."; exit 1 }

$sel = Join-Path $wt "experiments\047_encoder_finetuning\outputs\selected_lr.json"
if (-not (Test-Path $sel)) { Write-Error "selected_lr.json missing. EXP-047 confirm mode reads the RECORDED lr selection; do not pass one by hand."; exit 1 }

$d6Dir = Join-Path $wt "experiments\043_cap_at_depth_5_6\outputs"
$ftDir = Join-Path $wt "experiments\047_encoder_finetuning\outputs"
$nD6 = @($seeds | Where-Object { @(Get-ChildItem -Path $d6Dir -Filter ("exp043_capped_d6_regionalized_d6_s" + $_ + "_*.json") -ErrorAction SilentlyContinue).Count -gt 0 }).Count
$nE1 = @($seeds | Where-Object { Test-Path (Join-Path $ftDir ("exp047_ft_d6_lr0.0001_regionalized_d6_s" + $_ + "_sig0.0_encoder.pt")) }).Count

$head   = (& git rev-parse --short HEAD).Trim()
$branch = (& git rev-parse --abbrev-ref HEAD).Trim()
"library        = $where"
"worktree       = $branch @ $head"
"flatten_critic = $hasFlag"
"E0 encoders    = $e0 of 24"
"selected_lr    = present"
"EXP-043 d6 for seeds 14-23 = $nD6 of 10   (phase `baseline` produces these)"
"E1 encoders for seeds 14-23 = $nE1 of 10   (phase `finetune` produces these)"

$alive = @(Get-Process | Where-Object { $_.ProcessName -match "^python" }).Count
if ($alive -gt 2) { Write-Error "$alive python processes already running; refusing to start."; exit 1 }
"pyprocs        = $alive"

if ($Phase -eq "check") { "CHECK OK"; exit 0 }

if ($Workers -eq 0) { $Workers = 10 }

# 10 and 20 cells both divide by 10, giving clean waves. Per-cell time is WORSE at 10 workers
# than at 6 (0.16 vs 0.115 s/step, measured); the gain here is removing a ragged final wave, not
# parallelism. Memory: RL workers run about 1.05 GB private each, so 10 is ~10.5 GB of 31.4.
switch ($Phase) {
    "baseline" {
        $script  = "experiments\043_cap_at_depth_5_6\run.py"
        $cliArgs = @("--seeds") + $seeds + @("--depths", "6", "--workers", $Workers, "--skip-existing")
        $log     = Join-Path $wt "experiments\060_flattened_critic_replication\phase_baseline.log"
    }
    "finetune" {
        if ($nD6 -lt 10) {
            Write-Error "EXP-043 depth-6 records exist for only $nD6 of 10 seeds. EXP-047 confirm mode REFUSES without them - they are its paired baseline. Run -Phase baseline first."
            exit 1
        }
        $script  = "experiments\047_encoder_finetuning\run.py"
        $cliArgs = @("--mode", "confirm", "--seeds") + $seeds + @("--workers", $Workers, "--skip-existing")
        $log     = Join-Path $wt "experiments\060_flattened_critic_replication\phase_finetune.log"
    }
    "rl" {
        if ($nE1 -lt 10) {
            Write-Error "E1 encoders exist for only $nE1 of 10 seeds. Run -Phase finetune first."
            exit 1
        }
        $script  = "experiments\060_flattened_critic_replication\run.py"
        $cliArgs = @("--workers", $Workers)
        if ($SkipExisting) { $cliArgs += "--skip-existing" }
        $log     = Join-Path $wt "experiments\060_flattened_critic_replication\phase_rl.log"
    }
}

New-Item -ItemType Directory -Force (Join-Path $wt "experiments\060_flattened_critic_replication\outputs") | Out-Null

"script         = $script $cliArgs"
"log            = $log"
""

# -u so the log is not fully buffered. Tee-Object so the record survives an ssh drop: Windows has
# no SIGHUP semantics. Do NOT wrap in Start-Process over ssh; that dies with the session. Run in
# the ssh FOREGROUND and background the call on the controller side.
& $py -u $script @cliArgs 2>&1 | Tee-Object -FilePath $log
