# launch047.ps1 - dispatch the EXP-047 chain on SwizzlesDuo. Copy over with scp, run with
#   ssh -n laptop 'powershell -NoProfile -ExecutionPolicy Bypass -File C:\Users\mlgbr\launch047.ps1'
#
# FIVE PHASES, SEQUENTIAL, UNATTENDED. About 22 h nominal, nearer 15 h at this project's
# historical estimate-vs-actual ratio.
#
#   0  pretrain EXP-040 encoders for seeds 12-23      ~20 min   (needed by BOTH 1 and 4)
#   1  EXP-047 pilot: 3 rates x seeds 12,13           ~5.6 h    6 workers
#   1b probe the pilot encoders                       ~15 min
#   2  select_lr.py - MECHANICAL, PROBE-ONLY          ~1 min    may HALT phases 3/3b
#   3  EXP-047 confirmatory: selected rate, seeds 0-11 ~6.7 h   10 workers
#   3b probe the confirmatory encoders                ~30 min
#   4  fallback: depth 5 at seeds 12-23, BOTH arms    ~9.4 h    10 workers
#
# PHASE 4 RUNS EVEN IF PHASE 2 HALTS. The fallback is independent and already justified, so a
# halted selection leaves the machine doing useful work instead of idle.
#
# THE THREE FAILURE MODES FROM launch043.ps1 ARE AVOIDED HERE TOO. Do not "simplify" them back
# in; all three were paid for in week 17.
#
# 1. NO Start-Process. Windows OpenSSH tears down the session job object when the ssh connection
#    closes, which kills a detached child. Every run must be a direct child of this script, and
#    the script must be launched with `ssh -n` so it survives the pipe closing.
#
# 2. NO $ErrorActionPreference = 'Stop' with 2>&1. PowerShell escalates a native command's FIRST
#    stderr line into a terminating error, and reinforce.py emits a UserWarning on episode one.
#    That combination killed a run and orphaned 16 workers that looked healthy for 35 minutes.
#
# 3. PLAIN FILE REDIRECTION, never a pipeline. No Tee-Object. Orphaned workers inherit the stdout
#    handle and block a Tee-Object reader forever, so the log stops updating while the run is
#    still alive - which reads exactly like a hang.
#
# AND ONE MORE, EARNED ON EXP-046: exit codes here are NOT a health signal. `Tee-Object` plus
# torch's UserWarning made a perfectly good dispatch report `exit 1` twice. Every phase below is
# therefore gated on an ARTIFACT existing, never on $LASTEXITCODE.

$ErrorActionPreference = 'Continue'

$repo = 'C:\Users\mlgbr\Desktop\Projects\neuromorphic-nn-snn-research-project'
$exp  = Join-Path $repo 'experiments\047_encoder_finetuning'
$out  = Join-Path $exp 'outputs'
$enc  = Join-Path $repo 'experiments\040_pretrained_encoder_policy\outputs'
$py   = Join-Path $repo '.venv\Scripts\python.exe'

$pilotSeeds   = @(12, 13)
$newSeeds     = @(12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23)
$confirmSeeds = @(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)

function Say($msg) {
    Write-Output ("[{0}] {1}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $msg)
}

# THE SYNC IS NOT OPTIONAL AND ITS EXIT CODE IS NOT ADVISORY. A bare `git pull` here has
# silently failed twice, and this dispatch depends on a trainer change (`encoder_lr`) that does
# not exist in an old checkout - the run would silently produce FROZEN numbers under a
# fine-tuned tag, which is the worst possible failure of this particular experiment.
& powershell -NoProfile -ExecutionPolicy Bypass -File C:\Users\mlgbr\sync_repo.ps1
if ($LASTEXITCODE -ne 0) {
    Say "ABORT: sync_repo.ps1 exited $LASTEXITCODE. Not starting a 22-hour chain on an unknown tree."
    exit 1
}

Set-Location $repo
New-Item -ItemType Directory -Force $out | Out-Null

# THE SYMBOL CHECK. `SYNCED` proves the commit; it does not prove the code. EXP-047 is entirely
# about a switch that, when missing, produces numbers that look completely normal.
$hits = (Select-String -Path 'src\neuromorphic\training\cube_baseline.py' -Pattern 'encoder_lr').Count
$hits2 = (Select-String -Path 'src\neuromorphic\training\reinforce.py' -Pattern 'grad_brain').Count
Say "symbol check: encoder_lr x$hits in cube_baseline.py, grad_brain x$hits2 in reinforce.py"
if ($hits -lt 1 -or $hits2 -lt 1) {
    Say "ABORT: the fine-tuning seam is not present in this checkout. A run would report FROZEN"
    Say "       numbers under a fine-tuned tag. That is unrecoverable ambiguity, not a slow run."
    exit 1
}

# ---------------- phase 0: encoders for the new seeds ----------------
Say "PHASE 0: pretraining EXP-040 encoders for seeds $($newSeeds -join ',')"
# TEN workers, not twelve. Measured on this very dispatch: 12 workers took >130 min for the
# 12 encoders that EXP-040 built in 100 min at 10 workers (its RESULTS.md provenance,
# 2026-08-09 14:40-16:20, same laptop, same operation). Per-worker CPU sat at ~0.5 across all
# 12, i.e. ~6.0 effective cores against the ~8.4 EXP-040 got from 10. Twelve loses even though
# it fits every seed in ONE scheduling wave where 10 needs two - the machine is memory-bound,
# not core-bound, exactly as the playbook says. Extends "10 beat 16" down to "10 beat 12".
& $py -u "$exp\pretrain_seeds.py" --seeds $newSeeds --workers 10 >> "$out\phase0_pretrain.log" 2>&1

$haveAll = $true
foreach ($s in $newSeeds) {
    if (-not (Test-Path (Join-Path $enc "exp040_encoder_s$s.pt"))) { $haveAll = $false }
}
if (-not $haveAll) {
    Say "ABORT: phase 0 did not produce every encoder. See phase0_pretrain.log."
    exit 1
}
Say "PHASE 0 done: all $($newSeeds.Count) encoders present."

# ---------------- phase 1: the pilot ----------------
# SIX workers, not ten: there are only six cells, so more workers buys nothing and costs memory.
Say "PHASE 1: EXP-047 pilot, 3 rates x seeds $($pilotSeeds -join ','), 10,000 episodes"
& $py -u "$exp\run.py" --mode pilot --seeds $pilotSeeds --workers 6 --skip-existing >> "$out\phase1_pilot.log" 2>&1

$pilotRecords = @(Get-ChildItem -Path $out -Filter 'exp047_ft_d6_lr*.json' -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -match '_s(1[23])_sig' })
Say "PHASE 1 done: $($pilotRecords.Count) pilot record(s) (expected 6)."
if ($pilotRecords.Count -lt 6) {
    Say "WARNING: incomplete pilot. select_lr.py will refuse to select from it, and phases 3/3b"
    Say "         will not run. Phase 4 still will."
}

# ---------------- phase 1b: probe the pilot encoders ----------------
Say "PHASE 1b: probing pilot encoders (Claim 2 instrument, and the SELECTION input)"
& $py -u "$exp\probe_encoders.py" --mode pilot --workers 4 >> "$out\phase1b_probe.log" 2>&1

# ---------------- phase 2: selection ----------------
# MECHANICAL AND PROBE-ONLY. It reads probe_pilot.json and nothing else; it cannot see a success
# rate. Combined with pilot seeds 12-13 being disjoint from the confirmatory 0-11, nothing that
# decides a claim was used to make this choice. See spec section 5.2.
Say "PHASE 2: selecting encoder_lr (probe-only, mechanical, pre-registered)"
& $py -u "$exp\select_lr.py" >> "$out\phase2_select.log" 2>&1

$selected = $null
$selPath = Join-Path $out 'selected_lr.json'
if (Test-Path $selPath) {
    $selected = (Get-Content $selPath -Raw | ConvertFrom-Json).selected_lr
}

if ($null -eq $selected) {
    Say "PHASE 2 HALT: no rate passed the gate, or selection did not run."
    Say "  Per spec 5.2 step 3 this IS A RESULT: REINFORCE's gradient damages the pretrained"
    Say "  representation at every rate in the pre-registered grid. Do NOT widen the grid without"
    Say "  a new pre-registration. Skipping phases 3 and 3b; going straight to the fallback."
} else {
    Say "PHASE 2 selected encoder_lr = $selected"

    # ---------------- phase 3: the confirmatory arm ----------------
    Say "PHASE 3: EXP-047 confirmatory, seeds $($confirmSeeds -join ','), 10 workers"
    & $py -u "$exp\run.py" --mode confirm --seeds $confirmSeeds --workers 10 --skip-existing >> "$out\phase3_confirm.log" 2>&1

    $confirmRecords = @(Get-ChildItem -Path $out -Filter 'exp047_ft_d6_lr*.json' -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -match '_s([0-9]|1[01])_sig' })
    Say "PHASE 3 done: $($confirmRecords.Count) confirmatory record(s) (expected 12)."

    # ---------------- phase 3b: probe the confirmatory encoders ----------------
    Say "PHASE 3b: probing confirmatory encoders (Claim 2)"
    & $py -u "$exp\probe_encoders.py" --mode confirm --workers 4 >> "$out\phase3b_probe.log" 2>&1
}

# ---------------- phase 4: the fallback, unconditional ----------------
# Settles EXP-043's Claim 1 (depth 5, +0.1108 at p 0.0815, open since Aug 14). BOTH arms are
# needed: Claim 1 is a PAIRED delta against EXP-040's depth-5 cell, and EXP-040 phase 2 only
# ever ran seeds 0-11, so seeds 12-23 have no baseline to pair against unless it is run here.
# The handoff's "~4.6 h for the RL arm" budgeted only one of the two.
$fb = Join-Path $repo 'experiments\043_cap_at_depth_5_6'
$fb40 = Join-Path $repo 'experiments\040_pretrained_encoder_policy'

Say "PHASE 4a: EXP-040 depth 5 (UNCAPPED baseline arm), seeds $($newSeeds -join ',')"
& $py -u "$fb40\run.py" --seeds $newSeeds --depths 5 --workers 10 --skip-existing >> "$out\phase4a_exp040_d5.log" 2>&1

Say "PHASE 4b: EXP-043 depth 5 (CAPPED arm), seeds $($newSeeds -join ',')"
& $py -u "$fb\run.py" --seeds $newSeeds --depths 5 --workers 10 --skip-existing >> "$out\phase4b_exp043_d5.log" 2>&1

Say "CHAIN COMPLETE."
Say "  EXP-047 selected_lr : $selected"
Say "  aggregate with      : .venv\Scripts\python.exe experiments\047_encoder_finetuning\aggregate.py"
Say "  logs                : $out\phase*.log"
