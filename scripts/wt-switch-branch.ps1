# Switch the laptop worktree to a branch, without losing untracked checkpoints.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File C:\Users\mlgbr\wt-switch-branch.ps1 `
#       -Branch week25-reproducibility-audit
#
# WHY THIS EXISTS. Every experiment writes `*_head.pt` into `experiments/*/outputs/`, and those
# get committed later. So the files sit UNTRACKED in the worktree while being TRACKED on the
# branch being switched to, and `git checkout` refuses: "untracked working tree files would be
# overwritten". This has happened on every branch switch since EXP-063 and has been hand-fixed
# four times, once leaving the worktree as the only copy of 24 checkpoints.
#
# `git checkout -f` would work and is wrong: it DISCARDS the untracked files without ever
# comparing them, so a genuine difference would vanish silently. This stashes them, switches,
# then verifies every one against what git restored:
#
#   identical  -> git has it, drop the stashed copy
#   missing    -> git does NOT have it, move it back (it was never committed)
#   DIFFERENT  -> stop and say so, and keep the stash
#
# Run it as a FILE. Driving this through `ssh ... powershell -Command` fails: the quoting eats
# the backslashes and the whole thing silently produces no output. That is the documented
# playbook rule and it has bitten this project twice.

param(
    [Parameter(Mandatory=$true)][string]$Branch,
    [string]$Worktree = "C:\Users\mlgbr\wt-exp053",
    [string]$Stash = "C:\Users\mlgbr\wt_head_stash"
)

if (-not (Test-Path $Worktree)) { Write-Error "worktree missing: $Worktree"; exit 1 }
Set-Location $Worktree

New-Item -ItemType Directory -Force $Stash | Out-Null
if (@(Get-ChildItem $Stash -ErrorAction SilentlyContinue).Count -gt 0) {
    Write-Error "$Stash is not empty. A previous switch did not finish cleanly; inspect it before rerunning."
    exit 1
}

& git fetch origin $Branch 2>&1 | Select-Object -Last 1

$untracked = @(& git ls-files --others --exclude-standard | Where-Object { $_ -like "*_head.pt" })
"untracked checkpoints to stash = $($untracked.Count)"
foreach ($rel in $untracked) {
    $flat = $rel -replace "/", "~~"
    Move-Item -LiteralPath $rel -Destination (Join-Path $Stash $flat) -Force
}

& git checkout -B $Branch "origin/$Branch" 2>&1 | Select-Object -Last 1
$head = (& git rev-parse --short HEAD).Trim()
$now  = (& git rev-parse --abbrev-ref HEAD).Trim()
"branch = $now"
"head   = $head"

$same = 0; $movedBack = 0; $different = @()
foreach ($f in Get-ChildItem $Stash) {
    $rel = $f.Name -replace "~~", "\"
    if (Test-Path $rel) {
        if ((Get-FileHash $rel).Hash -eq (Get-FileHash $f.FullName).Hash) {
            $same++
        } else {
            $different += $rel
        }
    } else {
        New-Item -ItemType Directory -Force (Split-Path $rel) | Out-Null
        Move-Item $f.FullName $rel -Force
        $movedBack++
    }
}

"git-provided and identical = $same"
"not in git, moved back     = $movedBack"
"DIFFERENT                  = $($different.Count)"
if ($different.Count -gt 0) {
    $different | ForEach-Object { "  $_" }
    Write-Error "stash kept at $Stash. Do not delete it until these are explained."
    exit 1
}
Get-ChildItem $Stash | Remove-Item -Force
Remove-Item $Stash -Force -ErrorAction SilentlyContinue
"SWITCH OK"
