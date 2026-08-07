# sync_repo.ps1 - bring the laptop's checkout to origin/main before a dispatch, and FAIL
# LOUDLY if it did not get there.
#
# THE PROBLEM THIS EXISTS FOR
#
# Experiment head checkpoints are generated on the laptop and committed from the VPS.
# `.gitignore` ignores `experiments/*/outputs/*` but line 57 negates `*_head.pt`, so those
# checkpoints are tracked ON PURPOSE (EXP-030's memory re-ask needs them, and retraining to
# recover them costs 20 h). The consequence: after a run, the laptop holds its own UNTRACKED
# copies at exactly the paths the VPS has since committed. Git refuses to overwrite an
# untracked file on merge, so `git pull` fails and leaves the checkout on an OLD COMMIT.
#
# That is not hypothetical. Before the EXP-037 dispatch the laptop was 12 commits behind and
# `curriculum_weights` appeared zero times in its source; it was caught only by checking. It
# recurred before EXP-038 (48 colliding files at `022d8b8`).
#
# WHY NOT `git clean`
#
# A blanket `git clean -fd` would also delete the GITIGNORED `outputs/*.json` records, which
# are frequently the only copy of an experiment's data - EXP-036's records are the comparator
# for every EXP-037 claim and exist nowhere else. This moves aside EXACTLY the untracked files
# the incoming tree actually contains, and it MOVES them to an attic rather than deleting.
#
# Usage (scp it over first; it cannot live only in the repo, because the repo is what is broken):
#     ssh -n laptop 'powershell -NoProfile -ExecutionPolicy Bypass -File C:\Users\mlgbr\sync_repo.ps1'
#
# Exit codes: 0 = checkout is at origin/main. Non-zero = it is NOT, and no run should start.

param(
    [string]$Repo   = 'C:\Users\mlgbr\Desktop\Projects\neuromorphic-nn-snn-research-project',
    [string]$Attic  = 'C:\Users\mlgbr\repo-attic',
    [string]$Branch = 'main'
)

$ErrorActionPreference = 'Continue'

Set-Location $Repo
git fetch --all --prune | Out-Null

# Modified TRACKED files also block --ff-only, and unlike generated artifacts they may be real
# work. Never move or discard those silently; stop and let a human look.
$dirty = @(git diff --name-only HEAD)
if ($dirty.Count -gt 0) {
    Write-Output "ABORT: $($dirty.Count) modified tracked file(s) in the working tree:"
    $dirty | Select-Object -First 20 | ForEach-Object { Write-Output "    $_" }
    Write-Output "Resolve by hand. Refusing to touch tracked modifications."
    exit 2
}

# The intersection that matters: untracked-and-not-ignored files that the incoming tree also
# contains. Anything else untracked is harmless and is left exactly where it is.
$incoming = New-Object System.Collections.Generic.HashSet[string]
foreach ($p in @(git ls-tree -r --name-only "origin/$Branch")) { [void]$incoming.Add($p) }
$collisions = @(git ls-files --others --exclude-standard | Where-Object { $incoming.Contains($_) })

if ($collisions.Count -gt 0) {
    $stamp   = Get-Date -Format 'yyyyMMdd-HHmmss'
    $destDir = Join-Path $Attic $stamp
    Write-Output "$($collisions.Count) untracked file(s) collide with origin/$Branch; moving to $destDir"
    foreach ($rel in $collisions) {
        $src = Join-Path $Repo $rel
        $dst = Join-Path $destDir $rel
        New-Item -ItemType Directory -Force (Split-Path $dst) | Out-Null
        Move-Item -LiteralPath $src -Destination $dst -Force
    }
    Write-Output "moved. The tracked copies from origin are authoritative; the attic is the undo."
} else {
    Write-Output "no untracked collisions."
}

git checkout $Branch | Out-Null
git pull --ff-only "origin" $Branch

# VERIFY. A pull that printed something reassuring is not evidence. This is the check the
# handoff means by "do not trust a silenced pull".
$behind = (git rev-list --count "HEAD..origin/$Branch")
$head   = (git log --oneline -1)
Write-Output "repo at: $head"
Write-Output "behind origin/$($Branch): $behind"

if ($behind -ne '0') {
    Write-Output "FAILED: checkout is still $behind commit(s) behind origin/$Branch."
    git status --short | Select-Object -First 20
    exit 1
}

Write-Output "SYNCED"
exit 0
