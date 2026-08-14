# render_story.ps1 - render the manim story scenes on the laptop and pull stills out of them.
#
# WHY THIS IS A FILE AND NOT AN SSH ONE-LINER
#
# The remote default shell is cmd.exe. A `|` inside a quoted PowerShell string is interpreted by
# cmd BEFORE PowerShell sees it, so `Select-String -Pattern "Rendered|Error"` fails with
# "'Error' is not recognized as an internal or external command". That is the documented trap in
# docs/playbooks/remote-experiment-runs.md and it cost one render cycle on 2026-08-14.
#
# WHY IT EXTRACTS FRAMES
#
# manim reports success whether or not the layout is legible. The repo rule is to read real
# output, not green logs: a scene can render 25 animations perfectly and still have a caption
# sitting on top of a label. Stills are the only way to see that from a headless session.
#
# Usage (from the VPS, after sync_repo.ps1):
#     ssh -n laptop 'powershell -NoProfile -ExecutionPolicy Bypass -File `
#         C:\Users\mlgbr\Desktop\Projects\neuromorphic-nn-snn-research-project\scripts\laptop\render_story.ps1 -Quality ql'
#
# Then pull the stills:
#     scp 'laptop:C:/Users/mlgbr/manim-frames/*.png' <local dir>
#
# Requires: C:\Users\mlgbr\manim-venv (manim 0.21, NOT the project venv - it pulls scipy and this
# repo deliberately has none) and ffmpeg on PATH (winget Gyan.FFmpeg, installed 2026-08-14).

param(
    [string]  $Repo     = 'C:\Users\mlgbr\Desktop\Projects\neuromorphic-nn-snn-research-project',
    [string]  $Venv     = 'C:\Users\mlgbr\manim-venv',
    [string]  $MediaDir = 'C:\Users\mlgbr\manim-media',
    [string]  $FramesDir = 'C:\Users\mlgbr\manim-frames',
    [string]  $Quality  = 'ql',
    [int]     $Frames   = 3,
    [string[]]$Scenes   = @('TheBreakPointMoves', 'TheWall', 'ScaleOfTheCube', 'CollapseIsASymptom')
)

$ErrorActionPreference = 'Continue'

$python = Join-Path $Venv 'Scripts\python.exe'
$sceneDir = Join-Path $Repo 'viz\manim'
foreach ($p in @($python, $sceneDir)) {
    if (-not (Test-Path $p)) { Write-Output "MISSING: $p"; exit 2 }
}

# -ql writes to 480p15, -qh to 1080p60. Resolve rather than assume, so a quality change does not
# silently extract frames from last week's render.
$resDir = @{ 'ql' = '480p15'; 'qm' = '720p30'; 'qh' = '1080p60'; 'qk' = '2160p60' }[$Quality]
if (-not $resDir) { Write-Output "unknown quality '$Quality'"; exit 2 }

Set-Location $sceneDir
New-Item -ItemType Directory -Force -Path $FramesDir | Out-Null

$failed = @()
foreach ($scene in $Scenes) {
    $log = Join-Path $env:TEMP "manim_$scene.log"
    & $python -m manim "-$Quality" --media_dir $MediaDir scenes/story_scenes.py $scene *> $log
    $code = $LASTEXITCODE

    $video = Join-Path $MediaDir "videos\story_scenes\$resDir\$scene.mp4"
    if ($code -ne 0 -or -not (Test-Path $video)) {
        $failed += $scene
        Write-Output "FAILED $scene (exit $code) - last lines:"
        Get-Content $log -Tail 12 | ForEach-Object { Write-Output "    $_" }
        continue
    }

    $dur = [double](& ffprobe -v error -show_entries format=duration -of csv=p=0 $video)
    Write-Output ("OK {0}  {1:N1}s  {2}" -f $scene, $dur, $video)

    # Evenly spaced stills, and always the last frame - the closing composition is the one most
    # likely to have everything on screen at once, which is where overlaps show up.
    Remove-Item (Join-Path $FramesDir "$scene*.png") -ErrorAction SilentlyContinue
    for ($i = 1; $i -le $Frames; $i++) {
        $t = [math]::Round($dur * $i / ($Frames + 1), 2)
        & ffmpeg -v error -y -ss $t -i $video -frames:v 1 (Join-Path $FramesDir "${scene}_$i.png")
    }
    & ffmpeg -v error -y -sseof -0.5 -i $video -update 1 -frames:v 1 `
        (Join-Path $FramesDir "${scene}_final.png")
}

Write-Output "frames in $FramesDir"
if ($failed.Count -gt 0) { Write-Output ("FAILED: " + ($failed -join ', ')); exit 1 }
Write-Output "ALL RENDERED"
