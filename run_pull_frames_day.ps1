# Opens a NEW PowerShell window: full CSV pull inside Docker (ffmpeg + yt-dlp + ffprobe in the image).
# Requires Docker Desktop. Build pull image: docker compose -f docker-compose.full.yml build

$worker = Join-Path $PSScriptRoot "run_pull_frames_worker_docker.ps1"
if (-not (Test-Path $worker)) {
    Write-Error "Missing $worker"
    exit 1
}

Start-Process powershell.exe -ArgumentList @(
    "-NoExit",
    "-ExecutionPolicy", "Bypass",
    "-File", $worker
)

Write-Host "Started Docker frame fetch in a separate window."
Write-Host "Images go to .\data\images (mounted into the container as /data)."
Write-Host ""
