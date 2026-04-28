# Full streams_export.csv fetch via Docker. Started by run_pull_frames_day_docker.ps1.
# Writes PNGs to ./data/images (volume mount). Logs under data/pull_logs/.

$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot
New-Item -ItemType Directory -Force -Path "data\images", "data\pull_logs" | Out-Null

$log = Join-Path $PSScriptRoot ("data\pull_logs\pull_docker_{0:yyyyMMdd_HHmmss}.log" -f (Get-Date))
Write-Host "Docker frame fetch — full /data/streams_export.csv"
Write-Host "Log: $log"
Write-Host "Rebuilding image if needed (--build) so pull_youtube_frames.py is present..."
Write-Host ""

# If YouTube blocks downloads: place Netscape cookies in .\data\youtube_cookies.txt and add:
#   --cookies /data/youtube_cookies.txt
# Or: docker compose run ... -e YTDLP_COOKIES=/data/youtube_cookies.txt ...
docker compose -f docker-compose.full.yml run --rm --build labeler python /app/pull_youtube_frames.py `
    --data-dir /data `
    --csv /data/streams_export.csv `
    --count 3 `
    --workers 1 `
    --sleep-between-rows 10 *>&1 | Tee-Object -FilePath $log

$code = $LASTEXITCODE
Write-Host ""
Write-Host "--- Finished (exit $code) ---"
Write-Host "Log: $log"
Write-Host "Press Enter to close."
Read-Host
exit $code
