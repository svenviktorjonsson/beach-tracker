# Pull YouTube frames via Docker — ffmpeg, ffprobe, yt-dlp, and Node are in the image (not on the host).
# Host ./data is mounted at /data in the container. Use container paths: --csv /data/streams_export.csv
#
# Examples:
#   .\run_pull_frames.ps1 --csv /data/streams_export.csv --limit 1 --no-cookies
#   .\run_pull_frames.ps1 --csv /data/streams_export.csv --count 3 --no-cookies
#
# Full CSV in a separate window: .\run_pull_frames_day.ps1

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

docker compose -f docker-compose.full.yml run --rm --build labeler python /app/pull_youtube_frames.py --data-dir /data @args
exit $LASTEXITCODE
