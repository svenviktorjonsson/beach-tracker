# Run pull_youtube_clips.py in Docker: ~2 min MP4s under data/clips/, frames under data/images/.
# Example:
#   .\run_pull_clips_docker.ps1 --csv /data/streams_export.csv --limit 1 --count 8 --diff

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

docker compose -f docker-compose.full.yml run --rm --build labeler python /app/pull_youtube_clips.py --data-dir /data @args
exit $LASTEXITCODE
