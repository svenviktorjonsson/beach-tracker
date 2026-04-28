# Runs the full CSV pull in Docker (same as run_pull_frames_worker_docker.ps1).
& "$PSScriptRoot\run_pull_frames_worker_docker.ps1"
exit $LASTEXITCODE
