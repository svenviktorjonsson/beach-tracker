# Start the labeler in Docker and open the browser.
# Builds if needed, starts detached by default, and writes the chosen host port
# to .env and labeler-host-port.txt.
param(
    [int]$Port = 0,
    [switch]$Foreground
)

Set-Location $PSScriptRoot
. "$PSScriptRoot\labeler-port.ps1"

function Write-LabelerImageIndex {
    $imagesDir = Join-Path $PSScriptRoot "data\images"
    $indexPath = Join-Path $PSScriptRoot "data\labeler_image_index.json"
    if (-not (Test-Path -LiteralPath $imagesDir)) {
        return
    }

    @'
from pathlib import Path
import json
from datetime import datetime, UTC

images = Path(r"__IMAGES_DIR__")
index = Path(r"__INDEX_PATH__")
names = [
    p.name
    for p in images.iterdir()
    if p.is_file()
    and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    and not p.name.lower().endswith("_next.png")
    and not p.name.lower().endswith("_diff.png")
]
payload = {
    "generated_at": datetime.now(UTC).isoformat(),
    "images_dir": str(images),
    "images_dir_mtime_ns": images.stat().st_mtime_ns,
    "labelable_images": names,
}
index.write_text(json.dumps(payload, indent=2), encoding="utf-8")
print(len(names))
'@.Replace("__IMAGES_DIR__", $imagesDir).Replace("__INDEX_PATH__", $indexPath) | python - | ForEach-Object {
        Write-Host "Indexed $_ labelable images for fast container startup." -ForegroundColor DarkCyan
    }
}

function Get-LabelerPublishedHostPort {
    param(
        [string]$Service = "labeler",
        [int]$ContainerPort = 8080
    )

    try {
        $out = & docker compose port $Service $ContainerPort 2>&1
        if ($LASTEXITCODE -ne 0) {
            return $null
        }
        $line = if ($null -eq $out) {
            ""
        } elseif ($out -is [array]) {
            ($out | ForEach-Object { "$_" }) -join "`n"
        } else {
            "$out"
        }
        if ($line -match ':(\d+)\s*$') {
            return [int]$Matches[1]
        }
    }
    catch {
    }
    return $null
}

function New-RandomSecret {
    param([int]$Length = 20)
    $chars = 'ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz23456789'
    -join (1..$Length | ForEach-Object { $chars[(Get-Random -Minimum 0 -Maximum $chars.Length)] })
}

function Ensure-TrainerAuthConfig {
    $envMap = Get-LabelerDotEnvMap
    $changed = $false
    if (-not $envMap.Contains('TRAINER_BASIC_AUTH_USER') -or [string]::IsNullOrWhiteSpace([string]$envMap['TRAINER_BASIC_AUTH_USER'])) {
        $envMap['TRAINER_BASIC_AUTH_USER'] = 'trainer'
        $changed = $true
    }
    if (-not $envMap.Contains('TRAINER_BASIC_AUTH_PASS') -or [string]::IsNullOrWhiteSpace([string]$envMap['TRAINER_BASIC_AUTH_PASS'])) {
        $envMap['TRAINER_BASIC_AUTH_PASS'] = New-RandomSecret -Length 18
        $changed = $true
    }
    if ($changed) {
        Save-LabelerDotEnvMap -Map $envMap
    }
    return @{
        User = [string]$envMap['TRAINER_BASIC_AUTH_USER']
        Pass = [string]$envMap['TRAINER_BASIC_AUTH_PASS']
    }
}

function Start-CloudflaredTunnel {
    param(
        [Parameter(Mandatory = $true)][int]$Port
    )
    $cloudflared = Get-Command cloudflared -ErrorAction SilentlyContinue
    if (-not $cloudflared) {
        Write-Warning "cloudflared not found; skipping public tunnel."
        return $null
    }

    $pidFile = Join-Path $PSScriptRoot "data\cloudflared-labeler.pid"
    $logFile = Join-Path $PSScriptRoot "data\cloudflared-labeler.log"
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $pidFile) | Out-Null

    if (Test-Path -LiteralPath $pidFile) {
        try {
            $oldPid = [int]([IO.File]::ReadAllText($pidFile).Trim())
            $oldProc = Get-Process -Id $oldPid -ErrorAction SilentlyContinue
            if ($oldProc) {
                Stop-Process -Id $oldPid -Force -ErrorAction SilentlyContinue
            }
        } catch { }
        Remove-Item $pidFile -ErrorAction SilentlyContinue
    }
    Remove-Item $logFile -ErrorAction SilentlyContinue

    $args = @('tunnel', '--url', "http://127.0.0.1:$Port", '--no-autoupdate', '--logfile', $logFile)
    $proc = Start-Process -FilePath $cloudflared.Source -ArgumentList $args -PassThru -WindowStyle Hidden
    [IO.File]::WriteAllText($pidFile, "$($proc.Id)`r`n")

    $deadline = (Get-Date).AddSeconds(25)
    while ((Get-Date) -lt $deadline) {
        if (Test-Path -LiteralPath $logFile) {
            $text = Get-Content -LiteralPath $logFile -Raw -ErrorAction SilentlyContinue
            if ($text -match 'https://[a-z0-9.-]+trycloudflare\.com') {
                return $Matches[0]
            }
        }
        Start-Sleep -Milliseconds 500
    }
    Write-Warning "Cloudflare tunnel started but URL was not detected yet. Check $logFile"
    return $null
}

function Test-UrlReady {
    param(
        [Parameter(Mandatory = $true)][string]$Url,
        [int]$TimeoutSec = 5
    )
    try {
        $response = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec $TimeoutSec -ErrorAction Stop
        return ($response.StatusCode -ge 200 -and $response.StatusCode -lt 500)
    }
    catch {
        return $false
    }
}

if ($Port -le 0) {
    $Port = Get-LabelerHostPort -Preferred 8081
    Write-Host "Using host port $Port (first free in 8081-8099)." -ForegroundColor Cyan
} elseif (-not (Test-LabelerHostPortAvailable -Port $Port)) {
    Write-Warning "Port $Port is in use; choosing next free port in 8081-8099."
    $Port = Get-LabelerHostPort -Preferred $Port
}

Save-LabelerHostPort -Port $Port
$env:LABELER_HOST_PORT = "$Port"
$url = "http://localhost:$Port/labeler"
Write-Host "LABELER_HOST_PORT=$Port written to .env and labeler-host-port.txt." -ForegroundColor DarkCyan
$trainerAuth = Ensure-TrainerAuthConfig

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-Error "Docker CLI not found. Install Docker Desktop and ensure 'docker' is on PATH."
    exit 1
}

try {
    docker info 2>$null | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "docker info failed"
    }
}
catch {
    Write-Error "Docker is not running. Start Docker Desktop, wait until it is ready, then run this script again."
    exit 1
}

Write-LabelerImageIndex

if ($Foreground) {
    Write-Host "Running docker compose up --build in the foreground." -ForegroundColor Cyan
    Write-Host "Open $url in your browser when the server is ready." -ForegroundColor Cyan
    docker compose up --build
    exit $LASTEXITCODE
}

Write-Host "Running docker compose up --build -d..." -ForegroundColor Cyan
$composeAttempt = 0
$composeOk = $false
while ($composeAttempt -lt 18 -and -not $composeOk) {
    $composeAttempt++
    Save-LabelerHostPort -Port $Port
    $env:LABELER_HOST_PORT = "$Port"
    $log = Join-Path $env:TEMP "labeler-compose-$PID-$composeAttempt.log"
    Remove-Item $log -ErrorAction SilentlyContinue
    docker compose up --build -d 2>&1 | Tee-Object -FilePath $log -Append
    $exitCode = $LASTEXITCODE
    $composeErr = if (Test-Path $log) { Get-Content -LiteralPath $log -Raw } else { "" }
    if ($exitCode -eq 0) {
        $composeOk = $true
        break
    }

    $retryable = $composeErr -match 'port is already allocated|address already in use|Bind for 0\.0\.0\.0:|only one usage of each socket address|Error starting userland proxy'
    if ($retryable) {
        Write-Warning "Port $Port is in use. Trying the next free port in 8081-8099."
        $Port = Get-LabelerHostPort -Preferred ($Port + 1)
        $url = "http://localhost:$Port/labeler"
        continue
    }

    Write-Error "docker compose failed:`n$composeErr"
    exit $exitCode
}

if (-not $composeOk) {
    Write-Error "Could not start labeler after $composeAttempt attempts."
    exit 1
}

$published = $null
for ($pi = 0; $pi -lt 40; $pi++) {
    $published = Get-LabelerPublishedHostPort
    if ($published) {
        break
    }
    Start-Sleep -Milliseconds 250
}

if ($published) {
    if ($published -ne $Port) {
        Write-Warning "Docker published host port $published; updating the saved port."
        $Port = $published
        Save-LabelerHostPort -Port $Port
        $env:LABELER_HOST_PORT = "$Port"
    }
    $url = "http://localhost:$Port/labeler"
} else {
    Write-Warning "Could not read the published port from docker compose port; using $url"
}

Write-Host "Labeler URL: $url" -ForegroundColor Cyan

$deadline = (Get-Date).AddSeconds(90)
$ready = $false
$waitStart = Get-Date
$lastProgress = $waitStart
Write-Host "Polling $url until HTTP 200 (up to 90s)." -ForegroundColor DarkGray
while ((Get-Date) -lt $deadline) {
    try {
        $r = Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 3 -ErrorAction Stop
        if ($r.StatusCode -eq 200) {
            $ready = $true
            break
        }
    }
    catch {
        $now = Get-Date
        if (($now - $lastProgress).TotalSeconds -ge 3) {
            $elapsed = [int](($now - $waitStart).TotalSeconds)
            Write-Host "  ... still waiting (${elapsed}s)" -ForegroundColor DarkGray
            $lastProgress = $now
        }
        Start-Sleep -Milliseconds 500
    }
}

if ($ready) {
    Write-Host "Server responded OK." -ForegroundColor Green
} else {
    Write-Warning "Server did not respond in time. Showing recent container logs:"
    docker compose logs --tail 60 2>&1 | Write-Host
    Write-Warning "Opening $url anyway."
}

$publicUrl = Start-CloudflaredTunnel -Port $Port
$shareUrl = $url
if ($publicUrl) {
    Write-Host "Checking public tunnel: $publicUrl" -ForegroundColor DarkGray
    if (Test-UrlReady -Url $publicUrl -TimeoutSec 8) {
        $shareUrl = $publicUrl
        Write-Host "Shareable labeler URL: $publicUrl" -ForegroundColor Green
    } else {
        Write-Warning "Public tunnel did not respond; falling back to local URL."
    }
} else {
    Write-Warning "Public tunnel did not come up; falling back to local URL."
}
Write-Host "Trainer auth: $($trainerAuth.User) / $($trainerAuth.Pass)" -ForegroundColor Yellow

Write-Host "Opening browser: $shareUrl" -ForegroundColor Green
try {
    Start-Process $shareUrl
}
catch {
    Write-Warning "Could not launch the default browser. Open manually: $shareUrl"
}
