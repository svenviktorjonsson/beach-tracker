# Shared helpers for picking the Docker host port and persisting it for start-labeling.ps1.
# Dot-source from run_labeler.ps1 / start-labeling.ps1 (same directory).

$_here = $MyInvocation.MyCommand.Path
if (-not $_here) { $_here = $PSCommandPath }
$script:LabelerRoot = Split-Path -Parent $_here
$script:LabelerHostPortFile = Join-Path $script:LabelerRoot "labeler-host-port.txt"
# Docker Compose auto-loads .env from the project directory for ${VAR} substitution in compose YAML.
$script:LabelerDotEnvFile = Join-Path $script:LabelerRoot ".env"

function Get-LabelerDotEnvMap {
    $map = [ordered]@{}
    if (Test-Path -LiteralPath $script:LabelerDotEnvFile) {
        try {
            foreach ($line in [System.IO.File]::ReadAllLines($script:LabelerDotEnvFile)) {
                if ($line -match '^\s*#') { continue }
                if ($line -match '^\s*$') { continue }
                if ($line -match '^\s*([^=\s]+)\s*=\s*(.*)\s*$') {
                    $map[$Matches[1]] = $Matches[2]
                }
            }
        } catch { }
    }
    return $map
}

function Save-LabelerDotEnvMap {
    param([Parameter(Mandatory = $true)]$Map)
    try {
        $lines = @()
        foreach ($entry in $Map.GetEnumerator()) {
            $lines += "$($entry.Key)=$($entry.Value)"
        }
        [System.IO.File]::WriteAllText($script:LabelerDotEnvFile, ($lines -join "`r`n") + "`r`n")
    } catch {
        Write-Warning "Could not write $($script:LabelerDotEnvFile): $_"
    }
}

function Set-LabelerDotEnvValue {
    param(
        [Parameter(Mandatory = $true)][string]$Key,
        [Parameter(Mandatory = $true)][string]$Value
    )
    $map = Get-LabelerDotEnvMap
    $map[$Key] = $Value
    Save-LabelerDotEnvMap -Map $map
}

function Test-LabelerHostPortAvailable {
    param([Parameter(Mandatory = $true)][int]$Port)
    try {
        $l = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, $Port)
        $l.Start()
        $l.Stop()
        return $true
    } catch {
        return $false
    }
}

function Get-LabelerHostPort {
    param(
        [int]$Preferred = 8081,
        [int]$End = 8099
    )
    for ($p = $Preferred; $p -le $End; $p++) {
        if (Test-LabelerHostPortAvailable -Port $p) {
            return $p
        }
    }
    throw "No free TCP port on localhost in range $Preferred-$End. Close other apps or free a port."
}

function Save-LabelerHostPort {
    param([Parameter(Mandatory = $true)][int]$Port)
    try {
        [System.IO.File]::WriteAllText($script:LabelerHostPortFile, "$Port`r`n")
    } catch {
        Write-Warning "Could not write $($script:LabelerHostPortFile): $_"
    }
    Set-LabelerDotEnvValue -Key "LABELER_HOST_PORT" -Value "$Port"
}

function Read-LabelerHostPort {
    param([int]$Default = 8081)
    if (Test-Path -LiteralPath $script:LabelerHostPortFile) {
        try {
            $t = [System.IO.File]::ReadAllText($script:LabelerHostPortFile).Trim()
            if ($t -match '^\d+$') {
                return [int]$t
            }
        } catch { }
    }
    if (Test-Path -LiteralPath $script:LabelerDotEnvFile) {
        try {
            foreach ($line in [System.IO.File]::ReadAllLines($script:LabelerDotEnvFile)) {
                if ($line -match '^\s*LABELER_HOST_PORT\s*=\s*(\d+)\s*$') {
                    return [int]$Matches[1]
                }
            }
        } catch { }
    }
    return $Default
}
