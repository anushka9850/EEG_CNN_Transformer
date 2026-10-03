<# .SYNOPSIS
    Real-data pipeline: download check -> train -> evaluate.
#>
$ErrorActionPreference = "Stop"
Write-Host "=== EEG CNN-Transformer: Real Data Pipeline ==="

# Check if at least one dataset exists
if (-not (Test-Path "data\raw")) {
    Write-Host "No data\raw\ directory found. Run scripts\download_data.ps1 first."
    exit 1
}

$found = $false
foreach ($d in @("chbmit", "eegmmidb", "ds004504", "ds002778", "modma", "deap", "tuh_normal")) {
    if (Test-Path "data\raw\$d") {
        Write-Host "  Found: data\raw\$d"
        $found = $true
    }
}
if (-not $found) {
    Write-Host "No datasets found in data\raw\. Run scripts\download_data.ps1 first."
    exit 1
}

Write-Host ""
Write-Host "=== Data check ==="
python -m eegct check -c configs/real_data.yaml

Write-Host ""
Write-Host "=== Build + Train + Evaluate ==="
python -m eegct all -c configs/real_data.yaml

Write-Host ""
Write-Host "Done. Results in outputs_real/"
