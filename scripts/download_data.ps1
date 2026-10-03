<# .SYNOPSIS
    Download freely available EEG datasets into data\raw\.
    TUH (Healthy), DEAP, and MODMA require free registration — see README.
#>
$ErrorActionPreference = "Stop"
$rawDir = "data\raw"
if (-not (Test-Path $rawDir)) { New-Item -ItemType Directory -Force -Path $rawDir | Out-Null }

# ---- CHB-MIT (Epilepsy) via PhysioNet wget or mne.datasets.eegbci
Write-Host "=== CHB-MIT (Epilepsy) ==="
Write-Host "Downloading CHB-MIT from PhysioNet (a few patients)..."
$chbPatients = @("chb01", "chb02", "chb03", "chb05", "chb08", "chb10")
foreach ($p in $chbPatients) {
    $dest = Join-Path $rawDir "chbmit\$p"
    if (-not (Test-Path $dest)) { New-Item -ItemType Directory -Force -Path $dest | Out-Null }
    $url = "https://physionet.org/files/chbmit/1.0.0/$p/"
    Write-Host "  Downloading $p ..."
    try {
        # Use wget if available, else Invoke-WebRequest for the summary file
        $summaryUrl = "${url}${p}-summary.txt"
        Invoke-WebRequest -Uri $summaryUrl -OutFile (Join-Path $dest "$p-summary.txt") -ErrorAction SilentlyContinue
    } catch {
        Write-Host "  Could not download $p summary (manual download may be needed)"
    }
    Write-Host "  For full download, use: wget -r -N -c -np -nH --cut-dirs=3 -P $dest $url"
}
Write-Host "  Or download from Kaggle: kaggle datasets download -d beyzanurdin/chb-mit-scalp-eeg-database"

# ---- EEGMMIDB (extra Healthy controls) via mne.datasets.eegbci
Write-Host ""
Write-Host "=== EEGMMIDB (Healthy baselines) ==="
Write-Host "Downloading EEGMMIDB baseline runs via mne.datasets.eegbci..."
try {
    python -c @"
import mne, shutil, os
from pathlib import Path
dest = Path('data/raw/eegmmidb')
dest.mkdir(parents=True, exist_ok=True)
for s in range(1, 41):
    try:
        paths = mne.datasets.eegbci.load_data(s, [1, 2], update_path=False)
        for p in paths:
            p = Path(p)
            subdir = dest / f'S{s:03d}'
            subdir.mkdir(exist_ok=True)
            tgt = subdir / p.name
            if not tgt.exists():
                shutil.copy2(p, tgt)
        print(f'  S{s:03d} OK')
    except Exception as e:
        print(f'  S{s:03d} SKIP: {e}')
"@
} catch {
    Write-Host "  mne download failed. Install mne and retry, or download manually from PhysioNet."
}

# ---- OpenNeuro ds004504 (Alzheimer) and ds002778 (Parkinson)
Write-Host ""
Write-Host "=== OpenNeuro datasets ==="
try {
    pip install -q openneuro-py 2>$null
    Write-Host "Downloading ds004504 (Alzheimer)..."
    openneuro-py download --dataset ds004504 --target-dir "data/raw/ds004504" --exclude "derivatives"
    Write-Host "Downloading ds002778 (Parkinson)..."
    openneuro-py download --dataset ds002778 --target-dir "data/raw/ds002778"
} catch {
    Write-Host "  openneuro-py not available. Install with: pip install openneuro-py"
    Write-Host "  Then run:"
    Write-Host "    openneuro-py download --dataset ds004504 --target-dir data/raw/ds004504 --exclude derivatives"
    Write-Host "    openneuro-py download --dataset ds002778 --target-dir data/raw/ds002778"
}

# ---- Manual download instructions
Write-Host ""
Write-Host "=== Datasets requiring registration ==="
Write-Host ""
Write-Host "MODMA (Depression + Healthy):"
Write-Host "  1. Register at http://modma.lzu.edu.cn/data/index/"
Write-Host "  2. Download 128-channel resting EEG .mat files"
Write-Host "  3. Place in data/raw/modma/"
Write-Host ""
Write-Host "DEAP (Healthy controls):"
Write-Host "  1. Register at https://www.eecs.qmul.ac.uk/mmv/datasets/deap/"
Write-Host "  2. Download preprocessed Python .dat files"
Write-Host "  3. Place in data/raw/deap/"
Write-Host ""
Write-Host "TUH EEG Corpus (Healthy controls, optional):"
Write-Host "  1. Register at https://isip.piconepress.com/projects/tuh_eeg/"
Write-Host "  2. Download normal recordings"
Write-Host "  3. Place in data/raw/tuh_normal/"
Write-Host ""
Write-Host "Done. Run:  python -m eegct all -c configs/real_data.yaml"
