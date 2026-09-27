#!/usr/bin/env bash
# Download the freely available datasets into data/raw/.
# TUH (Healthy) and MODMA (Depression) require free registration - see README.
set -e
mkdir -p data/raw && cd data/raw

# ---- CHB-MIT (Epilepsy) ~43 GB full. Start with a few patients:
for p in chb01 chb02 chb03 chb05 chb08 chb10; do
  wget -r -N -c -np -nH --cut-dirs=3 -P chbmit "https://physionet.org/files/chbmit/1.0.0/$p/"
done
# (or the Kaggle mirror: kaggle datasets download -d beyzanurdin/chb-mit-scalp-eeg-database)

# ---- OpenNeuro ds004504 (Alzheimer / FTD / Controls)  and ds002778 (Parkinson / Controls)
pip install -q awscli || true
aws s3 sync --no-sign-request s3://openneuro.org/ds004504 ds004504 --exclude "derivatives/*"
aws s3 sync --no-sign-request s3://openneuro.org/ds002778 ds002778
# alternative: pip install openneuro-py && openneuro-py download --dataset ds004504 --target-dir ds004504

# ---- EEGMMIDB baseline runs (extra Healthy controls, ~ 2 files per subject)
for s in $(seq -f "%03g" 1 40); do
  for r in 01 02; do
    wget -N -c -P eegmmidb/S$s "https://physionet.org/files/eegmmidb/1.0.0/S$s/S${s}R$r.edf"
  done
done
echo "Done. Now:  python -m eegct all -c configs/real_data.yaml"
