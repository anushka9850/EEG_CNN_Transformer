#!/usr/bin/env bash
# Real-data pipeline: download check -> train -> evaluate.
set -e
echo "=== EEG CNN-Transformer: Real Data Pipeline ==="

# Check if at least one dataset exists
if [ ! -d "data/raw" ]; then
    echo "No data/raw/ directory found. Run scripts/download_data.sh first."
    exit 1
fi

found=0
for d in chbmit eegmmidb ds004504 ds002778 modma deap tuh_normal; do
    if [ -d "data/raw/$d" ]; then
        echo "  Found: data/raw/$d"
        found=1
    fi
done
if [ $found -eq 0 ]; then
    echo "No datasets found in data/raw/. Run scripts/download_data.sh first."
    exit 1
fi

echo ""
echo "=== Data check ==="
python -m eegct check -c configs/real_data.yaml

echo ""
echo "=== Build + Train + Evaluate ==="
python -m eegct all -c configs/real_data.yaml

echo ""
echo "Done. Results in outputs_real/"
