#!/usr/bin/env bash
# One-shot demo: install, generate synthetic data, train, evaluate, predict, test.
set -e
pip install -r requirements.txt
python -m eegct synth
python -m eegct all
python -m eegct predict data/synthetic/openneuro_ad/openneuro_ad_alz00_00.edf
python -m pytest -q
echo "Launch the web app with:  streamlit run app/streamlit_app.py"
