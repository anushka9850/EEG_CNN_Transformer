# Transformer-Based Deep Learning Framework for Robust EEG Signal Classification

**A Unified Explainable CNN-Transformer Framework for Multi-Disease EEG Classification**
Department of Artificial Intelligence, GHRCE Nagpur, Session 2026-27.

This repository is a complete, working implementation of the methodology in the
progress-seminar presentation:

```
 Multiple EEG ──► Dataset ──► Preprocessing ──► Window ──► CNN feature ──► Transformer ──► Multi-disease ──► Grad-CAM +
 datasets         harmonization (filter, notch,    segmentation extraction      encoder          classification    attention XAI
 (EDF/BDF/BIDS)   (montage, fs,  ICA, normalize)                                                                 ──► Final prediction
                   labels)
```

Classes: **Healthy · Epilepsy · Alzheimer · Parkinson · Depression**
(plus an optional binary **Seizure vs Non-seizure** task on CHB-MIT).

---

## 1. Quick start (runs end-to-end in ~25 min on a laptop CPU, no downloads)

```bash
pip install -r requirements.txt

python -m eegct synth          # 1. writes ~95 realistic synthetic EDF recordings (5 "datasets")
python -m eegct all            # 2. harmonize + preprocess + window → train → evaluate → XAI figures
python -m eegct predict data/synthetic/openneuro_ad/openneuro_ad_alz00_00.edf   # 3. diagnose one file
streamlit run app/streamlit_app.py                                                  # 4. web app
python -m pytest -q            # unit tests
```

> **Important:** the synthetic data exists so that the *software* can be verified without the
> 100+ GB of real recordings. Its accuracy is **not** a scientific result. For your report, run the
> same pipeline on the real datasets (Section 3) — it is one command once they are downloaded.

## 2. Project structure

```
EEG-CNN-Transformer/
├── configs/
│   ├── default.yaml            # every setting (datasets, harmonization, preprocessing, model, training)
│   ├── real_data.yaml          # all real public datasets enabled
│   ├── seizure_chbmit.yaml     # binary seizure detection on CHB-MIT
│   └── synthetic_seizure.yaml  # seizure detection demo on synthetic data
├── eegct/
│   ├── datasets.py     # loaders: CHB-MIT, TUH, OpenNeuro (BIDS) AD/PD, MODMA, EEGMMIDB, DEAP, CSV manifest
│   ├── harmonize.py    # channel-name normalisation, common montage, resampling, label mapping
│   ├── preprocess.py   # reading any format, band-pass + notch, ICA, windowing, normalization
│   ├── data.py         # window cache builder, subject-wise split, augmentation
│   ├── model.py        # CNN-Transformer
│   ├── explain.py      # Grad-CAM (channel × time) + attention rollout + figures
│   ├── train.py        # training loop, evaluation, metrics, plots
│   ├── predict.py      # inference on a new recording
│   ├── synthetic.py    # realistic synthetic multi-dataset EEG generator (EDF writer)
│   └── cli.py          # command-line interface (python -m eegct ...)
├── app/streamlit_app.py  # web demo: upload EEG → prediction + explanation
├── scripts/download_data.sh
├── tests/test_pipeline.py
└── outputs/            # created by training: best_model.pt, metrics, figures, xai/
```

## 3. Using the real datasets

| Class | Dataset | Access | Put it in | Label source |
|---|---|---|---|---|
| Epilepsy | CHB-MIT Scalp EEG (PhysioNet, or Kaggle mirror) | free | `data/raw/chbmit` | `chbXX-summary.txt` seizure times |
| Healthy | TUH EEG Corpus – *normal* recordings | free registration at isip.piconepress.com | `data/raw/tuh_normal` | folder (files under `abnormal/` skipped) |
| Alzheimer (+ controls) | OpenNeuro **ds004504** | free | `data/raw/ds004504` | `participants.tsv` Group: A→Alzheimer, C→Healthy, F (FTD) skipped |
| Parkinson (+ controls) | OpenNeuro **ds002778** | free | `data/raw/ds002778` | subject id `sub-pd*` / `sub-hc*` |
| Depression (+ controls) | MODMA 128-ch resting EEG | free registration at modma.lzu.edu.cn | `data/raw/modma` | subjects xlsx (`type` MDD/HC) or `labels_file` CSV |
| Healthy (extra) | PhysioNet EEGMMIDB baseline runs R01/R02 | free | `data/raw/eegmmidb` | all healthy volunteers |
| Healthy (extra) | DEAP (preprocessed `.dat`) | free registration | `data/raw/deap` | all healthy volunteers |

```bash
bash scripts/download_data.sh                    # CHB-MIT subset, ds004504, ds002778, EEGMMIDB
# download TUH-normal and MODMA manually after registering, then:
python -m eegct all -c configs/real_data.yaml     # outputs in outputs_real/
python -m eegct all -c configs/seizure_chbmit.yaml
```

Any other dataset can be added **without code** through a CSV manifest
(`path,label,subject[,dataset][,fs][,seizures]`) — enable `datasets.manifest` in the config.
Supported file formats: EDF, EDF+, BDF, EEGLAB `.set`, FIF, BrainVision, GDF, MATLAB `.mat`, `.npy`, `.csv`.

EEGMMIDB and DEAP (listed in `fyp_data_1.pdf`) contain only healthy volunteers, so the framework uses
them as extra **Healthy-control** data from additional sites. Adding controls from several datasets
stops the model from learning "which dataset is this" instead of "which disease is this".

**Hardware for real data:** the full five-dataset run needs roughly 8–16 GB RAM for the window
cache. `max_files_per_subject`, `max_files`, `max_subjects` and `max_windows_per_recording` in the
config cap memory use. A GPU helps but is not required: training uses CUDA automatically when it is available.

## 4. Method details

### 4.1 Dataset harmonization (`harmonize.py`)
Each dataset differs in sampling rate (128–512 Hz), channel naming (`EEG FP1-REF`, `Fp1.`, `T3` vs
`T7`, EGI `E22`…), reference, and montage (CHB-MIT is bipolar, the others are referential).
* **Channel names** are normalised to 10-20 names (old T3/T4/T5/T6 → T7/T8/P7/P8, EGI HydroCel-128
  electrodes → nearest 10-20 site).
* **Common montage:** 18-channel longitudinal bipolar *double banana* (FP1-F7 … CZ-PZ). CHB-MIT is
  recorded in this montage, and any referential recording converts to it exactly (A − B). That makes it
  the one montage all datasets can share without approximation. (`monopolar_1020`, 19 ch with average
  reference, is also available.)
* **Common sampling rate:** 128 Hz, using anti-aliased polyphase resampling.
* **Label mapping:** dataset-specific group codes (A, C, HC, PD, MDD, …) map to the 5 unified classes.

### 4.2 Preprocessing (`preprocess.py`)
4th-order zero-phase Butterworth band-pass 0.5–45 Hz, then 50 Hz and 60 Hz notch filters (the datasets
come from both mains regions). Optional ICA ocular-artifact removal (`preprocessing.ica: true`) uses
FP1/FP2 as EOG proxies. Amplitudes are clipped, the signal is cut into 4 s windows (50 % overlap, 75 %
around seizures), and each window gets a per-channel robust z-score (median/IQR). The z-score removes
amplifier-gain differences between sites.

### 4.3 Model (`model.py`) — ≈118 k parameters
| Stage | Layer | Output (B = batch) |
|---|---|---|
| input | harmonized window | B × 18 ch × 512 samples |
| CNN | temporal conv 1×33 (8 filters) + BN + ELU + pool 2 | B × 8 × 18 × 256 |
| CNN | depthwise temporal conv 1×15 (16) + BN + ELU  ← **Grad-CAM layer** | B × 16 × 18 × 256 |
| CNN | spatial depthwise conv 18×1 (×4) + BN + ELU + pool 4 | B × 64 × 1 × 64 |
| CNN | separable conv 1×15 → 64 + BN + ELU + pool 2 | B × 64 × 1 × 32 |
| Transformer | [CLS] + learnable positional embedding, 3 pre-norm layers, 4 heads, d=64 | B × 33 × 64 |
| Head | concat([CLS], mean tokens) → MLP → softmax | B × 5 |

Each token covers 125 ms of EEG. The CNN learns frequency- and spatial filters. The Transformer
models long-range temporal dependencies across the 4 s window.

### 4.4 Training (`train.py`)
* **Subject-wise split** 70/15/15 (stratified by class). No person appears in both train and test.
  Splitting windows at random lets the same subject leak into both sets, and that leakage inflates the
  accuracy reported in many EEG papers.
* AdamW + One-Cycle LR, class-weighted cross-entropy with label smoothing, gradient clipping.
* Augmentation: amplitude scaling, time shift, Gaussian noise, channel dropout, polarity flip.
* Early stopping on validation macro-F1.
* Metrics: accuracy, balanced accuracy, macro-F1, ROC-AUC (one-vs-rest), per-class report, confusion
  matrix, **recording-level diagnosis** (window probabilities averaged per recording), and **per-dataset
  accuracy** (a generalisation check).

### 4.5 Explainability (`explain.py`)
* **Grad-CAM** on the last CNN layer that still keeps the channel × time layout. The result is a
  heat-map that shows which electrodes and which moments drove the decision.
* **Attention rollout** (Abnar & Zuidema 2020) across all Transformer layers from the [CLS] token. It
  shows which time segments the Transformer relied on.
* Per-class **channel importance** (averaged over test windows) is saved to `outputs/xai/top_channels_per_class.json`.

## 5. Outputs
After `python -m eegct all` the `outputs/` folder contains:
`best_model.pt` (weights + config + class names), `history.json`, `training_curves.png`,
`test_metrics.json`, `confusion_matrix.png`, `split.npz`, and `xai/gradcam_<Class>_0.png` for every class.

## 6. Results on the synthetic demo data
See `RESULTS.md` (produced by the run included in this package). Real-data results will differ and
are expected to be lower. Cross-subject, cross-dataset EEG diagnosis is a hard problem.

## 7. Mapping to the project objectives
| Objective (slide 7) | Where |
|---|---|
| Integrate heterogeneous public EEG datasets | `datasets.py` (7 loaders + manifest) |
| Preprocess and harmonize EEG signals | `harmonize.py`, `preprocess.py` |
| Extract spatial features using CNN | `model.py` temporal + spatial conv blocks |
| Capture temporal dependencies using Transformer | `model.py` encoder with multi-head attention |
| Classify multiple neurological disorders | 5-class head, `train.py` |
| Improve interpretability using Grad-CAM | `explain.py`, `app/streamlit_app.py` |

## 8. Limitations
* Each disease comes mostly from one dataset, so disease and recording site are partly confounded.
  Harmonization and multi-site healthy controls reduce this but do not remove it. Report the
  per-dataset accuracy and, if possible, validate on an external dataset.
* Research prototype, not a medical device.
# EEG-CNN-Transformer
