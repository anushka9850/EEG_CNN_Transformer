# Real Data Results

Auto-generated from `test_metrics.json`.

## Overall

| Metric | Value |
|--------|-------|
| Test windows | 1519 |
| Test subjects | 13 |
| Window accuracy | 0.7479 |
| Balanced accuracy | 0.8436 |
| Macro F1 | 0.8125 |
| Recording accuracy | 0.9333 |
| ROC-AUC (OvR) | 0.9354 |

## Per-class metrics

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|-----|---------|
| Alzheimer | 0.989 | 0.983 | 0.986 | 178.0 |
| Depression | 0.994 | 1.000 | 0.997 | 178.0 |
| Epilepsy | 0.977 | 0.474 | 0.638 | 540.0 |
| Healthy | 0.552 | 0.800 | 0.653 | 445.0 |
| Parkinson | 0.668 | 0.961 | 0.788 | 178.0 |

## Per-dataset accuracy

| Dataset | Accuracy |
|---------|----------|
| chbmit | 0.4741 |
| modma | 0.6966 |
| openneuro_ad | 0.9831 |
| openneuro_pd | 0.9738 |
| tuh | 0.9700 |

## Shortcut check (within-dataset accuracy)

| Dataset | Within-dataset accuracy |
|---------|------------------------|
| openneuro_pd | 0.9738 |
| modma | 1.0000 |
