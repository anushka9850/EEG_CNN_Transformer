"""Command-line interface.

    python -m eegct synth                  # generate synthetic demo EDF data
    python -m eegct build   [-c cfg]       # load + harmonize + preprocess + window -> cache
    python -m eegct train   [-c cfg]       # train CNN-Transformer
    python -m eegct evaluate [-c cfg]      # test metrics, confusion matrix, XAI figures
    python -m eegct predict FILE [--ckpt]  # diagnose a new EEG file (+ Grad-CAM figure)
    python -m eegct all     [-c cfg]       # build + train + evaluate
    python -m eegct check   [-c cfg]       # print dataset summary (no training)
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
from pathlib import Path

import numpy as np
import yaml


def load_config(path: str | None, overrides: list[str] | None = None) -> dict:
    base = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"
    cfg = yaml.safe_load(open(base))
    if path and Path(path).resolve() != base:
        _merge(cfg, yaml.safe_load(open(path)) or {})
    for ov in overrides or []:  # key.sub=value
        k, v = ov.split("=", 1)
        d = cfg
        ks = k.split(".")
        for kk in ks[:-1]:
            d = d.setdefault(kk, {})
        d[ks[-1]] = yaml.safe_load(v)
    return cfg


def _merge(a, b):
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(a.get(k), dict):
            _merge(a[k], v)
        else:
            a[k] = copy.deepcopy(v)


def xai_figures(cfg, n_per_class=1):
    """Save Grad-CAM + attention figures for correctly classified test windows of each class."""
    import torch
    from .data import cache_path, load_cache
    from .explain import explain_window, plot_explanation
    from .train import load_model, predict_proba
    out = Path(cfg["paths"]["output_dir"])
    model, ck = load_model(out / "best_model.pt")
    d = load_cache(cache_path(cfg))
    te = np.load(out / "split.npz")["test"]
    X, y = d["X"][te], d["y"][te]
    P = predict_proba(model, X)
    (out / "xai").mkdir(exist_ok=True)
    summary = {}
    for c, name in enumerate(ck["classes"]):
        idx = np.where((y == c) & (P.argmax(1) == c))[0]
        if len(idx) == 0:
            continue
        # class-level channel importance averaged over up to 50 correct windows
        imp = []
        for i in idx[np.argsort(-P[idx, c])][:50]:
            imp.append(explain_window(model, X[i].astype(np.float32), c)["channel_importance"])
        imp = np.mean(imp, 0)
        summary[name] = {ck["channels"][k]: round(float(imp[k]), 3) for k in np.argsort(-imp)[:5]}
        for j, i in enumerate(idx[np.argsort(-P[idx, c])][:n_per_class]):
            x = X[i].astype(np.float32)
            e = explain_window(model, x, c)
            plot_explanation(x, e, ck["channels"], ck["classes"], ck["fs"],
                             title=f"True: {name}  |  Pred: {name} ({P[i, c]:.0%})",
                             path=out / "xai" / f"gradcam_{name}_{j}.png")
    json.dump(summary, open(out / "xai" / "top_channels_per_class.json", "w"), indent=1)
    logging.getLogger(__name__).info("XAI figures -> %s", out / "xai")


def shortcut_check(cfg: dict):
    """After evaluation: per-source-dataset accuracy and within-dataset Healthy-vs-disease accuracy."""
    from .data import cache_path, load_cache
    from .train import load_model, predict_proba
    from sklearn.metrics import accuracy_score
    log = logging.getLogger(__name__)
    out = Path(cfg["paths"]["output_dir"])
    model, ck = load_model(out / "best_model.pt")
    d = load_cache(cache_path(cfg))
    sp = np.load(out / "split.npz")
    te = sp["test"]
    X, y, ds = d["X"][te], d["y"][te], d["dataset"][te]
    classes = list(ck["classes"])
    P = predict_proba(model, X)
    pred = P.argmax(1)
    result = {}
    # (a) per-source-dataset accuracy
    per_ds = {}
    for dsn in np.unique(ds):
        mask = ds == dsn
        per_ds[str(dsn)] = float(accuracy_score(y[mask], pred[mask]))
    result["per_dataset_accuracy"] = per_ds
    # (b) within-dataset Healthy-vs-disease accuracy
    # Datasets that have both Healthy and a disease class
    within_ds_map = {
        "openneuro_ad": "Alzheimer", "ds004504": "Alzheimer",
        "openneuro_pd": "Parkinson", "ds002778": "Parkinson",
        "modma": "Depression",
    }
    healthy_idx = classes.index("Healthy") if "Healthy" in classes else None
    within = {}
    for dsn, disease in within_ds_map.items():
        if disease not in classes or healthy_idx is None:
            continue
        disease_idx = classes.index(disease)
        mask = ds == dsn
        if not mask.any():
            continue
        # Within this dataset, only consider Healthy vs the disease
        submask = np.isin(y[mask], [healthy_idx, disease_idx])
        if not submask.any() or len(set(y[mask][submask])) < 2:
            continue
        y_sub = (y[mask][submask] == disease_idx).astype(int)
        p_sub = (pred[mask][submask] == disease_idx).astype(int)
        within[str(dsn)] = float(accuracy_score(y_sub, p_sub))
    result["within_dataset_accuracy"] = within
    # Write output
    json.dump(result, open(out / "shortcut_check.json", "w"), indent=1)
    log.info("shortcut check -> %s", out / "shortcut_check.json")
    # Warn if within-dataset accuracy is much lower than overall
    overall_acc = float(accuracy_score(y, pred))
    for dsn, acc in within.items():
        if overall_acc - acc > 0.15:
            log.warning("SHORTCUT WARNING: within-dataset acc for %s (%.3f) is much lower than overall (%.3f) - "
                        "possible dataset-level shortcut!", dsn, acc, overall_acc)
    return result


def generate_results_md(cfg: dict):
    """Generate RESULTS_REAL.md from test_metrics.json and shortcut_check.json."""
    out = Path(cfg["paths"]["output_dir"])
    metrics_path = out / "test_metrics.json"
    shortcut_path = out / "shortcut_check.json"
    if not metrics_path.exists():
        return
    metrics = json.load(open(metrics_path))
    lines = ["# Real Data Results\n",
             "Auto-generated from `test_metrics.json`.\n",
             "## Overall\n",
             f"| Metric | Value |",
             f"|--------|-------|",
             f"| Test windows | {metrics['n_test_windows']} |",
             f"| Test subjects | {metrics['n_test_subjects']} |",
             f"| Window accuracy | {metrics['window_accuracy']:.4f} |",
             f"| Balanced accuracy | {metrics['window_balanced_accuracy']:.4f} |",
             f"| Macro F1 | {metrics['window_macro_f1']:.4f} |"]
    if "recording_accuracy" in metrics:
        lines.append(f"| Recording accuracy | {metrics['recording_accuracy']:.4f} |")
    if "window_roc_auc_ovr" in metrics:
        lines.append(f"| ROC-AUC (OvR) | {metrics['window_roc_auc_ovr']:.4f} |")
    # Per-class
    lines += ["\n## Per-class metrics\n",
              "| Class | Precision | Recall | F1 | Support |",
              "|-------|-----------|--------|-----|---------|"]
    pc = metrics.get("per_class", {})
    for cls in sorted(pc.keys()):
        if cls in ("accuracy", "macro avg", "weighted avg"):
            continue
        c = pc[cls]
        lines.append(f"| {cls} | {c['precision']:.3f} | {c['recall']:.3f} | {c['f1-score']:.3f} | {c['support']} |")
    # Per-dataset accuracy
    if "per_dataset_accuracy" in metrics:
        lines += ["\n## Per-dataset accuracy\n",
                  "| Dataset | Accuracy |",
                  "|---------|----------|"]
        for ds, acc in metrics["per_dataset_accuracy"].items():
            lines.append(f"| {ds} | {acc:.4f} |")
    # Shortcut check
    if shortcut_path.exists():
        sc = json.load(open(shortcut_path))
        if sc.get("within_dataset_accuracy"):
            lines += ["\n## Shortcut check (within-dataset accuracy)\n",
                      "| Dataset | Within-dataset accuracy |",
                      "|---------|------------------------|"]
            for ds, acc in sc["within_dataset_accuracy"].items():
                lines.append(f"| {ds} | {acc:.4f} |")
    root = Path(__file__).resolve().parent.parent
    (root / "RESULTS_REAL.md").write_text("\n".join(lines) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="eegct", description="Unified Explainable CNN-Transformer EEG framework")
    ap.add_argument("command", choices=["synth", "build", "train", "evaluate", "predict", "all", "check"])
    ap.add_argument("file", nargs="?", help="EEG file for 'predict'")
    ap.add_argument("-c", "--config", default=None)
    ap.add_argument("-s", "--set", nargs="*", default=[], help="override config: key.sub=value")
    ap.add_argument("--force", action="store_true", help="rebuild cache")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--fs", type=float, default=None, help="sampling rate for .mat/.npy/.csv input")
    ap.add_argument("--out", default="data/synthetic", help="output folder for 'synth'")
    ap.add_argument("--scale", type=float, default=1.0, help="synthetic: subjects multiplier")
    ap.add_argument("--duration", type=float, default=180.0, help="synthetic: seconds per recording")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    log = logging.getLogger("eegct")
    cfg = load_config(a.config, a.set)

    # Set up file logging for run_log.txt
    out_dir = Path(cfg["paths"]["output_dir"])
    if a.command in ("build", "train", "evaluate", "all"):
        out_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(out_dir / "run_log.txt", mode="w")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"))
        logging.getLogger().addHandler(fh)

    if a.command == "synth":
        from .synthetic import generate
        mf = generate(a.out, seed=cfg["training"]["seed"], dur_s=a.duration, scale=a.scale)
        log.info("synthetic data written; manifest: %s", mf)
        return
    if a.command == "check":
        from .datasets import check_datasets
        check_datasets(cfg)
        return
    if a.command in ("build", "all"):
        from .data import build_cache
        build_cache(cfg, force=a.force or a.command == "all")
    if a.command in ("train", "all"):
        from .train import train
        train(cfg)
    if a.command in ("evaluate", "all"):
        from .train import evaluate
        evaluate(cfg, a.ckpt)
        xai_figures(cfg)
        try:
            shortcut_check(cfg)
        except Exception as e:
            log.warning("shortcut check failed: %s", e)
        try:
            generate_results_md(cfg)
        except Exception as e:
            log.warning("RESULTS_REAL.md generation failed: %s", e)
    if a.command == "predict":
        if not a.file:
            ap.error("predict needs an EEG file")
        from .predict import predict_file
        ckpt = a.ckpt or Path(cfg["paths"]["output_dir"]) / "best_model.pt"
        fig = Path(cfg["paths"]["output_dir"]) / "predictions" / (Path(a.file).stem + "_gradcam.png")
        res, _, _ = predict_file(ckpt, a.file, fs=a.fs, explain_path=fig)
        short = {k: res[k] for k in ("file", "prediction", "confidence", "probabilities", "n_windows", "top_channels")}
        print(json.dumps(short, indent=2))
        log.info("explanation figure: %s", fig)


if __name__ == "__main__":
    sys.exit(main())
