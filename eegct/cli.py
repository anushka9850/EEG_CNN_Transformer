"""Command-line interface.

    python -m eegct synth                  # generate synthetic demo EDF data
    python -m eegct build   [-c cfg]       # load + harmonize + preprocess + window -> cache
    python -m eegct train   [-c cfg]       # train CNN-Transformer
    python -m eegct evaluate [-c cfg]      # test metrics, confusion matrix, XAI figures
    python -m eegct predict FILE [--ckpt]  # diagnose a new EEG file (+ Grad-CAM figure)
    python -m eegct all     [-c cfg]       # build + train + evaluate
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


def main(argv=None):
    ap = argparse.ArgumentParser(prog="eegct", description="Unified Explainable CNN-Transformer EEG framework")
    ap.add_argument("command", choices=["synth", "build", "train", "evaluate", "predict", "all"])
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

    if a.command == "synth":
        from .synthetic import generate
        mf = generate(a.out, seed=cfg["training"]["seed"], dur_s=a.duration, scale=a.scale)
        log.info("synthetic data written; manifest: %s", mf)
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
