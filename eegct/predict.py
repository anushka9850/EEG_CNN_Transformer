"""Inference on a new EEG file (any format, any channel naming, any sampling rate)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .explain import explain_window, plot_explanation
from .preprocess import Recording, read_any, preprocess_recording, segment, normalize_windows
from .train import load_model, predict_proba


def predict_file(ckpt_path, eeg_path, fs=None, overlap=0.0, explain_path=None, model_bundle=None):
    model, ck = model_bundle or load_model(ckpt_path)
    cfg = ck["config"]
    data, rfs, names = read_any(eeg_path, fs=fs)
    return predict_array(model, ck, data, rfs, names, overlap, explain_path, source=str(eeg_path))


def predict_array(model, ck, data, fs, names, overlap=0.0, explain_path=None, source=""):
    cfg = ck["config"]
    rec = Recording(data, fs, names, "?", "?", "?", source)
    sig = preprocess_recording(rec, cfg)
    X, starts = segment(sig, ck["fs"], cfg["windows"]["length_s"], overlap)
    if len(X) == 0:
        raise ValueError(f"recording shorter than one window ({cfg['windows']['length_s']} s)")
    X = normalize_windows(X, cfg["preprocessing"].get("normalization", "robust_zscore"))
    P = predict_proba(model, X)
    classes = ck["classes"]
    mean = P.mean(0)
    k = int(mean.argmax())
    best_w = int(P[:, k].argmax())
    exp = explain_window(model, X[best_w], k)
    result = dict(
        file=source, prediction=classes[k], confidence=float(mean[k]),
        probabilities={c: float(p) for c, p in zip(classes, mean)},
        n_windows=int(len(X)), window_starts_s=starts.tolist(), window_probabilities=P.tolist(),
        explained_window=best_w,
        top_channels=[ck["channels"][i] for i in np.argsort(-exp["channel_importance"])[:5]],
    )
    if explain_path:
        Path(explain_path).parent.mkdir(parents=True, exist_ok=True)
        plot_explanation(X[best_w], exp, ck["channels"], classes, ck["fs"],
                         title=f"{Path(source).name}: {classes[k]} ({mean[k]:.0%}) - window @ {starts[best_w]:.0f}s",
                         path=explain_path)
    return result, X, exp
