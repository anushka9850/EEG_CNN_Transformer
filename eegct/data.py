"""Build the harmonized window cache and provide PyTorch datasets / subject-wise splits."""
from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .datasets import iter_recordings
from .harmonize import target_channels
from .preprocess import recording_to_windows

log = logging.getLogger(__name__)

# Fixed canonical order - absent classes are dropped but the relative order is kept.
ALL_CLASSES = ["Healthy", "Epilepsy", "Alzheimer", "Parkinson", "Depression"]


def class_names(cfg: dict) -> list[str]:
    if cfg["task"] == "seizure":
        return ["Non-seizure", "Seizure"]
    return list(cfg["classes"])


def cache_path(cfg: dict) -> Path:
    return Path(cfg["paths"]["cache_dir"]) / f"windows_{cfg['task']}_{cfg['harmonization']['montage']}.npz"


def build_cache(cfg: dict, force: bool = False) -> Path:
    out = cache_path(cfg)
    if out.exists() and not force:
        log.info("cache exists: %s (use --force to rebuild)", out)
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    classes = class_names(cfg)
    cidx = {c: i for i, c in enumerate(classes)}
    rng = np.random.default_rng(cfg["training"]["seed"])
    Xs, ys, subs, dss, recs = [], [], [], [], []
    n_rec = 0
    skipped = Counter()
    for rec in iter_recordings(cfg):
        try:
            X, labels = recording_to_windows(rec, cfg, rng)
        except Exception as e:
            log.warning("failed %s: %s", rec.path, e)
            continue
        keep = [i for i, l in enumerate(labels) if l in cidx]
        for l in labels:
            if l not in cidx:
                skipped[l] += 1
        if not keep:
            continue
        Xs.append(X[keep].astype(np.float16))
        ys += [cidx[labels[i]] for i in keep]
        subs += [rec.subject] * len(keep)
        dss += [rec.dataset] * len(keep)
        recs += [f"{rec.subject}|{Path(rec.path).name}"] * len(keep)
        n_rec += 1
        if n_rec % 10 == 0:
            log.info("  processed %d recordings, %d windows", n_rec, len(ys))
    if not Xs:
        raise RuntimeError("no windows produced - check dataset paths in the config")
    if skipped:
        log.warning("labels not in class list were dropped: %s", dict(skipped))
    X = np.concatenate(Xs)

    # Dynamic n_classes: keep only classes that actually have windows
    present_indices = sorted(set(ys))
    if cfg["task"] != "seizure" and len(present_indices) < len(classes):
        old2new = {old: new for new, old in enumerate(present_indices)}
        classes = [classes[i] for i in present_indices]
        ys = [old2new[y_] for y_ in ys]
        log.info("dynamic classes (dropped absent): %s", classes)

    np.savez_compressed(out, X=X, y=np.array(ys, np.int64), subject=np.array(subs),
                        dataset=np.array(dss), recording=np.array(recs),
                        classes=np.array(classes), channels=np.array(target_channels(cfg["harmonization"]["montage"])),
                        fs=cfg["harmonization"]["target_fs"])
    log.info("saved %s: X=%s, %d recordings", out, X.shape, n_rec)
    log.info("class counts: %s", {classes[k]: v for k, v in sorted(Counter(ys).items())})
    return out


def load_cache(path: Path) -> dict:
    d = np.load(path, allow_pickle=False)
    return {k: d[k] for k in d.files}


def subject_split(y: np.ndarray, subjects: np.ndarray, fractions, seed: int):
    """Subject-wise stratified split: no subject appears in more than one split
    (prevents the window-level leakage that inflates many EEG papers' accuracy).
    Each subject is assigned by its majority class so every class is represented."""
    rng = np.random.default_rng(seed)
    subj_label = {}
    for s in np.unique(subjects):
        subj_label[s] = Counter(y[subjects == s]).most_common(1)[0][0]
    parts = [[], [], []]
    for c in sorted(set(subj_label.values())):
        ss = np.array(sorted(s for s, l in subj_label.items() if l == c))
        rng.shuffle(ss)
        n = len(ss)
        n_te = max(1, int(round(n * fractions[2]))) if n >= 3 else 0
        n_va = max(1, int(round(n * fractions[1]))) if n >= 3 else 0
        parts[2] += list(ss[:n_te])
        parts[1] += list(ss[n_te:n_te + n_va])
        parts[0] += list(ss[n_te + n_va:])

    # Assert no subject overlap across splits
    train_subj = set(parts[0])
    val_subj = set(parts[1])
    test_subj = set(parts[2])
    assert not (train_subj & val_subj), "subject overlap: train & val"
    assert not (train_subj & test_subj), "subject overlap: train & test"
    assert not (val_subj & test_subj), "subject overlap: val & test"

    return [np.where(np.isin(subjects, p))[0] for p in parts]


class WindowDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, augment: bool = False):
        self.X, self.y, self.augment = X, y, augment

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        x = self.X[i].astype(np.float32)
        if self.augment:
            x = augment(x)
        return torch.from_numpy(x), int(self.y[i])


def augment(x: np.ndarray) -> np.ndarray:
    """Label-preserving EEG augmentations."""
    rng = np.random
    x = x * rng.uniform(0.8, 1.2)                                   # amplitude scaling
    x = np.roll(x, rng.randint(-x.shape[1] // 8, x.shape[1] // 8), axis=1)  # time shift
    x = x + rng.normal(0, 0.1, x.shape).astype(np.float32)          # gaussian noise
    if rng.rand() < 0.3:                                            # channel dropout
        x[rng.randint(0, x.shape[0])] = 0
    if rng.rand() < 0.2:                                            # sign flip (polarity)
        x = -x
    return x.astype(np.float32)
