"""Training and evaluation."""
from __future__ import annotations

import json
import logging
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score, roc_auc_score)
from torch.utils.data import DataLoader

from .data import WindowDataset, cache_path, load_cache, subject_split
from .model import build_model, count_params

log = logging.getLogger(__name__)


def set_seed(s):
    np.random.seed(s)
    torch.manual_seed(s)


@torch.no_grad()
def predict_proba(model, X: np.ndarray, batch=256) -> np.ndarray:
    model.eval()
    dev = next(model.parameters()).device
    out = []
    for i in range(0, len(X), batch):
        xb = torch.from_numpy(X[i:i + batch].astype(np.float32)).to(dev)
        out.append(torch.softmax(model(xb), 1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, model.hparams["n_classes"]))


def train(cfg: dict) -> Path:
    tc = cfg["training"]
    set_seed(tc["seed"])
    if tc.get("num_threads"):
        torch.set_num_threads(tc["num_threads"])
    d = load_cache(cache_path(cfg))
    X, y, subj = d["X"], d["y"], d["subject"]
    classes = [str(c) for c in d["classes"]]
    tr, va, te = subject_split(y, subj, tc["split"], tc["seed"])
    log.info("windows: train %d | val %d | test %d  (subjects %d/%d/%d)", len(tr), len(va), len(te),
             len(set(subj[tr])), len(set(subj[va])), len(set(subj[te])))
    out = Path(cfg["paths"]["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "split.npz", train=tr, val=va, test=te)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg, X.shape[1], X.shape[2], len(classes)).to(dev)
    log.info("device: %s", dev)
    log.info("model parameters: %s", f"{count_params(model):,}")
    counts = np.bincount(y[tr], minlength=len(classes)).astype(np.float32)
    weights = torch.tensor(counts.sum() / (len(classes) * np.maximum(counts, 1)), dtype=torch.float32)
    crit = nn.CrossEntropyLoss(weight=weights.to(dev), label_smoothing=tc["label_smoothing"])
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=tc["lr"], epochs=tc["epochs"],
                                                steps_per_epoch=max(1, int(np.ceil(len(tr) / tc["batch_size"]))))
    dl = DataLoader(WindowDataset(X[tr], y[tr], augment=tc["augment"]), batch_size=tc["batch_size"],
                    shuffle=True, drop_last=len(tr) > tc["batch_size"])

    best, best_ep, history = -1.0, 0, []
    ckpt = out / "best_model.pt"
    for ep in range(1, tc["epochs"] + 1):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        for xb, yb in dl:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item() * len(yb)
            n += len(yb)
        pv = predict_proba(model, X[va])
        f1 = f1_score(y[va], pv.argmax(1), average="macro") if len(va) else 0.0
        acc = accuracy_score(y[va], pv.argmax(1)) if len(va) else 0.0
        history.append(dict(epoch=ep, loss=tot / max(n, 1), val_acc=acc, val_f1=f1))
        log.info("epoch %3d  loss %.4f  val_acc %.4f  val_macroF1 %.4f  (%.1fs)", ep, tot / max(n, 1), acc, f1,
                 time.time() - t0)
        if f1 > best:
            best, best_ep = f1, ep
            torch.save(dict(state_dict={k: v.cpu() for k, v in model.state_dict().items()}, hparams=model.hparams, classes=classes,
                            channels=[str(c) for c in d["channels"]], fs=int(d["fs"]), config=cfg,
                            epoch=ep, val_f1=f1), ckpt)
        elif ep - best_ep >= tc["patience"]:
            log.info("early stopping (best epoch %d, val macro-F1 %.4f)", best_ep, best)
            break
    json.dump(history, open(out / "history.json", "w"), indent=1)
    _plot_history(history, out / "training_curves.png")
    return ckpt


def load_model(ckpt_path) -> tuple:
    from .model import CNNTransformer
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    m = CNNTransformer(**ck["hparams"])
    m.load_state_dict(ck["state_dict"])
    m.eval()
    return m, ck


def evaluate(cfg: dict, ckpt_path=None) -> dict:
    out = Path(cfg["paths"]["output_dir"])
    model, ck = load_model(ckpt_path or out / "best_model.pt")
    d = load_cache(cache_path(cfg))
    sp = np.load(out / "split.npz")
    te = sp["test"]
    X, y = d["X"][te], d["y"][te]
    classes = ck["classes"]
    labels = list(range(len(classes)))
    P = predict_proba(model, X)
    pred = P.argmax(1)
    res = dict(
        n_test_windows=int(len(te)), n_test_subjects=int(len(set(d["subject"][te]))),
        window_accuracy=float(accuracy_score(y, pred)),
        window_balanced_accuracy=float(balanced_accuracy_score(y, pred)),
        window_macro_f1=float(f1_score(y, pred, average="macro", labels=labels, zero_division=0)),
        per_class=classification_report(y, pred, labels=labels, target_names=classes, output_dict=True,
                                        zero_division=0),
        confusion_matrix=confusion_matrix(y, pred, labels=labels).tolist(),
    )
    try:
        present = sorted(set(y))
        if len(classes) == 2:
            res["window_roc_auc"] = float(roc_auc_score(y, P[:, 1]))
        elif len(present) == len(classes):
            res["window_roc_auc_ovr"] = float(roc_auc_score(y, P, multi_class="ovr", labels=labels))
    except ValueError:
        pass
    # recording-level diagnosis: average window probabilities per recording (clinically relevant)
    rec = d["recording"][te]
    if cfg["task"] != "seizure":
        agg = defaultdict(list)
        for i, r in enumerate(rec):
            agg[r].append(i)
        ry, rp = [], []
        for r, idx in agg.items():
            ry.append(Counter(y[idx]).most_common(1)[0][0])
            rp.append(int(P[idx].mean(0).argmax()))
        res["recording_accuracy"] = float(accuracy_score(ry, rp))
        res["recording_macro_f1"] = float(f1_score(ry, rp, average="macro", labels=labels, zero_division=0))
        res["n_test_recordings"] = len(agg)
    # generalisation: accuracy per source dataset
    ds = d["dataset"][te]
    res["per_dataset_accuracy"] = {str(k): float(accuracy_score(y[ds == k], pred[ds == k])) for k in np.unique(ds)}
    json.dump(res, open(out / "test_metrics.json", "w"), indent=1)
    _plot_cm(np.array(res["confusion_matrix"]), classes, out / "confusion_matrix.png")
    log.info("TEST  acc %.4f | balanced acc %.4f | macro-F1 %.4f", res["window_accuracy"],
             res["window_balanced_accuracy"], res["window_macro_f1"])
    if "recording_accuracy" in res:
        log.info("TEST  recording-level acc %.4f (%d recordings)", res["recording_accuracy"], res["n_test_recordings"])
    log.info("\n%s", classification_report(y, pred, labels=labels, target_names=classes, zero_division=0))
    return res


def _plot_history(h, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ep = [r["epoch"] for r in h]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    ax[0].plot(ep, [r["loss"] for r in h], color="#1f3b73")
    ax[0].set_title("training loss")
    ax[1].plot(ep, [r["val_acc"] for r in h], label="val accuracy", color="#1f3b73")
    ax[1].plot(ep, [r["val_f1"] for r in h], label="val macro-F1", color="#e07b39")
    ax[1].legend()
    ax[1].set_title("validation")
    for a in ax:
        a.set_xlabel("epoch")
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _plot_cm(cm, classes, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f"{cm[i, j]}\n{cmn[i, j]:.0%}", ha="center", va="center", fontsize=8,
                    color="white" if cmn[i, j] > 0.5 else "black")
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=30, ha="right")
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels(classes)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title("Test confusion matrix (windows)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
