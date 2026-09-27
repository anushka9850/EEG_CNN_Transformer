"""Explainability: Grad-CAM (channel x time) and Transformer attention rollout."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    """Grad-CAM on the last CNN layer that still has the (channel x time) layout
    (``model.temporal2``). Output: heat-map (C, T) in [0,1] showing which electrodes
    and which moments drove the prediction for the chosen class."""

    def __init__(self, model):
        self.model = model
        self.acts = None
        self.grads = None
        layer = model.temporal2
        self.handles = [layer.register_forward_hook(self._fwd),
                        layer.register_full_backward_hook(self._bwd)]

    def remove(self):
        for h in self.handles:
            h.remove()

    def _fwd(self, m, i, o):
        self.acts = o

    def _bwd(self, m, gi, go):
        self.grads = go[0]

    def __call__(self, x: torch.Tensor, class_idx: int | None = None):
        self.model.eval()
        x = x.clone().requires_grad_(True)
        logits = self.model(x)
        if class_idx is None:
            class_idx = int(logits.argmax(1)[0])
        self.model.zero_grad()
        logits[:, class_idx].sum().backward()
        w = self.grads.mean(dim=(2, 3), keepdim=True)            # (B,K,1,1)
        cam = F.relu((w * self.acts).sum(1))                     # (B,C,T)
        cam = F.interpolate(cam.unsqueeze(1), size=(x.shape[1], x.shape[2]), mode="bilinear",
                            align_corners=False)[:, 0]           # back to input resolution
        cam = cam[0].detach().cpu().numpy()
        cam = _smooth(cam, 9)
        cam = cam / (cam.max() + 1e-8)
        return cam, class_idx, torch.softmax(logits, 1)[0].detach().cpu().numpy()


def _smooth(a: np.ndarray, k: int) -> np.ndarray:
    ker = np.ones(k) / k
    return np.stack([np.convolve(r, ker, mode="same") for r in a])


def attention_rollout(attn_maps, n_times: int) -> np.ndarray:
    """Attention rollout (Abnar & Zuidema, 2020): propagate attention through layers
    (averaging heads, adding identity for residuals) and read the [CLS] row.
    Returns importance per time sample (length n_times) in [0,1]."""
    rollout = None
    for a in attn_maps:
        a = a[0].mean(0).cpu().numpy()                 # (L,L)
        a = a + np.eye(a.shape[0])
        a = a / a.sum(-1, keepdims=True)
        rollout = a if rollout is None else a @ rollout
    cls = rollout[0, 1:]
    cls = cls / (cls.max() + 1e-8)
    return np.interp(np.linspace(0, len(cls) - 1, n_times), np.arange(len(cls)), cls)


def explain_window(model, x: np.ndarray, class_idx: int | None = None):
    """x: (C,T) window -> dict(cam, channel_importance, time_attention, probs, class_idx)."""
    gc = GradCAM(model)
    xt = torch.from_numpy(x[None].astype(np.float32))
    try:
        cam, ci, probs = gc(xt, class_idx)
    finally:
        gc.remove()
    with torch.no_grad():
        model(xt)
    att = attention_rollout(model.attention_maps(), x.shape[-1])
    ch_imp = cam.mean(1)
    ch_imp = ch_imp / (ch_imp.max() + 1e-8)
    return dict(cam=cam, channel_importance=ch_imp, time_attention=att, probs=probs, class_idx=ci)


def plot_explanation(x, exp, channels, classes, fs, title="", path=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    C, T = x.shape
    t = np.arange(T) / fs
    fig = plt.figure(figsize=(13, 8))
    gs = fig.add_gridspec(3, 2, width_ratios=[4, 1], height_ratios=[3, 3, 1], hspace=0.35, wspace=0.28)
    ax = fig.add_subplot(gs[0, 0])
    off = 6.0
    for i in range(C):
        ax.plot(t, x[i] + off * (C - 1 - i), lw=0.6, color="#1f3b73")
    ax.imshow(exp["cam"], aspect="auto", cmap="Reds", alpha=0.45,
              extent=[0, T / fs, -off / 2, off * (C - 0.5)], origin="upper")
    ax.set_yticks([off * (C - 1 - i) for i in range(C)])
    ax.set_yticklabels(channels, fontsize=7)
    ax.set_xlim(0, T / fs)
    ax.set_title(title or "EEG window with Grad-CAM overlay")
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax)
    ax2.imshow(exp["cam"], aspect="auto", cmap="inferno", extent=[0, T / fs, C, 0])
    ax2.set_yticks(np.arange(C) + 0.5)
    ax2.set_yticklabels(channels, fontsize=7)
    ax2.set_title("Grad-CAM heat-map (channel x time)")
    ax3 = fig.add_subplot(gs[2, 0], sharex=ax)
    ax3.fill_between(t, exp["time_attention"], color="#e07b39", alpha=0.7)
    ax3.set_ylabel("attn")
    ax3.set_xlabel("time (s)")
    ax3.set_title("Transformer attention rollout", fontsize=9)
    ax4 = fig.add_subplot(gs[0:2, 1])
    ax4.barh(np.arange(C), exp["channel_importance"], color="#c0392b")
    ax4.set_yticks(np.arange(C))
    ax4.set_yticklabels(channels, fontsize=7)
    ax4.invert_yaxis()
    ax4.set_title("Channel importance")
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.barh(classes, exp["probs"], color="#1f3b73")
    ax5.set_xlim(0, 1)
    ax5.tick_params(labelsize=7)
    if path:
        fig.savefig(path, dpi=110, bbox_inches="tight")
        plt.close(fig)
    return fig
