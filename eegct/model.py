"""CNN-Transformer model for multi-disease EEG classification.

Input  : (B, C, T)  harmonized EEG window (C=18 bipolar channels, T=512 = 4 s @ 128 Hz)

  1. CNN feature extraction
     - temporal conv block 1  (1 x 33)  : learns frequency-selective filters per channel (+ pool 2)
     - temporal conv block 2  (1 x 15)  : <- Grad-CAM target (keeps channel x time layout)
     - spatial depthwise conv (C x 1)   : learns spatial filters across electrodes
     - separable temporal conv + pooling -> sequence of tokens (d_model, T/16)
  2. Transformer encoder
     - [CLS] token + learnable positional embedding
     - N pre-norm encoder layers with multi-head self-attention (attention maps are stored)
  3. Classification head on [CLS] + mean-pooled tokens
"""
from __future__ import annotations

import torch
import torch.nn as nn


class EncoderLayer(nn.Module):
    """Pre-norm transformer encoder layer that keeps its attention weights for XAI."""

    def __init__(self, d, heads, ff, dropout):
        super().__init__()
        self.n1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.n2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(ff, d))
        self.drop = nn.Dropout(dropout)
        self.last_attn = None

    def forward(self, x):
        h = self.n1(x)
        a, w = self.attn(h, h, h, need_weights=True, average_attn_weights=False)
        self.last_attn = w.detach()  # (B, heads, L, L)
        x = x + self.drop(a)
        return x + self.drop(self.ff(self.n2(x)))


class CNNTransformer(nn.Module):
    def __init__(self, n_channels: int, n_times: int, n_classes: int, f1=8, f2=16, depth_mult=4,
                 d_model=64, n_heads=4, n_layers=3, ff_dim=128, dropout=0.25):
        super().__init__()
        self.hparams = dict(n_channels=n_channels, n_times=n_times, n_classes=n_classes, f1=f1, f2=f2,
                            depth_mult=depth_mult, d_model=d_model, n_heads=n_heads, n_layers=n_layers,
                            ff_dim=ff_dim, dropout=dropout)
        # ---- CNN feature extractor
        self.temporal1 = nn.Sequential(nn.Conv2d(1, f1, (1, 33), padding=(0, 16), bias=False),
                                       nn.BatchNorm2d(f1), nn.ELU(), nn.AvgPool2d((1, 2)))
        # depthwise temporal conv (f1 -> f2): cheap on CPU, keeps channel x time layout
        self.temporal2 = nn.Sequential(nn.Conv2d(f1, f2, (1, 15), padding=(0, 7), groups=f1, bias=False),
                                       nn.BatchNorm2d(f2), nn.ELU())
        fs_ = f2 * depth_mult
        self.spatial = nn.Sequential(nn.Conv2d(f2, fs_, (n_channels, 1), groups=f2, bias=False),
                                     nn.BatchNorm2d(fs_), nn.ELU(), nn.AvgPool2d((1, 4)), nn.Dropout(dropout))
        self.separable = nn.Sequential(
            nn.Conv2d(fs_, fs_, (1, 15), padding=(0, 7), groups=fs_, bias=False),
            nn.Conv2d(fs_, d_model, 1, bias=False),
            nn.BatchNorm2d(d_model), nn.ELU(), nn.AvgPool2d((1, 2)), nn.Dropout(dropout))
        self.n_tokens = n_times // 16
        # ---- Transformer encoder
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos = nn.Parameter(torch.randn(1, self.n_tokens + 1, d_model) * 0.02)
        self.layers = nn.ModuleList([EncoderLayer(d_model, n_heads, ff_dim, dropout) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        # ---- head
        self.head = nn.Sequential(nn.Linear(2 * d_model, d_model), nn.GELU(), nn.Dropout(dropout),
                                  nn.Linear(d_model, n_classes))

    def features(self, x):
        """(B,C,T) -> CNN token sequence (B, L, d)."""
        x = x.unsqueeze(1)                    # (B,1,C,T)
        x = self.temporal1(x)
        x = self.temporal2(x)                 # (B,f2,C,T/2)   <- Grad-CAM layer
        x = self.spatial(x)                   # (B,f2*D,1,T/8)
        x = self.separable(x)                 # (B,d,1,T/16)
        return x.squeeze(2).transpose(1, 2)   # (B,L,d)

    def forward(self, x):
        tok = self.features(x)
        tok = tok[:, : self.n_tokens]
        z = torch.cat([self.cls.expand(len(tok), -1, -1), tok], 1) + self.pos[:, : tok.shape[1] + 1]
        for layer in self.layers:
            z = layer(z)
        z = self.norm(z)
        return self.head(torch.cat([z[:, 0], z[:, 1:].mean(1)], -1))

    def attention_maps(self):
        return [l.last_attn for l in self.layers]


def build_model(cfg: dict, n_channels: int, n_times: int, n_classes: int) -> CNNTransformer:
    return CNNTransformer(n_channels, n_times, n_classes, **cfg["model"])


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
