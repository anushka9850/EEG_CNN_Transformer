"""Dataset harmonization: common channel set / montage, common sampling rate, label mapping.

Every public EEG dataset names its electrodes differently and uses a different
reference and sampling rate. This module maps any recording onto ONE common
representation:

* ``bipolar_banana`` (default) - the 18-channel longitudinal bipolar
  "double banana" montage. CHB-MIT is *recorded* in this montage, and any
  monopolar (referential) recording can be converted to it exactly
  (bipolar = electrode A - electrode B), so this is the only montage in which
  all five datasets can be represented without approximation.
* ``monopolar_1020`` - 19 standard 10-20 electrodes (for datasets that are all
  referential).
"""
from __future__ import annotations

import re
import numpy as np
from scipy.signal import resample_poly
from fractions import Fraction

# 19 standard 10-20 electrodes (modern names: T7/T8/P7/P8 == old T3/T4/T5/T6)
MONOPOLAR_1020 = ["FP1", "FP2", "F7", "F3", "FZ", "F4", "F8", "T7", "C3", "CZ",
                  "C4", "T8", "P7", "P3", "PZ", "P4", "P8", "O1", "O2"]

# Longitudinal bipolar "double banana" (exactly the 18 core CHB-MIT channels)
BIPOLAR_BANANA = [
    ("FP1", "F7"), ("F7", "T7"), ("T7", "P7"), ("P7", "O1"),
    ("FP1", "F3"), ("F3", "C3"), ("C3", "P3"), ("P3", "O1"),
    ("FP2", "F4"), ("F4", "C4"), ("C4", "P4"), ("P4", "O2"),
    ("FP2", "F8"), ("F8", "T8"), ("T8", "P8"), ("P8", "O2"),
    ("FZ", "CZ"), ("CZ", "PZ"),
]
BIPOLAR_NAMES = [f"{a}-{b}" for a, b in BIPOLAR_BANANA]

# Old/alternative names -> canonical
ALIASES = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}

# EGI GSN HydroCel-128 (MODMA) -> nearest 10-20 electrode
HYDROCEL128_TO_1020 = {
    "E22": "FP1", "E9": "FP2", "E33": "F7", "E24": "F3", "E11": "FZ", "E124": "F4",
    "E122": "F8", "E45": "T7", "E36": "C3", "E129": "CZ", "CZ": "CZ", "E104": "C4",
    "E108": "T8", "E58": "P7", "E52": "P3", "E62": "PZ", "E92": "P4", "E96": "P8",
    "E70": "O1", "E83": "O2",
}


def canonical_name(name: str) -> str:
    """Normalise a raw channel label, e.g. 'EEG FP1-REF' -> 'FP1', 'Fp1.' -> 'FP1',
    'T3-LE' -> 'T7', 'E22' -> 'FP1'. Bipolar labels ('FP1-F7') are returned as 'FP1-F7'."""
    n = name.strip().upper()
    n = re.sub(r"^(EEG|EOG|ECG|EMG)\s+", "", n)
    n = n.replace(".", "").replace(" ", "")
    n = re.sub(r"-(REF|LE|AR|AVG|A1|A2|M1|M2)$", "", n)
    if n in HYDROCEL128_TO_1020:
        return HYDROCEL128_TO_1020[n]
    if "-" in n:  # bipolar label
        parts = n.split("-")
        if len(parts) >= 2:
            a = ALIASES.get(parts[0], parts[0])
            b = ALIASES.get(parts[1], parts[1])
            return f"{a}-{b}"
    return ALIASES.get(n, n)


def target_channels(montage: str) -> list[str]:
    if montage == "bipolar_banana":
        return list(BIPOLAR_NAMES)
    if montage == "monopolar_1020":
        return list(MONOPOLAR_1020)
    raise ValueError(f"unknown montage {montage}")


def to_common_montage(data: np.ndarray, ch_names: list[str], montage: str) -> np.ndarray:
    """Map (n_channels, n_times) data with arbitrary channel names onto the common montage.

    Works for monopolar input (differences are computed) and bipolar input (channels
    are picked by name; reversed pairs are negated). Missing channels raise ValueError
    unless at most 2 are missing, in which case they are zero-filled (and later ignored
    by per-channel normalization)."""
    canon = [canonical_name(c) for c in ch_names]
    index: dict[str, int] = {}
    for i, c in enumerate(canon):
        index.setdefault(c, i)  # keep first occurrence (CHB-MIT repeats T8-P8)

    n_t = data.shape[1]
    if montage == "monopolar_1020":
        out = np.zeros((len(MONOPOLAR_1020), n_t), dtype=np.float32)
        missing = []
        for k, e in enumerate(MONOPOLAR_1020):
            if e in index:
                out[k] = data[index[e]]
            else:
                missing.append(e)
        # common average reference over the 19 electrodes
        present = [k for k, e in enumerate(MONOPOLAR_1020) if e not in missing]
        if len(missing) > 2:
            raise ValueError(f"too many missing electrodes: {missing}")
        out[present] -= out[present].mean(axis=0, keepdims=True)
        return out

    out = np.zeros((len(BIPOLAR_BANANA), n_t), dtype=np.float32)
    missing = []
    for k, (a, b) in enumerate(BIPOLAR_BANANA):
        if f"{a}-{b}" in index:
            out[k] = data[index[f"{a}-{b}"]]
        elif f"{b}-{a}" in index:
            out[k] = -data[index[f"{b}-{a}"]]
        elif a in index and b in index:
            out[k] = data[index[a]] - data[index[b]]
        else:
            missing.append(f"{a}-{b}")
    if len(missing) > 2:
        raise ValueError(f"cannot build bipolar montage, missing: {missing}")
    return out


def resample(data: np.ndarray, fs_in: float, fs_out: float) -> np.ndarray:
    """Polyphase resampling to the common rate (anti-aliased)."""
    if abs(fs_in - fs_out) < 1e-6:
        return data.astype(np.float32)
    frac = Fraction(fs_out / fs_in).limit_denominator(1000)
    return resample_poly(data, frac.numerator, frac.denominator, axis=-1).astype(np.float32)


# ------------------------------------------------------------------ labels
LABEL_ALIASES = {
    "healthy": "Healthy", "hc": "Healthy", "control": "Healthy", "normal": "Healthy", "c": "Healthy",
    "epilepsy": "Epilepsy", "seizure": "Epilepsy", "ictal": "Epilepsy", "epileptic": "Epilepsy",
    "alzheimer": "Alzheimer", "alzheimers": "Alzheimer", "ad": "Alzheimer", "a": "Alzheimer",
    "parkinson": "Parkinson", "parkinsons": "Parkinson", "pd": "Parkinson",
    "depression": "Depression", "mdd": "Depression", "depressed": "Depression",
}


def map_label(label: str) -> str:
    key = re.sub(r"[^a-z]", "", str(label).lower())
    return LABEL_ALIASES.get(key, str(label))
