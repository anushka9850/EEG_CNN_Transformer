"""EEG preprocessing: reading, filtering, ICA artifact removal, windowing, normalization."""
from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.signal import butter, iirnotch, sosfiltfilt, filtfilt

from .harmonize import to_common_montage, resample, canonical_name, MONOPOLAR_1020

log = logging.getLogger(__name__)


@dataclass
class Recording:
    """One continuous EEG recording (already in microvolts)."""
    data: np.ndarray                 # (n_channels, n_times) uV
    fs: float
    ch_names: list[str]
    label: str
    subject: str
    dataset: str
    path: str = ""
    seizures: list[tuple[float, float]] = field(default_factory=list)  # (start_s, end_s)
    is_epilepsy_patient: bool = False


# --------------------------------------------------------------------- reading
def read_any(path: str | Path, fs: float | None = None, ch_names: list[str] | None = None):
    """Read an EEG file of any common format -> (data_uV (C,T), fs, ch_names)."""
    path = Path(path)
    suf = path.suffix.lower()
    if suf in {".edf", ".bdf", ".set", ".fif", ".vhdr", ".gdf", ".cnt"} or path.name.endswith(".fif.gz"):
        import mne
        mne.set_log_level("ERROR")
        raw = mne.io.read_raw(str(path), preload=True, verbose="ERROR")
        picks = mne.pick_types(raw.info, eeg=True, eog=False, ecg=False, emg=False, misc=False, stim=False)
        if len(picks) == 0:  # some EDFs mark everything as misc
            picks = list(range(len(raw.ch_names)))
        data = raw.get_data(picks=picks) * 1e6
        names = [raw.ch_names[p] for p in picks]
        return data.astype(np.float32), float(raw.info["sfreq"]), names
    if suf == ".mat":
        return _read_mat(path, fs, ch_names)
    if suf == ".npy":
        data = np.load(path)
        if fs is None or ch_names is None:
            raise ValueError(".npy input needs fs and ch_names")
        return data.astype(np.float32), float(fs), list(ch_names)
    if suf == ".csv":
        import pandas as pd
        df = pd.read_csv(path)
        if fs is None:
            raise ValueError(".csv input needs fs")
        return df.values.T.astype(np.float32), float(fs), list(df.columns)
    raise ValueError(f"unsupported file type: {path}")


def _read_mat(path: Path, fs, ch_names):
    """MATLAB file: take the largest 2-D numeric array as (channels x time)."""
    try:
        from scipy.io import loadmat
        m = loadmat(str(path))
        arrays = {k: v for k, v in m.items() if not k.startswith("__") and isinstance(v, np.ndarray)}
    except NotImplementedError:  # v7.3 -> HDF5
        import h5py
        arrays = {}
        with h5py.File(path, "r") as f:
            for k in f.keys():
                arrays[k] = np.array(f[k])
    arrays = {k: v for k, v in arrays.items() if v.ndim == 2 and np.issubdtype(v.dtype, np.number)}
    if not arrays:
        raise ValueError(f"no 2-D array in {path}")
    data = max(arrays.values(), key=lambda a: a.size).astype(np.float32)
    if data.shape[0] > data.shape[1]:
        data = data.T
    if ch_names is None:
        ch_names = [f"E{i + 1}" for i in range(data.shape[0])]  # EGI numbering (MODMA)
    if fs is None:
        raise ValueError(".mat input needs fs")
    return data, float(fs), list(ch_names)[: data.shape[0]]


def read_deap(path: str | Path):
    """DEAP preprocessed python .dat -> list of (data_uV (32,T), fs=128, names) per trial."""
    names = ["Fp1", "AF3", "F3", "F7", "FC5", "FC1", "C3", "T7", "CP5", "CP1", "P3", "P7", "PO3",
             "O1", "Oz", "Pz", "Fp2", "AF4", "Fz", "F4", "F8", "FC6", "FC2", "Cz", "C4", "T8",
             "CP6", "CP2", "P4", "P8", "PO4", "O2"]
    with open(path, "rb") as f:
        d = pickle.load(f, encoding="latin1")
    out = []
    for trial in d["data"]:
        out.append((trial[:32, 3 * 128:].astype(np.float32), 128.0, names))  # drop 3 s baseline
    return out


# --------------------------------------------------------------------- filtering
def bandpass_notch(data: np.ndarray, fs: float, band=(0.5, 45.0), notch=(50.0, 60.0)) -> np.ndarray:
    nyq = fs / 2.0
    lo, hi = band
    hi = min(hi, nyq * 0.95)
    sos = butter(4, [lo / nyq, hi / nyq], btype="band", output="sos")
    out = sosfiltfilt(sos, data, axis=-1)
    for f0 in notch or []:
        if f0 < nyq * 0.98:
            b, a = iirnotch(f0 / nyq, Q=30.0)
            out = filtfilt(b, a, out, axis=-1)
    return out.astype(np.float32)


def ica_clean(data: np.ndarray, fs: float, ch_names: list[str], n_components: int = 15) -> np.ndarray:
    """Remove ocular artifacts with ICA (MNE). Components correlated with the
    frontopolar channels (blink proxies FP1/FP2) are zeroed. Monopolar input only."""
    import mne
    canon = [canonical_name(c) for c in ch_names]
    if any("-" in c for c in canon):
        return data  # bipolar recording (e.g. CHB-MIT) - ICA not meaningful here
    eog = [ch_names[i] for i, c in enumerate(canon) if c in ("FP1", "FP2")]
    if not eog:
        return data
    info = mne.create_info(ch_names, fs, ch_types="eeg")
    raw = mne.io.RawArray(data * 1e-6, info, verbose="ERROR")
    raw_f = raw.copy().filter(1.0, None, verbose="ERROR")  # ICA fits better on >1 Hz data
    n = min(n_components, len(ch_names) - 1)
    ica = mne.preprocessing.ICA(n_components=n, random_state=97, max_iter="auto", verbose="ERROR")
    ica.fit(raw_f, verbose="ERROR")
    bads = []
    for ch in eog:
        idx, _ = ica.find_bads_eog(raw_f, ch_name=ch, threshold=3.0, verbose="ERROR")
        bads += idx
    ica.exclude = sorted(set(bads))[:3]
    ica.apply(raw, verbose="ERROR")
    return (raw.get_data() * 1e6).astype(np.float32)


def normalize_windows(x: np.ndarray, method: str = "robust_zscore") -> np.ndarray:
    """Per-window, per-channel normalization. x: (N, C, T)."""
    if method == "zscore":
        mu = x.mean(-1, keepdims=True)
        sd = x.std(-1, keepdims=True)
    else:
        mu = np.median(x, -1, keepdims=True)
        q75, q25 = np.percentile(x, [75, 25], axis=-1, keepdims=True)
        sd = (q75 - q25) / 1.349
    sd = np.where(sd < 1e-6, 1.0, sd)
    return np.clip((x - mu) / sd, -20, 20).astype(np.float32)


# --------------------------------------------------------------------- pipeline
def preprocess_recording(rec: Recording, cfg: dict) -> np.ndarray:
    """Recording -> harmonized, filtered, resampled continuous signal (C_common, T) in uV."""
    pp, hz = cfg["preprocessing"], cfg["harmonization"]
    data = np.nan_to_num(rec.data)
    if pp.get("ica"):
        try:
            data = ica_clean(data, rec.fs, rec.ch_names, pp.get("ica_components", 15))
        except Exception as e:  # never fail the whole pipeline on ICA
            log.warning("ICA skipped for %s: %s", rec.path, e)
    data = to_common_montage(data, rec.ch_names, hz["montage"])
    data = bandpass_notch(data, rec.fs, tuple(pp["bandpass"]), tuple(pp.get("notch") or ()))
    data = resample(data, rec.fs, hz["target_fs"])
    clip = pp.get("clip_uv")
    if clip:
        data = np.clip(data, -clip, clip)
    return data


def segment(signal: np.ndarray, fs: float, length_s: float, overlap: float):
    """Continuous (C,T) -> windows (N,C,W) and their start times (s)."""
    w = int(round(length_s * fs))
    step = max(1, int(round(w * (1 - overlap))))
    if signal.shape[1] < w:
        return np.zeros((0, signal.shape[0], w), np.float32), np.zeros(0)
    starts = np.arange(0, signal.shape[1] - w + 1, step)
    idx = starts[:, None] + np.arange(w)[None, :]
    wins = signal[:, idx].transpose(1, 0, 2)
    return np.ascontiguousarray(wins, dtype=np.float32), starts / fs


def _in_seizure(t0: float, t1: float, seizures) -> float:
    """Fraction of [t0,t1] overlapping any seizure."""
    ov = 0.0
    for s, e in seizures:
        ov += max(0.0, min(t1, e) - max(t0, s))
    return ov / (t1 - t0)


def recording_to_windows(rec: Recording, cfg: dict, rng: np.random.Generator):
    """Full pipeline for one recording -> (X (N,C,W), labels list[str])."""
    sig = preprocess_recording(rec, cfg)
    fs = cfg["harmonization"]["target_fs"]
    wcfg = cfg["windows"]
    L = wcfg["length_s"]
    cap = wcfg.get("max_windows_per_recording") or 10 ** 9
    task = cfg["task"]

    if rec.seizures or (rec.is_epilepsy_patient and task == "seizure"):
        # dense windows so the (rare) ictal class gets enough samples
        X, t = segment(sig, fs, L, max(wcfg["overlap"], 0.75))
        frac = np.array([_in_seizure(s, s + L, rec.seizures) for s in t])
        ictal = np.where(frac >= 0.5)[0]
        inter = np.where(frac == 0.0)[0]
        ratio = cfg["datasets"].get("chbmit", {}).get("interictal_per_ictal", 1.0)
        n_inter = int(max(len(ictal) * ratio, 8 if len(ictal) == 0 else 0))
        inter = rng.choice(inter, size=min(n_inter, len(inter), cap), replace=False) if len(inter) else inter
        ictal = ictal[:cap]
        if task == "seizure":
            labels = ["Seizure"] * len(ictal) + ["Non-seizure"] * len(inter)
        else:
            labels = ["Epilepsy"] * (len(ictal) + len(inter))
        sel = np.concatenate([ictal, inter]).astype(int)
        X = X[sel]
    else:
        if task == "seizure":
            return np.zeros((0, sig.shape[0], int(round(L * fs))), np.float32), []
        X, _ = segment(sig, fs, L, wcfg["overlap"])
        if rec.is_epilepsy_patient:  # seizure-free file of an epilepsy patient: keep a sample
            cap = min(cap, 20)
        if len(X) > cap:
            X = X[np.sort(rng.choice(len(X), cap, replace=False))]
        labels = [rec.label] * len(X)
    X = normalize_windows(X, cfg["preprocessing"].get("normalization", "robust_zscore"))
    return X, labels
