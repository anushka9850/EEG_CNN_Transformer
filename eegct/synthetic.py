"""Realistic synthetic multi-dataset EEG generator (for demo / testing without downloads).

It writes REAL EDF files that imitate the heterogeneity of the five public datasets
(different sampling rates, channel naming conventions, montages, mains frequency and
artifacts), so the complete loading -> harmonization -> preprocessing -> training ->
XAI pipeline is exercised exactly as with the real data.

Disease signatures follow well-documented EEG biomarkers:
* Healthy    : posterior alpha (~10 Hz) with eyes-closed reactivity, low delta/theta
* Alzheimer  : "EEG slowing" - alpha peak shifted to 7-8 Hz, reduced alpha, raised delta/theta
* Parkinson  : milder slowing (alpha ~8.5 Hz) + increased central beta (~20 Hz)
* Depression : frontal alpha asymmetry (right > left) + increased frontal theta / beta
* Epilepsy   : interictal epileptiform discharges (spike-and-wave) and ictal rhythmic
               3-8 Hz evolving seizures (annotated start/end like CHB-MIT)

!!! Results obtained on synthetic data demonstrate that the software works; they are
!!! NOT clinical performance figures. Use the real datasets for your report.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

# approximate 2-D scalp coordinates of the 19 10-20 electrodes
POS = {"FP1": (-0.3, 0.95), "FP2": (0.3, 0.95), "F7": (-0.8, 0.55), "F3": (-0.4, 0.5), "FZ": (0, 0.5),
       "F4": (0.4, 0.5), "F8": (0.8, 0.55), "T7": (-0.95, 0), "C3": (-0.5, 0), "CZ": (0, 0), "C4": (0.5, 0),
       "T8": (0.95, 0), "P7": (-0.8, -0.55), "P3": (-0.4, -0.5), "PZ": (0, -0.5), "P4": (0.4, -0.5),
       "P8": (0.8, -0.55), "O1": (-0.3, -0.95), "O2": (0.3, -0.95)}
ELEC = list(POS)
XY = np.array([POS[e] for e in ELEC])


def _pink(n, rng):
    f = np.fft.rfftfreq(n)
    spec = rng.normal(size=len(f)) + 1j * rng.normal(size=len(f))
    spec[1:] /= f[1:] ** 0.5
    spec[0] = 0
    x = np.fft.irfft(spec, n)
    return x / x.std()


def _osc(n, fs, f0, rng, bw=0.8):
    """Narrow-band oscillation with waxing/waning amplitude."""
    t = np.arange(n) / fs
    inst_f = f0 + bw * np.convolve(rng.normal(size=n), np.ones(int(fs)) / fs, "same")
    phase = 2 * np.pi * np.cumsum(inst_f) / fs + rng.uniform(0, 2 * np.pi)
    env = 1 + 0.6 * np.sin(2 * np.pi * rng.uniform(0.05, 0.2) * t + rng.uniform(0, 6))
    return np.sin(phase) * np.clip(env, 0.2, None)


def _weights(center, width):
    d = np.linalg.norm(XY - np.array(center), axis=1)
    return np.exp(-(d ** 2) / (2 * width ** 2))


def _spike_wave(fs):
    t = np.arange(int(0.35 * fs)) / fs
    spike = -np.exp(-((t - 0.04) ** 2) / (2 * 0.012 ** 2)) * 5
    wave = np.exp(-((t - 0.18) ** 2) / (2 * 0.06 ** 2)) * 2
    return spike + wave


def simulate(label: str, dur_s: float, fs: float, rng, subj: dict, seizures=(), mains=50.0) -> np.ndarray:
    """Return monopolar 19-channel EEG (uV) for the given class."""
    n = int(dur_s * fs)
    mix = np.exp(-np.linalg.norm(XY[:, None] - XY[None], axis=-1) ** 2 / 0.3)
    mix /= mix.sum(1, keepdims=True)
    X = mix @ np.stack([_pink(n, rng) for _ in ELEC]) * 12 * subj["gain"]

    fa, amp_a = subj["alpha_f"], subj["alpha_amp"]
    w_post = _weights((0, -0.8), 0.55)
    w_front = _weights((0, 0.6), 0.5)
    w_cent = _weights((0, 0), 0.45)
    theta_amp, delta_amp, beta_amp = 4.0, 5.0, 2.5
    X += np.outer(_weights((0, 0), 1.2), _osc(n, fs, 2.0, rng, 0.5)) * delta_amp * subj["slow"]
    if label == "Alzheimer":
        fa -= 2.3
        amp_a *= 0.55
        theta_amp *= 2.3
        delta_amp *= 2.0
    elif label == "Parkinson":
        fa -= 1.3
        amp_a *= 0.8
        theta_amp *= 1.6
        beta_amp *= 2.4
    elif label == "Depression":
        theta_amp *= 1.6
        beta_amp *= 1.7
    X += np.outer(w_post, _osc(n, fs, fa, rng)) * amp_a
    X += np.outer(_weights((0, 0.2), 0.8), _osc(n, fs, 6.0 + rng.normal(0, 0.3), rng)) * theta_amp
    X += np.outer(_weights((0, 0), 1.4), _osc(n, fs, 1.5, rng, 0.4)) * delta_amp * (0.5 + 0.5 * (label == "Alzheimer"))
    X += np.outer(w_cent, _osc(n, fs, 20 + rng.normal(0, 1), rng, 2)) * beta_amp
    if label == "Depression":  # frontal alpha asymmetry: right frontal alpha > left
        asym = np.zeros(len(ELEC))
        for e, v in {"F4": 1.0, "F8": 0.8, "FP2": 0.7, "C4": 0.4}.items():
            asym[ELEC.index(e)] = v
        X += np.outer(asym, _osc(n, fs, fa - 0.3, rng)) * amp_a * 0.9
        X += np.outer(w_front, _osc(n, fs, 6.5, rng)) * 3.0
    if label == "Epilepsy":
        side = subj["focus"]
        w_focus = _weights((0.6 * side, 0.1), 0.45)
        sw = _spike_wave(fs)
        for _ in range(int(dur_s * subj["spike_rate"])):  # interictal discharges
            s = rng.integers(0, n - len(sw))
            X[:, s:s + len(sw)] += np.outer(w_focus, sw) * rng.uniform(8, 16)
        t = np.arange(n) / fs
        for a, b in seizures:  # ictal evolving rhythmic discharge
            i0, i1 = int(a * fs), int(b * fs)
            L = i1 - i0
            tt = t[:L]
            f_ict = np.linspace(subj["ictal_f"] + 2.5, subj["ictal_f"], L)
            sig = np.sin(2 * np.pi * np.cumsum(f_ict) / fs)
            sig += 0.6 * np.sign(sig) * np.abs(sig) ** 6  # sharpened (spike-wave-like)
            env = np.clip(tt / (0.25 * L / fs), 0, 1) * np.clip((L / fs - tt) / 3, 0, 1)
            spread = w_focus[:, None] + np.clip(tt / (L / fs), 0, 1)[None] * _weights((0, 0), 0.9)[:, None] * 0.7
            X[:, i0:i1] += spread * sig * env * rng.uniform(35, 60)
    # --- artifacts
    for _ in range(int(dur_s * rng.uniform(0.1, 0.3))):  # eye blinks on frontopolar
        s = rng.integers(0, n - int(0.4 * fs))
        blink = np.hanning(int(0.4 * fs)) * rng.uniform(60, 150)
        X[:, s:s + len(blink)] += np.outer(_weights((0, 1.0), 0.35), blink)
    for _ in range(int(dur_s * 0.02)):  # muscle bursts on temporal channels
        s = rng.integers(0, n - int(fs))
        m = rng.normal(size=int(fs)) * rng.uniform(10, 25)
        X[:, s:s + int(fs)] += np.outer(_weights((rng.choice([-1, 1]) * 0.95, 0), 0.25), m)
    X += np.sin(2 * np.pi * mains * np.arange(n) / fs)[None] * rng.uniform(2, 10)  # mains hum
    X += rng.normal(0, 1.0, X.shape)
    return X


def _subject(rng, label):
    return dict(gain=rng.uniform(0.7, 1.4), alpha_f=rng.normal(10.0, 0.6), alpha_amp=rng.uniform(8, 16),
                slow=rng.uniform(0.6, 1.4), focus=rng.choice([-1, 1]), spike_rate=rng.uniform(0.05, 0.25),
                ictal_f=rng.uniform(3, 6))


def _write_edf(path: Path, data: np.ndarray, fs: float, names: list[str]):
    import pyedflib
    path.parent.mkdir(parents=True, exist_ok=True)
    hdrs = []
    for nm, ch in zip(names, data):
        lim = float(np.ceil(np.abs(ch).max() + 10))
        hdrs.append(dict(label=nm, dimension="uV", sample_frequency=fs, physical_min=-lim, physical_max=lim,
                         digital_min=-32768, digital_max=32767, transducer="", prefilter=""))
    w = pyedflib.EdfWriter(str(path), len(names), file_type=pyedflib.FILETYPE_EDFPLUS)
    w.setSignalHeaders(hdrs)
    w.writeSamples([np.ascontiguousarray(c, dtype=np.float64) for c in data])
    w.close()


# ---------------------------------------------------------------- dataset "styles"
_MIXED = {"FP1": "Fp1", "FP2": "Fp2", "FZ": "Fz", "CZ": "Cz", "PZ": "Pz"}

def _style_tuh(X):  # referential, old T3/T4 names + ear electrodes, "EEG XX-REF"
    old = {"T7": "T3", "T8": "T4", "P7": "T5", "P8": "T6"}
    names = [f"EEG {old.get(e, e)}-REF" for e in ELEC] + ["EEG A1-REF", "EEG A2-REF"]
    return np.vstack([X, X[[ELEC.index("T7")]] * 0.3, X[[ELEC.index("T8")]] * 0.3]), names


def _style_chbmit(X):  # bipolar double banana, CHB-MIT naming (+ duplicated T8-P8 like the real files)
    from .harmonize import BIPOLAR_BANANA
    pairs = BIPOLAR_BANANA + [("T8", "P8")]
    data = np.stack([X[ELEC.index(a)] - X[ELEC.index(b)] for a, b in pairs])
    return data, [f"{a}-{b}" for a, b in pairs]


def _style_ad(X):  # ds004504: 19 ch, old names, mixed case
    old = {"T7": "T3", "T8": "T4", "P7": "T5", "P8": "T6"}
    return X, [_MIXED.get(e, old.get(e, e)) for e in ELEC]


def _style_pd(X):  # ds002778: BioSemi-32 names incl. extra electrodes
    extra = {"AF3": ("FP1", "F3"), "AF4": ("FP2", "F4"), "FC1": ("F3", "C3"), "FC2": ("F4", "C4"),
             "FC5": ("F7", "C3"), "FC6": ("F8", "C4"), "CP1": ("C3", "P3"), "CP2": ("C4", "P4"),
             "CP5": ("T7", "P3"), "CP6": ("T8", "P4"), "PO3": ("P3", "O1"), "PO4": ("P4", "O2"), "OZ": ("O1", "O2")}
    data = [X[i] for i in range(len(ELEC))] + [(X[ELEC.index(a)] + X[ELEC.index(b)]) / 2 for a, b in extra.values()]
    names = [_MIXED.get(e, e) for e in ELEC] + [k.replace("OZ", "Oz") for k in extra]
    return np.stack(data), names


def _style_modma(X):  # EGI HydroCel-128 electrode numbers (subset) + Cz reference channel
    from .harmonize import HYDROCEL128_TO_1020
    inv = {v: k for k, v in HYDROCEL128_TO_1020.items() if k != "CZ"}
    names = [inv[e] for e in ELEC]
    return X, names


STYLES = {  # name: (style fn, fs, mains, classes {label: n_subjects})
    "tuh": (_style_tuh, 250.0, 60.0, {"Healthy": 14}),
    "chbmit": (_style_chbmit, 256.0, 60.0, {"Epilepsy": 12}),
    "openneuro_ad": (_style_ad, 500.0, 50.0, {"Alzheimer": 12, "Healthy": 7}),
    "openneuro_pd": (_style_pd, 512.0, 60.0, {"Parkinson": 12, "Healthy": 7}),
    "modma": (_style_modma, 250.0, 50.0, {"Depression": 12, "Healthy": 7}),
}


def generate(out_dir: str | Path, seed: int = 0, dur_s: float = 180.0, scale: float = 1.0) -> Path:
    """Write synthetic EDFs for all five datasets + manifest.csv. Returns manifest path."""
    out = Path(out_dir)
    rng = np.random.default_rng(seed)
    rows = []
    for ds, (style, fs, mains, groups) in STYLES.items():
        for label, n_sub in groups.items():
            for k in range(max(2, int(round(n_sub * scale)))):
                sid = f"{label[:3].lower()}{k:02d}"
                subj = _subject(rng, label)
                n_files = 2 if ds == "chbmit" else 1
                for fi in range(n_files):
                    if ds == "chbmit":
                        d = dur_s * 2
                        L = rng.uniform(min(40, 0.2 * d), min(90, 0.35 * d))
                        s0 = rng.uniform(0.15 * d, d - L - 0.1 * d)
                        seizures = [(round(s0), round(s0 + L))]
                    else:
                        d, seizures = dur_s, []
                    X = simulate(label, d, fs, rng, subj, seizures, mains)
                    data, names = style(X)
                    f = out / ds / f"{ds}_{sid}_{fi:02d}.edf"
                    _write_edf(f, data, fs, names)
                    rows.append(dict(path=str(f.relative_to(out)), label=label, subject=sid, dataset=ds,
                                     seizures=";".join(f"{a}-{b}" for a, b in seizures)))
    mf = out / "manifest.csv"
    with open(mf, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["path", "label", "subject", "dataset", "seizures"])
        w.writeheader()
        w.writerows(rows)
    return mf
