"""Unit tests:  python -m pytest -q"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eegct.cli import load_config
from eegct.data import subject_split
from eegct.datasets import parse_chbmit_summary
from eegct.explain import explain_window
from eegct.harmonize import BIPOLAR_NAMES, canonical_name, map_label, resample, to_common_montage
from eegct.model import CNNTransformer
from eegct.preprocess import Recording, bandpass_notch, normalize_windows, recording_to_windows, segment


def test_dashboard_loads():
    from streamlit.testing.v1 import AppTest

    app_path = Path(__file__).resolve().parent.parent / "app" / "streamlit_app.py"
    app = AppTest.from_file(str(app_path)).run(timeout=30)
    assert not app.exception
    assert any(header.value == "Analyze a recording" for header in app.header)
    # Dataset selectbox only appears when synthetic sample files exist


def test_canonical_names():
    assert canonical_name("EEG FP1-REF") == "FP1"
    assert canonical_name("Fp1.") == "FP1"
    assert canonical_name("T3") == "T7"
    assert canonical_name("EEG T5-LE") == "P7"
    assert canonical_name("E22") == "FP1"
    assert canonical_name("FP1-F7") == "FP1-F7"
    assert canonical_name("T8-P8-0") == "T8-P8"


def test_monopolar_to_bipolar_equals_difference():
    rng = np.random.default_rng(0)
    names = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T3", "C3", "Cz", "C4", "T4", "T5", "P3", "Pz",
             "P4", "T6", "O1", "O2"]
    x = rng.normal(size=(19, 100)).astype(np.float32)
    b = to_common_montage(x, names, "bipolar_banana")
    assert b.shape == (18, 100)
    assert np.allclose(b[0], x[0] - x[2])  # FP1-F7
    # bipolar input is passed through (reversed pair negated)
    b2 = to_common_montage(b[::-1].copy(), BIPOLAR_NAMES[::-1], "bipolar_banana")
    assert np.allclose(b2, b)


def test_resample_and_filter():
    x = np.random.randn(3, 5120).astype(np.float32)
    y = resample(x, 256, 128)
    assert y.shape == (3, 2560)
    t = np.arange(2560) / 128
    sig = np.sin(2 * np.pi * 10 * t) + np.sin(2 * np.pi * 50 * t)
    f = bandpass_notch(sig[None], 128)
    spec = np.abs(np.fft.rfft(f[0]))
    fr = np.fft.rfftfreq(2560, 1 / 128)
    assert spec[np.argmin(abs(fr - 50))] < 0.05 * spec[np.argmin(abs(fr - 10))]


def test_segment_and_normalize():
    X, t = segment(np.random.randn(18, 1280).astype(np.float32), 128, 4, 0.5)
    assert X.shape == (4, 18, 512) and np.allclose(t, [0, 2, 4, 6])
    Z = normalize_windows(X)
    assert np.abs(np.median(Z, -1)).max() < 1e-4


def test_chbmit_summary(tmp_path):
    s = tmp_path / "chb01-summary.txt"
    s.write_text("File Name: chb01_03.edf\nNumber of Seizures in File: 1\nSeizure Start Time: 2996 seconds\n"
                 "Seizure End Time: 3036 seconds\n\nFile Name: chb01_04.edf\nNumber of Seizures in File: 0\n"
                 "File Name: chb04_28.edf\nSeizure 1 Start Time: 1679 seconds\nSeizure 1 End Time: 1781 seconds\n"
                 "Seizure 2 Start Time: 3782 seconds\nSeizure 2 End Time: 3898 seconds\n")
    d = parse_chbmit_summary(s)
    assert d["chb01_03.edf"] == [(2996, 3036)] and d["chb01_04.edf"] == []
    assert d["chb04_28.edf"] == [(1679, 1781), (3782, 3898)]


def test_labels():
    assert map_label("MDD") == "Depression" and map_label("HC") == "Healthy" and map_label("A") == "Alzheimer"


def test_recording_to_windows_epilepsy():
    cfg = load_config(None)
    fs = 256
    rec = Recording(np.random.randn(18, fs * 120).astype(np.float32) * 20, fs, BIPOLAR_NAMES, "Epilepsy",
                    "s", "chbmit", seizures=[(40, 70)], is_epilepsy_patient=True)
    X, lab = recording_to_windows(rec, cfg, np.random.default_rng(0))
    assert X.shape[1:] == (18, 512) and len(lab) == len(X) and set(lab) == {"Epilepsy"}
    cfg["task"] = "seizure"
    X, lab = recording_to_windows(rec, cfg, np.random.default_rng(0))
    assert lab.count("Seizure") >= 20 and "Non-seizure" in lab


def test_subject_split_no_leak():
    subj = np.repeat([f"s{i}" for i in range(30)], 10)
    y = np.repeat(np.arange(30) % 3, 10)
    tr, va, te = subject_split(y, subj, [0.7, 0.15, 0.15], 0)
    assert not (set(subj[tr]) & set(subj[te])) and not (set(subj[tr]) & set(subj[va]))
    assert not (set(subj[va]) & set(subj[te]))
    assert len(tr) + len(va) + len(te) == len(y)
    assert set(y[te]) == {0, 1, 2}


def test_model_and_gradcam():
    m = CNNTransformer(18, 512, 5)
    out = m(torch.randn(4, 18, 512))
    assert out.shape == (4, 5)
    e = explain_window(m, np.random.randn(18, 512).astype(np.float32))
    assert e["cam"].shape == (18, 512) and e["time_attention"].shape == (512,)
    assert abs(e["probs"].sum() - 1) < 1e-4
    assert len(m.temporal2._forward_hooks) == 0  # hooks cleaned up
