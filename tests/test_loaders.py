"""Loader tests: build miniature copies of every real dataset's folder layout / file format
and check that each loader finds the files, labels them correctly and that they harmonize."""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import savemat

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eegct.cli import load_config
from eegct.data import build_cache, load_cache
from eegct.datasets import (load_bids, load_chbmit, load_deap, load_eegmmidb, load_modma, load_tuh)
from eegct.harmonize import to_common_montage
from eegct.synthetic import (ELEC, _style_ad, _style_chbmit, _style_modma, _style_pd, _style_tuh,
                             _subject, _write_edf, simulate)

RNG = np.random.default_rng(1)


def _sim(label, dur=30, fs=256, seizures=()):
    return simulate(label, dur, fs, RNG, _subject(RNG, label), seizures)


def _check(recs, labels):
    assert recs, "loader returned nothing"
    assert {r.label for r in recs} == set(labels)
    for r in recs:
        b = to_common_montage(r.data, r.ch_names, "bipolar_banana")
        assert b.shape[0] == 18 and np.abs(b).sum(1).min() > 0  # every derivation present


def test_chbmit(tmp_path):
    root = tmp_path / "chbmit" / "chb01"
    d, n = _style_chbmit(_sim("Epilepsy", 60, 256, [(20, 40)]))
    _write_edf(root / "chb01_03.edf", d, 256, n)
    d, n = _style_chbmit(_sim("Epilepsy", 60, 256))
    _write_edf(root / "chb01_04.edf", d, 256, n)
    (root / "chb01-summary.txt").write_text(
        "File Name: chb01_03.edf\nSeizure Start Time: 20 seconds\nSeizure End Time: 40 seconds\n"
        "File Name: chb01_04.edf\nNumber of Seizures in File: 0\n")
    recs = list(load_chbmit({"root": tmp_path / "chbmit"}))
    _check(recs, ["Epilepsy"])
    assert {Path(r.path).name: r.seizures for r in recs}["chb01_03.edf"] == [(20, 40)]


def test_tuh(tmp_path):
    d, n = _style_tuh(_sim("Healthy", 30, 250))
    _write_edf(tmp_path / "normal" / "01_tcp_ar" / "aaaaaaav_s001_t000.edf", d, 250, n)
    _write_edf(tmp_path / "abnormal" / "01_tcp_ar" / "aaaaabbb_s001_t000.edf", d, 250, n)
    recs = list(load_tuh({"root": tmp_path}))
    assert len(recs) == 1 and recs[0].subject == "tuh_aaaaaaav"
    _check(recs, ["Healthy"])


def test_openneuro_ad(tmp_path):
    pd.DataFrame({"participant_id": ["sub-001", "sub-002", "sub-003"], "Group": ["A", "C", "F"]}) \
        .to_csv(tmp_path / "participants.tsv", sep="\t", index=False)
    for s, lab in [("sub-001", "Alzheimer"), ("sub-002", "Healthy"), ("sub-003", "Healthy")]:
        d, n = _style_ad(_sim(lab, 30, 500))
        _write_edf(tmp_path / s / "eeg" / f"{s}_task-eyesclosed_eeg.edf", d, 500, n)
    recs = list(load_bids({"root": tmp_path}, "openneuro_ad"))
    assert len(recs) == 2  # FTD skipped
    _check(recs, ["Alzheimer", "Healthy"])


def test_openneuro_pd(tmp_path):
    for s, ses, lab in [("sub-hc1", "hc", "Healthy"), ("sub-pd3", "off", "Parkinson"), ("sub-pd3", "on", "Parkinson")]:
        d, n = _style_pd(_sim(lab, 30, 512))
        _write_edf(tmp_path / s / f"ses-{ses}" / "eeg" / f"{s}_ses-{ses}_task-rest_eeg.edf", d, 512, n)
    recs = list(load_bids({"root": tmp_path}, "openneuro_pd"))
    assert len(recs) == 3
    _check(recs, ["Healthy", "Parkinson"])


def test_modma(tmp_path):
    for sid, lab in [("02010002", "Depression"), ("02030004", "Healthy")]:
        x, names = _style_modma(_sim(lab, 30, 250))
        full = np.zeros((129, x.shape[1]), np.float32)  # E1..E129 (E129 = Cz)
        for row, nm in zip(x, names):
            full[int(nm[1:]) - 1] = row
        savemat(tmp_path / f"{sid}rest 20150416 1017..mat", {"mat_data": full})
    pd.DataFrame({"subject id": [2010002, 2030004], "type": ["MDD", "HC"]}).to_excel(
        tmp_path / "subjects_information.xlsx", index=False)
    recs = list(load_modma({"root": tmp_path, "fs": 250}))
    _check(recs, ["Depression", "Healthy"])


def test_eegmmidb(tmp_path):
    names = [f"{e.capitalize() if len(e) == 2 and e[1].isdigit() else e.title()}." for e in ELEC]
    for r in (1, 2, 3):
        _write_edf(tmp_path / "S001" / f"S001R{r:02d}.edf", _sim("Healthy", 30, 160), 160, names)
    recs = list(load_eegmmidb({"root": tmp_path, "runs": [1, 2]}))
    assert len(recs) == 2
    _check(recs, ["Healthy"])


def test_deap(tmp_path):
    from eegct.preprocess import read_deap
    order = ["FP1", "AF3", "F3", "F7", "FC5", "FC1", "C3", "T7", "CP5", "CP1", "P3", "P7", "PO3", "O1", "OZ", "PZ",
             "FP2", "AF4", "FZ", "F4", "F8", "FC6", "FC2", "CZ", "C4", "T8", "CP6", "CP2", "P4", "P8", "PO4", "O2"]
    trials = []
    for _ in range(2):
        x = _sim("Healthy", 63, 128)
        rows = [x[ELEC.index(c)] if c in ELEC else x[0] * 0.5 for c in order] + [np.zeros(x.shape[1])] * 8
        trials.append(np.stack(rows))
    with open(tmp_path / "s01.dat", "wb") as f:
        pickle.dump({"data": np.stack(trials), "labels": np.zeros((2, 4))}, f)
    assert read_deap(tmp_path / "s01.dat")[0][0].shape == (32, 60 * 128)
    recs = list(load_deap({"root": tmp_path}))
    assert len(recs) == 2
    _check(recs, ["Healthy"])


def test_build_cache_multi_dataset(tmp_path):
    """Two different datasets -> one harmonized cache."""
    (tmp_path / "ad").mkdir()
    (tmp_path / "modma").mkdir()
    test_openneuro_ad(tmp_path / "ad")
    test_modma(tmp_path / "modma")
    cfg = load_config(None, [f"paths.cache_dir={tmp_path}/cache", "datasets.manifest.enabled=false",
                             "datasets.openneuro_ad.enabled=true", f"datasets.openneuro_ad.root={tmp_path}/ad",
                             "datasets.modma.enabled=true", f"datasets.modma.root={tmp_path}/modma"])
    d = load_cache(build_cache(cfg))
    assert d["X"].shape[1:] == (18, 512)
    assert set(d["dataset"]) == {"openneuro_ad", "modma"}
    assert {str(d["classes"][k]) for k in set(d["y"])} == {"Alzheimer", "Healthy", "Depression"}
