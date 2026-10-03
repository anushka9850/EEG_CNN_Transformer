"""Loaders for the public EEG datasets used in the project.

Each loader is a generator of :class:`Recording` objects so that datasets far larger
than RAM can be streamed through the preprocessing pipeline.

| Class       | Dataset                         | Loader           |
|-------------|---------------------------------|------------------|
| Healthy     | TUH EEG Corpus (normal)         | ``load_tuh``     |
| Epilepsy    | CHB-MIT Scalp EEG               | ``load_chbmit``  |
| Alzheimer   | OpenNeuro ds004504 (+ controls) | ``load_bids``    |
| Parkinson   | OpenNeuro ds002778 (+ controls) | ``load_bids``    |
| Depression  | MODMA 128-ch (+ controls)       | ``load_modma``   |
| Healthy     | PhysioNet EEGMMIDB (baseline)   | ``load_eegmmidb``|
| Healthy     | DEAP                            | ``load_deap``    |
| any         | CSV manifest                    | ``load_manifest``|
"""
from __future__ import annotations

import csv
import logging
import re
from pathlib import Path
from typing import Iterator

import numpy as np

from .harmonize import map_label
from .preprocess import Recording, read_any, read_deap

log = logging.getLogger(__name__)
EEG_EXT = {".edf", ".bdf", ".set", ".fif", ".vhdr", ".gdf"}


def _ci_rglob(root: Path, pattern: str) -> list[Path]:
    """Case-insensitive recursive glob. Handles .edf/.EDF/.Edf etc."""
    # pathlib.rglob is case-sensitive on Linux; collect all and filter
    ext = Path(pattern).suffix.lower()
    stem_pattern = Path(pattern).stem
    if stem_pattern == "*":
        return sorted(f for f in root.rglob("*") if f.is_file() and f.suffix.lower() == ext)
    # for patterns like "S001R*.edf" we use rglob then filter case-insensitively
    return sorted(f for f in root.rglob("*") if f.is_file()
                  and f.suffix.lower() == ext
                  and re.match(stem_pattern.replace("*", ".*"), f.stem, re.IGNORECASE))


def _safe_read(path, **kw):
    try:
        return read_any(path, **kw)
    except Exception as e:
        log.warning("skipping unreadable file %s (%s)", path, e)
        return None


# ------------------------------------------------------------------ CHB-MIT
def parse_chbmit_summary(summary: Path) -> dict[str, list[tuple[float, float]]]:
    """Parse chbXX-summary.txt -> {file_name: [(start_s, end_s), ...]}."""
    out: dict[str, list[tuple[float, float]]] = {}
    cur = None
    starts: list[float] = []
    for line in summary.read_text(errors="ignore").splitlines():
        m = re.match(r"\s*File Name:\s*(\S+)", line)
        if m:
            cur = m.group(1).strip()
            out[cur] = []
            continue
        if cur is None:
            continue
        m = re.search(r"Seizure(?:\s*\d+)?\s*Start Time:\s*(\d+)", line)
        if m:
            starts.append(float(m.group(1)))
            continue
        m = re.search(r"Seizure(?:\s*\d+)?\s*End Time:\s*(\d+)", line)
        if m and starts:
            out[cur].append((starts.pop(0), float(m.group(1))))
    return out


def load_chbmit(cfg: dict) -> Iterator[Recording]:
    root = Path(cfg["root"])
    seizures: dict[str, list] = {}
    for s in root.rglob("*summary.txt"):
        seizures.update(parse_chbmit_summary(s))
    # also support per-file *.edf.seizures annotations being absent: summary is the source of truth
    # Case-insensitive glob to handle .edf/.EDF
    files = _ci_rglob(root, "*.edf")
    by_subj: dict[str, list[Path]] = {}
    for f in files:
        # Support both PhysioNet layout (chbXX/chbXX_YY.edf) and Kaggle mirror
        # (nested folders with chbXX in name or parent folder)
        subj = re.match(r"(chb\d+)", f.name, re.IGNORECASE)
        if subj:
            subj_id = subj.group(1).lower()
        else:
            # Try parent directory names for Kaggle mirror layout
            for parent in f.parents:
                m = re.match(r"(chb\d+)", parent.name, re.IGNORECASE)
                if m:
                    subj_id = m.group(1).lower()
                    break
            else:
                subj_id = f.parent.name.lower()
        subj_id = "chb01" if subj_id == "chb21" else subj_id  # chb21 is the same patient as chb01
        by_subj.setdefault(subj_id, []).append(f)
    maxf = cfg.get("max_files_per_subject")
    for subj, fl in by_subj.items():
        with_sz = [f for f in fl if seizures.get(f.name)]
        without = [f for f in fl if not seizures.get(f.name)]
        chosen = (with_sz + without)[: maxf or None]
        for f in chosen:
            r = _safe_read(f)
            if r is None:
                continue
            data, fs, names = r
            yield Recording(data, fs, names, "Epilepsy", f"chbmit_{subj}", "chbmit", str(f),
                            seizures=seizures.get(f.name, []), is_epilepsy_patient=True)


# ------------------------------------------------------------------ TUH
def load_tuh(cfg: dict) -> Iterator[Recording]:
    root = Path(cfg["root"])
    files = [f for f in _ci_rglob(root, "*.edf") if "abnormal" not in str(f).lower()]
    for f in files[: cfg.get("max_files") or None]:
        r = _safe_read(f)
        if r is None:
            continue
        subj = f.name.split("_")[0]
        yield Recording(*r, label=cfg.get("label", "Healthy"), subject=f"tuh_{subj}", dataset="tuh", path=str(f))


# ------------------------------------------------------------------ OpenNeuro / BIDS
def _participants(root: Path) -> dict[str, str]:
    p = root / "participants.tsv"
    if not p.exists():
        return {}
    out = {}
    with open(p) as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    for r in rows:
        pid = r.get("participant_id", "")
        grp = None
        for key in ("Group", "group", "diagnosis", "Diagnosis", "dx", "type"):
            if key in r and r[key] not in (None, "", "n/a"):
                grp = r[key]
                break
        if grp is not None:
            out[pid] = grp
    return out


def load_bids(cfg: dict, name: str) -> Iterator[Recording]:
    """Generic BIDS EEG loader (OpenNeuro ds004504 Alzheimer, ds002778 Parkinson, ...).

    Label per subject from participants.tsv (Group column: A=Alzheimer, C=control,
    F=FTD is skipped) or from the subject id prefix (sub-hc*, sub-pd*)."""
    root = Path(cfg["root"])
    groups = _participants(root)
    group_map = {"A": "Alzheimer", "C": "Healthy", "HC": "Healthy", "PD": "Parkinson",
                 "MDD": "Depression", "F": None, "FTD": None}
    group_map.update(cfg.get("group_map", {}))
    files = [f for f in sorted(root.rglob("*_eeg.*")) if f.suffix.lower() in EEG_EXT]
    raw_files = [f for f in files if "derivatives" not in f.parts]
    files = raw_files or files
    for f in files:
        m = re.search(r"(sub-[A-Za-z0-9]+)", str(f))
        if not m:
            continue
        sid = m.group(1)
        grp = groups.get(sid)
        if grp is None:
            m2 = re.match(r"sub-([a-zA-Z]+)", sid)
            if m2:
                pre = m2.group(1).upper()
                grp = pre
            else:
                continue
        label = group_map.get(str(grp).upper(), map_label(grp))
        if label is None:
            continue
        r = _safe_read(f)
        if r is None:
            continue
        yield Recording(*r, label=label, subject=f"{name}_{sid}", dataset=name, path=str(f))


# ------------------------------------------------------------------ MODMA
def _modma_labels(cfg: dict) -> dict[str, str]:
    root = Path(cfg["root"])
    labels: dict[str, str] = {}
    lf = cfg.get("labels_file")
    if lf:
        with open(lf) as fh:
            for r in csv.DictReader(fh):
                labels[str(r["subject"]).zfill(8)] = map_label(r["label"])
        return labels
    for x in root.rglob("*.xlsx"):
        try:
            import pandas as pd
            df = pd.read_excel(x)
            sc = next(c for c in df.columns if "subject" in str(c).lower())
            tc = next(c for c in df.columns if str(c).lower().strip() in ("type", "group", "label"))
            for s, t in zip(df[sc], df[tc]):
                labels[str(s).split(".")[0].zfill(8)] = map_label(t)
        except Exception as e:
            log.warning("could not parse %s: %s", x, e)
    return labels


def load_modma(cfg: dict) -> Iterator[Recording]:
    root = Path(cfg["root"])
    labels = _modma_labels(cfg)
    files = sorted(list(root.rglob("*.mat")) + [f for f in root.rglob("*") if f.suffix.lower() in EEG_EXT])
    for f in files:
        m = re.match(r"(\d{8})", f.name)
        if not m:
            continue
        sid = m.group(1)
        label = labels.get(sid)
        if label is None:  # MODMA id convention: 0201xxxx = MDD, 0203xxxx = HC
            label = "Depression" if sid.startswith("0201") else "Healthy" if sid.startswith("0203") else None
            if label is None:
                continue
        r = _safe_read(f, fs=cfg.get("fs", 250))
        if r is None:
            continue
        yield Recording(*r, label=label, subject=f"modma_{sid}", dataset="modma", path=str(f))


# ------------------------------------------------------------------ EEGMMIDB
def load_eegmmidb(cfg: dict) -> Iterator[Recording]:
    root = Path(cfg["root"])
    runs = {int(r) for r in cfg.get("runs", [1, 2])}
    # Case-insensitive search for subject directories
    subs = sorted({p.name for p in root.rglob("*") if p.is_dir() and re.match(r"S\d{3}$", p.name, re.IGNORECASE)})
    max_subj = cfg.get("max_subjects")
    subs = subs[: max_subj or None]
    for s in subs:
        for f in _ci_rglob(root, f"{s}R*.edf"):
            m = re.search(r"R(\d+)", f.stem, re.IGNORECASE)
            if not m:
                continue
            run = int(m.group(1))
            if run not in runs:
                continue
            r = _safe_read(f)
            if r is None:
                continue
            yield Recording(*r, label="Healthy", subject=f"eegmmidb_{s}", dataset="eegmmidb", path=str(f))


# ------------------------------------------------------------------ DEAP
def load_deap(cfg: dict) -> Iterator[Recording]:
    root = Path(cfg["root"])
    # Case-insensitive glob for .dat files
    files = sorted(f for f in root.rglob("*") if f.suffix.lower() == ".dat" and re.match(r"s\d+", f.stem, re.IGNORECASE))
    for f in files:
        try:
            trials = read_deap(f)
        except Exception as e:
            log.warning("skip %s: %s", f, e)
            continue
        for k, (data, fs, names) in enumerate(trials):
            yield Recording(data, fs, names, "Healthy", f"deap_{f.stem}", "deap", f"{f}#trial{k}")


# ------------------------------------------------------------------ manifest (any data)
def load_manifest(cfg: dict) -> Iterator[Recording]:
    """CSV with columns: path,label,subject[,dataset][,fs][,seizures].
    ``seizures`` = "start-end;start-end" in seconds (marks an epilepsy recording).
    Relative paths are resolved against the manifest's folder."""
    mf = Path(cfg["file"])
    with open(mf) as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        p = Path(r["path"])
        if not p.is_absolute():
            p = mf.parent / p
        res = _safe_read(p, fs=float(r["fs"]) if r.get("fs") else None)
        if res is None:
            continue
        sz = []
        if r.get("seizures"):
            for part in r["seizures"].split(";"):
                if part.strip():
                    a, b = part.split("-")
                    sz.append((float(a), float(b)))
        label = map_label(r["label"])
        ds = r.get("dataset") or "manifest"
        yield Recording(*res, label=label, subject=f"{ds}_{r['subject']}", dataset=ds, path=str(p),
                        seizures=sz, is_epilepsy_patient=(label == "Epilepsy"))


# ------------------------------------------------------------------ data check
def check_datasets(cfg: dict):
    """Print summary of each configured dataset: found? #subjects, #recordings, labels, fs, channels."""
    import json
    from collections import Counter
    for name, dcfg in cfg["datasets"].items():
        if not dcfg or not dcfg.get("enabled"):
            continue
        if name == "manifest":
            continue
        root = Path(dcfg.get("root", ""))
        if not root.exists():
            print(f"{name:15s}  found: False  (not found: {root})")
            log.warning("SKIP %-15s  (not found: %s)", name, root)
            continue
        if name not in LOADERS:
            print(f"{name:15s}  found: False  (unknown loader)")
            log.warning("SKIP %-15s  (unknown loader)", name)
            continue
        recs = []
        subjects = set()
        label_counts: Counter = Counter()
        fs_set = set()
        ch_set = set()
        try:
            for rec in LOADERS[name](dcfg):
                recs.append(rec)
                subjects.add(rec.subject)
                label_counts[rec.label] += 1
                fs_set.add(rec.fs)
                ch_set.add(len(rec.ch_names))
        except Exception as e:
            print(f"{name:15s}  found: True   ERROR reading: {e}")
            log.error("ERROR loading %s: %s", name, e)
            continue
        print(f"{name:15s}  found: True   #subjects={len(subjects)}, #recordings={len(recs)}, labels={dict(label_counts)}, fs={sorted(fs_set)}, channels={sorted(ch_set)}")
        log.info("%-15s  %d subjects, %d recordings, labels=%s, fs=%s, ch=%s",
                 name, len(subjects), len(recs), dict(label_counts),
                 sorted(fs_set), sorted(ch_set))


LOADERS = {
    "chbmit": lambda c: load_chbmit(c),
    "tuh": lambda c: load_tuh(c),
    "openneuro_ad": lambda c: load_bids(c, "openneuro_ad"),
    "openneuro_pd": lambda c: load_bids(c, "openneuro_pd"),
    "modma": lambda c: load_modma(c),
    "eegmmidb": lambda c: load_eegmmidb(c),
    "deap": lambda c: load_deap(c),
    "manifest": lambda c: load_manifest(c),
}


def iter_recordings(cfg: dict) -> Iterator[Recording]:
    for name, dcfg in cfg["datasets"].items():
        if not dcfg or not dcfg.get("enabled"):
            continue
        if name not in LOADERS:
            raise KeyError(f"unknown dataset '{name}'")
        root = Path(dcfg.get("root", ""))
        if name != "manifest" and not root.exists():
            log.warning("SKIP dataset '%s': root not found (%s)", name, root)
            continue
        if name == "manifest":
            mf = Path(dcfg.get("file", ""))
            if not mf.exists():
                log.warning("SKIP dataset 'manifest': file not found (%s)", mf)
                continue
        log.info("loading dataset: %s", name)
        yield from LOADERS[name](dcfg)
