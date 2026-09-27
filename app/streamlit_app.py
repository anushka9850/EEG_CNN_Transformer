"""Web demo: upload an EEG recording -> disease prediction + explainability.

Run:  streamlit run app/streamlit_app.py
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eegct.explain import plot_explanation  # noqa: E402
from eegct.predict import predict_file  # noqa: E402
from eegct.train import load_model  # noqa: E402

st.set_page_config(page_title="Explainable EEG Diagnosis", page_icon="🧠", layout="wide")


@st.cache_resource
def get_model(path):
    return load_model(path)


st.title("🧠 Explainable CNN-Transformer EEG Diagnosis")
st.caption("Unified multi-disease EEG classification with Grad-CAM and Transformer-attention explanations. "
           "Research prototype - not a medical device.")

with st.sidebar:
    st.header("Model")
    ckpt = st.text_input("Checkpoint", str(ROOT / "outputs" / "best_model.pt"))
    if not Path(ckpt).exists():
        st.error("Checkpoint not found. Train first:  python -m eegct all")
        st.stop()
    model, ck = get_model(ckpt)
    st.success(f"Loaded ({ck['config']['task']} task, epoch {ck['epoch']}, val macro-F1 {ck['val_f1']:.3f})")
    st.write("**Classes:**", ", ".join(ck["classes"]))
    st.write(f"**Montage:** {ck['config']['harmonization']['montage']} ({len(ck['channels'])} ch) "
             f"@ {ck['fs']} Hz, {ck['config']['windows']['length_s']} s windows")
    st.header("Input")
    fs_in = st.number_input("Sampling rate (only for .mat/.npy/.csv)", value=250.0)
    overlap = st.slider("Window overlap for inference", 0.0, 0.75, 0.0, 0.25)

tab_up, tab_sample = st.tabs(["Upload EEG file", "Use a sample recording"])
path = None
with tab_up:
    up = st.file_uploader("EDF / BDF / FIF / MAT file", type=["edf", "bdf", "fif", "mat", "npy", "csv"])
    if up is not None:
        tmp = Path(tempfile.mkdtemp()) / up.name
        tmp.write_bytes(up.getbuffer())
        path = tmp
with tab_sample:
    samples = sorted((ROOT / "data" / "synthetic").rglob("*.edf"))
    if samples:
        choice = st.selectbox("Sample", ["-"] + [str(s.relative_to(ROOT)) for s in samples])
        if choice != "-" and path is None:
            path = ROOT / choice
    else:
        st.info("No samples yet - run `python -m eegct synth`.")

if path is None:
    st.stop()

with st.spinner("Harmonizing, preprocessing and classifying..."):
    res, X, exp = predict_file(ckpt, path, fs=fs_in, overlap=overlap, model_bundle=(model, ck))

c1, c2, c3 = st.columns(3)
c1.metric("Prediction", res["prediction"])
c2.metric("Confidence", f"{res['confidence']:.1%}")
c3.metric("Windows analysed", res["n_windows"])

left, right = st.columns([1, 2])
with left:
    st.subheader("Class probabilities (recording level)")
    st.bar_chart(pd.Series(res["probabilities"]).rename("probability"))
    st.subheader("Most influential channels")
    st.write(", ".join(res["top_channels"]))
with right:
    st.subheader("Probability over time (per window)")
    dfw = pd.DataFrame(res["window_probabilities"], columns=ck["classes"],
                       index=np.round(res["window_starts_s"], 1))
    dfw.index.name = "window start (s)"
    st.line_chart(dfw)

st.subheader(f"Explanation of the most characteristic window (#{res['explained_window']})")
fig = plot_explanation(X[res["explained_window"]], exp, ck["channels"], ck["classes"], ck["fs"],
                       title=f"Prediction: {res['prediction']}")
st.pyplot(fig)
st.caption("Red overlay / heat-map = Grad-CAM: which electrodes (bipolar derivations) and time points drove the "
           "decision. Orange = Transformer attention rollout from the [CLS] token over time.")
