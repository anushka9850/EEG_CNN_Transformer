"""Interactive research dashboard for EEG file inference."""
import json
import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eegct.explain import explain_window, plot_explanation  # noqa: E402
from eegct.predict import predict_file  # noqa: E402
from eegct.train import load_model  # noqa: E402

st.set_page_config(page_title="EEG analysis workspace", page_icon=":material/neurology:", layout="wide")
st.session_state.setdefault("analysis", None)
st.session_state.setdefault("analysis_error", None)


def _default_ckpt() -> str:
    """Return the best available checkpoint path."""
    for p in [ROOT / "outputs_real" / "best_model.pt", ROOT / "outputs" / "best_model.pt"]:
        if p.is_file():
            return str(p)
    return str(ROOT / "outputs" / "best_model.pt")


@st.cache_resource
def get_model(path):
    return load_model(path)


@st.cache_data
def get_samples():
    return sorted((ROOT / "data" / "synthetic").rglob("*.edf"))


@st.cache_data
def get_sample_labels():
    manifest = ROOT / "data" / "synthetic" / "manifest.csv"
    if not manifest.exists():
        return {}
    rows = pd.read_csv(manifest)
    return {
        Path(str(row.path).replace("\\", "/")).name: str(row.label)
        for row in rows.itertuples()
    }


st.title("EEG analysis workspace", icon=":material/neurology:")
st.caption("Research prototype for EEG classification and explainability. Not for clinical use.")

with st.sidebar:
    st.header("Model")
    ckpt = st.text_input("Checkpoint", _default_ckpt())
    if not Path(ckpt).is_file():
        st.error("Checkpoint not found.")
        st.caption("Train a model with `python -m eegct all`, or point to an existing checkpoint.")
        st.stop()
    try:
        model, ck = get_model(ckpt)
    except Exception as exc:
        st.error(f"Could not load checkpoint: {exc}")
        st.stop()

    # Class names are loaded from the checkpoint, not hard-coded
    classes = ck["classes"]
    st.success(f"Ready · epoch {ck['epoch']} · validation macro-F1 {ck['val_f1']:.3f}")
    with st.expander("Model details"):
        st.write("**Task:**", ck["config"]["task"])
        st.write("**Classes:**", ", ".join(classes))
        st.write("**Montage:**", ck["config"]["harmonization"]["montage"])
        st.write("**Channels:**", len(ck["channels"]))
        st.write("**Sampling rate:**", f"{ck['fs']} Hz")
        st.write("**Window length:**", f"{ck['config']['windows']['length_s']} s")
    overlap = st.slider("Inference window overlap", 0.0, 0.75, 0.0, 0.25)

st.header("Analyze a recording")
source_mode = st.radio("Recording source", ["Bundled test sample", "Upload a recording"], horizontal=True)
samples = get_samples()
sample_labels = get_sample_labels()
sample_path = None
uploaded = None
expected_label = None
fs_in = 250.0

with st.form("analysis_form"):
    if source_mode == "Bundled test sample":
        if samples:
            datasets = sorted({sample.parent.name for sample in samples})
            selected_dataset = st.selectbox("Dataset", datasets)
            options = [sample for sample in samples if sample.parent.name == selected_dataset]
            sample_path = st.selectbox(
                "Recording",
                options,
                format_func=lambda path: f"{sample_labels.get(path.name, 'Synthetic')} · {path.name}",
            )
            expected_label = sample_labels.get(sample_path.name)
            if expected_label:
                st.caption(f"Synthetic reference class: {expected_label}")
        else:
            st.info("No bundled samples found. Generate them with `python -m eegct synth`.")
    else:
        uploaded = st.file_uploader(
            "Choose an EEG recording",
            type=["edf", "bdf", "fif", "mat", "npy", "csv", "dat"],
        )
        fs_in = st.number_input("Sampling rate for MAT / NPY / CSV (Hz)", min_value=1.0, value=250.0)
    submitted = st.form_submit_button("Analyze recording", type="primary", icon=":material/analytics:")

if submitted:
    source_path = sample_path
    temporary_path = None
    try:
        if source_mode == "Upload a recording":
            if uploaded is None:
                raise ValueError("Choose a recording before starting analysis.")
            with tempfile.NamedTemporaryFile(suffix=Path(uploaded.name).suffix, delete=False) as tmp:
                tmp.write(uploaded.getbuffer())
                temporary_path = Path(tmp.name)
            source_path = temporary_path
        if source_path is None:
            raise ValueError("Choose a bundled sample before starting analysis.")

        with st.spinner("Preprocessing recording and calculating explanations..."):
            result, windows, explanation = predict_file(
                ckpt,
                source_path,
                fs=fs_in,
                overlap=overlap,
                model_bundle=(model, ck),
            )
        st.session_state.analysis = {
            "result": result,
            "windows": windows,
            "explanation": explanation,
            "expected_label": expected_label,
            "filename": Path(source_path).name,
        }
        st.session_state.analysis_error = None
    except Exception as exc:
        st.session_state.analysis = None
        st.session_state.analysis_error = str(exc)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

if st.session_state.analysis_error:
    st.error(f"Analysis failed: {st.session_state.analysis_error}")

analysis = st.session_state.analysis
if analysis:
    result = analysis["result"]
    st.header("Recording results")
    metrics = st.container(horizontal=True)
    with metrics:
        st.metric("Predicted class", result["prediction"], border=True)
        st.metric("Mean confidence", f"{result['confidence']:.1%}", border=True)
        st.metric("Windows analyzed", result["n_windows"], border=True)
        if analysis["expected_label"]:
            st.metric("Synthetic reference", analysis["expected_label"], border=True)

    st.caption(f"File: {analysis['filename']} · Influential channels: {', '.join(result['top_channels'])}")
    st.subheader("XAI: what drove this prediction?")
    st.caption(
        "Grad-CAM highlights electrode and time regions that influenced the selected class. "
        "Transformer attention rollout shows which moments received the most attention."
    )
    with st.container(horizontal=True):
        selected_window = st.slider(
            "Window to explain",
            min_value=1,
            max_value=result["n_windows"],
            value=result["explained_window"] + 1,
            help="Choose the EEG segment used to calculate the explanation.",
        ) - 1
        selected_class = st.selectbox(
            "Explain class",
            classes,
            index=classes.index(result["prediction"]),
            help="Grad-CAM is recalculated to explain this class for the selected segment.",
        )

    selected_start = result["window_starts_s"][selected_window]
    window_length = ck["config"]["windows"]["length_s"]
    class_index = classes.index(selected_class)
    selected_probability = result["window_probabilities"][selected_window][class_index]
    st.caption(
        f"Explaining {selected_class} in window {selected_window + 1} "
        f"({selected_start:.1f}–{selected_start + window_length:.1f} s); "
        f"window probability: {selected_probability:.1%}."
    )
    selected_explanation = explain_window(
        model,
        analysis["windows"][selected_window],
        class_index,
    )
    explanation_tab, probabilities_tab, timeline_tab = st.tabs(
        ["XAI explanation", "Class probabilities", "Window timeline"]
    )
    with probabilities_tab:
        probability_mode = st.segmented_control(
            "Probability view",
            ["Recording average", "Selected window"],
            default="Recording average",
        )
        probability_values = (
            result["probabilities"]
            if probability_mode == "Recording average"
            else result["window_probabilities"][selected_window]
        )
        probability_data = pd.DataFrame(
            {"Probability": probability_values}
        ).sort_values("Probability", ascending=True)
        st.bar_chart(probability_data, horizontal=True)
    with timeline_tab:
        window_data = pd.DataFrame(
            result["window_probabilities"],
            columns=classes,
            index=np.round(result["window_starts_s"], 1),
        )
        window_data.index.name = "Window start (s)"
        st.line_chart(window_data)
    with explanation_tab:
        st.info(
            "Brighter heat-map regions indicate stronger Grad-CAM influence. "
            "The orange curve shows Transformer attention over time; channel importance ranks electrodes."
        )
        fig = plot_explanation(
            analysis["windows"][selected_window],
            selected_explanation,
            ck["channels"],
            classes,
            ck["fs"],
            title=f"{selected_class} explanation - window {selected_window + 1}",
        )
        st.pyplot(fig)
        plt.close(fig)
        channel_data = pd.DataFrame(
            {"Channel": ck["channels"], "Importance": selected_explanation["channel_importance"]}
        ).sort_values("Importance", ascending=False)
        st.dataframe(channel_data.head(8), hide_index=True, width="stretch")

    st.download_button(
        "Download results as JSON",
        data=json.dumps(result, indent=2),
        file_name=f"{Path(analysis['filename']).stem}_results.json",
        mime="application/json",
        icon=":material/download:",
    )