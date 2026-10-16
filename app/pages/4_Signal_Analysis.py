import plotly.graph_objects as go
import streamlit as st
import numpy as np
from src.evaluation.peaks import peak_agreement

from app.charts.waveform_chart import waveform_figure
from app.components.metric_card import section_label
from app.styles.theme import (
    COLORS,
    data_source,
    disclaimer,
    inject_theme,
    page_header,
    research_note,
)
from src.data.cache import load_or_build_split
from src.data.source import get_data_source
from src.data.splits import get_split
from src.preprocessing.filtering import remove_baseline_wander
from src.preprocessing.qrs_detect import detect_r_peaks
from src.utils.config import load_config

st.set_page_config(page_title="Signal Analysis — ECG Research Project", layout="wide")
inject_theme()

page_header(
    "04 / Signal processing",
    "Signal analysis",
    "Inspect baseline drift, detected peaks and the amplitude scale.",
)

with st.sidebar:
    mode = data_source()
    disclaimer()

cfg = load_config("dataset.yaml")
source = get_data_source(mode)
records = source.available_records()
if not records:
    st.warning(
        "Raw records are not bundled in the research preview. Select Synthetic demo for an offline walkthrough, or run `python -m scripts.prepare_protocol --download`."
    )
    st.stop()

record_id = st.selectbox("Record", records)
record = source.load_record(str(record_id))

window_s = st.slider("Window (seconds)", 2, 20, 8)
n_samples = min(window_s * record.fs, len(record.signal))
raw_segment = record.signal[:n_samples]
filtered_full = remove_baseline_wander(record.signal, record.fs)
filtered_segment = filtered_full[:n_samples]

detected_peaks = detect_r_peaks(record.signal, record.fs)
detected_in_window = detected_peaks[detected_peaks < n_samples]

col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(
        waveform_figure(
            raw_segment,
            fs=record.fs,
            title="Raw",
            amplitude_label="amplitude (mV)"
            if mode == "research"
            else "amplitude (synthetic units)",
        ),
        width="stretch",
    )
with col2:
    st.plotly_chart(
        waveform_figure(
            filtered_segment,
            fs=record.fs,
            title="Baseline-corrected",
            r_peaks=detected_in_window,
            amplitude_label="amplitude (mV)"
            if mode == "research"
            else "amplitude (synthetic units)",
        ),
        width="stretch",
    )

symbols = {s for labels in cfg["label_map"].values() for s in labels}
reference = np.array(
    [
        peak
        for peak, symbol in zip(record.ann_samples, record.ann_symbols)
        if symbol in symbols and peak < n_samples
    ]
)
agreement = peak_agreement(reference, detected_in_window, record.fs)
st.caption(
    f"Displayed window · 50 ms matching tolerance · peak precision {agreement['precision']:.1%} · peak recall {agreement['recall']:.1%} · {agreement['matched']} matches. This measures timing agreement, not classification accuracy."
)

st.markdown("---")
section_label("PER-BEAT AMPLITUDE DISTRIBUTION")

split = get_split(mode, cfg)
X, y, groups = load_or_build_split(mode, "test", split["test"], cfg)
if len(X):
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=X.max(axis=1),
            marker=dict(color=COLORS["accent"]),
            name="per-beat max amplitude",
        )
    )
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor=COLORS["surface"],
        plot_bgcolor=COLORS["surface"],
        font=dict(color=COLORS["text"]),
        height=280,
        margin=dict(l=40, r=20, t=10, b=40),
        xaxis=dict(
            title="Maximum beat amplitude (z-score units)", gridcolor=COLORS["border"]
        ),
        yaxis=dict(title="count", gridcolor=COLORS["border"]),
    )
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "This is what the FGSM/PGD epsilon sweep on the Robustness Benchmark page is calibrated against -- "
        "epsilon is a fraction of this same per-beat standard deviation."
    )
else:
    st.info("No processed beats to summarize yet.")
