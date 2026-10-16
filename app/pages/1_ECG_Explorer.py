import numpy as np
import streamlit as st
import torch

from app.charts.waveform_chart import waveform_figure
from app.components.loaders import get_test_data, get_model, tag_for
from app.components.metric_card import not_benchmarked, section_label
from app.components.predictions import predicted_class_and_score
from app.styles.theme import (
    data_source,
    disclaimer,
    inject_theme,
    page_header,
    research_note,
)

st.set_page_config(page_title="ECG Explorer — ECG Research Project", layout="wide")
inject_theme()

page_header(
    "01 / Inspect",
    "ECG explorer",
    "Compare an annotated heartbeat with the model’s decision.",
)

with st.sidebar:
    mode = data_source()
    disclaimer()


X, y, groups = get_test_data(mode)
research_note(mode)
if len(y) == 0:
    st.warning(
        f"No processed {mode} data found. Run `scripts/prepare_data.py --mode {mode}` first."
    )
    st.stop()

record_options = sorted(set(groups.tolist()), key=str)
record = st.selectbox("Record", record_options)
indices = np.where(groups == record)[0]
beat_pos = st.slider("Beat index within record", 0, len(indices) - 1, 0)
idx = indices[beat_pos]

beat = X[idx]
true_label = int(y[idx])
true_label_name = "Arrhythmia" if true_label == 1 else "Normal"

col1, col2 = st.columns([2, 1])
with col1:
    st.plotly_chart(
        waveform_figure(beat[:250], title=f"Record {record} — beat {beat_pos}"),
        width="stretch",
    )

with col2:
    section_label("GROUND TRUTH")
    st.markdown(f"### {true_label_name}")

    kind = st.selectbox(
        "Model",
        ["standard", "defended"],
        format_func=lambda k: "Standard CNN" if k == "standard" else "Defended CNN",
    )
    tag = tag_for(mode, kind)
    loaded = get_model(tag)
    if loaded is None:
        not_benchmarked(f"{tag} checkpoint")
    else:
        x_tensor = torch.from_numpy(beat).float().view(1, 1, -1)
        with torch.no_grad():
            prob = torch.sigmoid(loaded.model(x_tensor)).item()
        pred_name, confidence = predicted_class_and_score(prob, loaded.threshold)

        section_label("PREDICTION")
        correct = pred_name == true_label_name
        st.markdown(f"### {pred_name} {'✓' if correct else '✗'}")
        st.progress(min(max(confidence, 0.0), 1.0))
        st.caption(
            f"Arrhythmia score {confidence:.3f} · decision threshold {loaded.threshold:.3f}"
        )

        st.caption(
            "The score is not a calibrated probability. A label changes when the score crosses the threshold."
        )

if len(beat) > 250:
    st.caption(
        "Four past-RR context features accompany the 250 waveform samples shown here."
    )
