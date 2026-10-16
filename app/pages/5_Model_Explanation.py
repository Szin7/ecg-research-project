import streamlit as st
import torch

from app.charts.waveform_chart import attribution_figure
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
from src.attacks.pgd import pgd_attack
from src.explainability.integrated_gradients import integrated_gradients
from src.explainability.saliency import compute_saliency

st.set_page_config(page_title="Model Explanation — ECG Research Project", layout="wide")
inject_theme()

page_header(
    "05 / Interpret",
    "Model attribution",
    "Inspect sensitivity of the arrhythmia logit to the input waveform.",
)

with st.sidebar:
    mode = data_source()
    method = st.radio("Attribution method", ["Saliency", "Integrated Gradients"])
    epsilon = st.slider("Adversarial epsilon for comparison", 0.0, 1.0, 0.3, 0.01)
    disclaimer()


X, y, groups = get_test_data(mode)
research_note(mode)
if len(y) == 0:
    st.warning(
        f"No processed {mode} data. Run `scripts/prepare_data.py --mode {mode}` first."
    )
    st.stop()

standard = get_model(tag_for(mode, "standard"))
if standard is None:
    not_benchmarked("Standard model")
    st.stop()

idx = st.slider("Beat index", 0, len(y) - 1, 0)
beat = X[idx]
true_label = int(y[idx])

x_tensor = torch.from_numpy(beat).float().view(1, 1, -1)
y_tensor = torch.tensor([float(true_label)])

with torch.no_grad():
    clean_prob = torch.sigmoid(standard.model(x_tensor)).item()

x_adv = (
    pgd_attack(standard.model, x_tensor, y_tensor, epsilon, epsilon * 0.25, 10)
    if epsilon > 0
    else x_tensor.clone()
)
with torch.no_grad():
    adv_prob = torch.sigmoid(standard.model(x_adv)).item()

if method == "Saliency":
    clean_attr = compute_saliency(standard.model, x_tensor)
    adv_attr = compute_saliency(standard.model, x_adv)
else:
    clean_attr = integrated_gradients(standard.model, x_tensor)
    adv_attr = integrated_gradients(standard.model, x_adv)

col1, col2 = st.columns(2)
with col1:
    section_label("CLEAN")
    pred, conf = predicted_class_and_score(clean_prob, standard.threshold)
    st.markdown(f"prediction: **{pred}** · arrhythmia score {conf:.3f}")
    st.plotly_chart(
        attribution_figure(
            beat[:250], clean_attr.squeeze().numpy()[:250], title="Clean attribution"
        ),
        width="stretch",
    )
with col2:
    section_label("ADVERSARIAL")
    pred, conf = predicted_class_and_score(adv_prob, standard.threshold)
    st.markdown(f"prediction: **{pred}** · arrhythmia score {conf:.3f}")
    st.plotly_chart(
        attribution_figure(
            x_adv.squeeze().numpy()[:250],
            adv_attr.squeeze().numpy()[:250],
            title="Adversarial attribution",
        ),
        width="stretch",
    )

st.caption(
    "Attribution shows what the model responded to, not a claim about physiological causality. Shading shows absolute attribution, normalized separately per chart; compare locations, not absolute magnitudes. Model scores are uncalibrated."
)

if len(beat) > 250:
    import pandas as pd

    st.subheader("Timing-feature attribution")
    st.dataframe(
        pd.DataFrame(
            {
                "Feature": [
                    "Previous RR",
                    "Past 10 mean RR",
                    "Previous / local",
                    "Previous / preceding",
                ],
                "Log value": beat[250:],
                "Clean attribution": clean_attr.squeeze().numpy()[250:],
                "Adversarial attribution": adv_attr.squeeze().numpy()[250:],
            }
        ),
        hide_index=True,
    )
