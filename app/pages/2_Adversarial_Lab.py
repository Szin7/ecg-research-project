import numpy as np
import streamlit as st
import torch
import torch.nn as nn

from app.charts.waveform_chart import overlay_figure, perturbation_figure
from app.components.loaders import get_test_data, get_test_scores, get_model, tag_for
from app.components.metric_card import metric_row, not_benchmarked, section_label
from app.components.predictions import predicted_class_and_score
from app.styles.theme import (
    data_source,
    disclaimer,
    inject_theme,
    page_header,
    research_note,
)
from src.attacks.fgsm import fgsm_attack
from src.attacks.pgd import pgd_attack
from src.evaluation.robustness import perturbation_stats

st.set_page_config(page_title="Adversarial Lab — ECG Research Project", layout="wide")
inject_theme()

page_header(
    "02 / Stress test",
    "Adversarial lab",
    "Apply a bounded perturbation and inspect the change in the model’s decision.",
)

with st.sidebar:
    mode = data_source()
    with st.form("attack_controls"):
        attack_name = st.selectbox("Attack", ["PGD", "FGSM"])
        epsilon = st.slider(
            "Perturbation budget ε",
            0.0,
            1.0,
            0.3,
            0.01,
            help="L∞ bound in per-beat z-score units",
        )
        pgd_iters = st.slider("PGD iterations", 1, 50, 10)
        pgd_alpha_ratio = st.slider("PGD step / ε", 0.05, 1.0, 0.25)
        attack_seed = st.number_input(
            "Attack seed", min_value=0, max_value=2147483647, value=42
        )
        st.form_submit_button("Apply attack", type="primary", width="stretch")
    disclaimer()


X, y, groups = get_test_data(mode)
research_note(mode)
if len(y) == 0:
    st.warning(
        f"No processed {mode} data. Run `scripts/prepare_data.py --mode {mode}` first."
    )
    st.stop()

standard = get_model(tag_for(mode, "standard"))
defended = get_model(tag_for(mode, "defended"))
if standard is None:
    not_benchmarked("Standard model")
    st.stop()


scores = get_test_scores(tag_for(mode, "standard"), mode, X)
correct_idx = np.where((scores >= standard.threshold).astype(int) == y)[0]
st.caption(
    f"{len(correct_idx)} of {len(y)} test beats are correctly classified by the standard model and available to attack."
)

state_key = f"cg_selected_idx_{mode}"
if state_key not in st.session_state and len(correct_idx):
    st.session_state[state_key] = int(np.random.default_rng().choice(correct_idx))

col_pick, col_btn = st.columns([3, 1])
with col_pick:
    label_filter = st.radio("Filter", ["Any", "Normal", "Arrhythmia"], horizontal=True)
with col_btn:
    st.write("")
    if st.button("Pick random beat", width="stretch"):
        pool = correct_idx
        if label_filter == "Normal":
            pool = correct_idx[y[correct_idx] == 0]
        elif label_filter == "Arrhythmia":
            pool = correct_idx[y[correct_idx] == 1]
        if len(pool):
            st.session_state[state_key] = int(np.random.default_rng().choice(pool))

pool = correct_idx
if label_filter != "Any":
    pool = correct_idx[y[correct_idx] == (1 if label_filter == "Arrhythmia" else 0)]
if not len(pool):
    st.info("No correctly classified beats match this filter.")
    st.stop()
if st.session_state.get(state_key) not in pool:
    st.session_state[state_key] = int(pool[0])
idx = st.session_state.get(state_key)
if idx is None:
    st.warning("No correctly classified beats available.")
    st.stop()

beat = X[idx]
true_label = int(y[idx])

x_tensor = torch.from_numpy(beat).float().view(1, 1, -1)
y_tensor = torch.tensor([float(true_label)])

with torch.no_grad():
    clean_prob = torch.sigmoid(standard.model(x_tensor)).item()
clean_pred, clean_conf = predicted_class_and_score(clean_prob, standard.threshold)

loss_fn = nn.BCEWithLogitsLoss()
if epsilon == 0:
    x_adv = x_tensor.clone()
elif attack_name == "FGSM":
    x_adv = fgsm_attack(standard.model, x_tensor, y_tensor, epsilon, loss_fn)
else:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(attack_seed))
        x_adv = pgd_attack(
            standard.model,
            x_tensor,
            y_tensor,
            epsilon,
            pgd_alpha_ratio * epsilon,
            pgd_iters,
            loss_fn,
        )

with torch.no_grad():
    adv_prob = torch.sigmoid(standard.model(x_adv)).item()
adv_pred, adv_conf = predicted_class_and_score(adv_prob, standard.threshold)
stats = perturbation_stats(x_tensor[..., :250], x_adv[..., :250])

st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    section_label("ORIGINAL — STANDARD MODEL")
    st.markdown(f"### {clean_pred}")
    st.caption(
        f"Arrhythmia score {clean_conf:.3f} · true label: {'Arrhythmia' if true_label else 'Normal'}"
    )
with col2:
    section_label("ADVERSARIAL — STANDARD MODEL")
    flipped = adv_pred != clean_pred
    st.markdown(f"### {adv_pred} {'⚠' if flipped else ''}")
    st.caption(f"Arrhythmia score {adv_conf:.3f}")
with col3:
    section_label("TRANSFER CHECK — DEFENDED MODEL")
    if defended is None:
        not_benchmarked("Defended model")
    else:
        with torch.no_grad():
            defended_prob = torch.sigmoid(defended.model(x_adv)).item()
        defended_pred, defended_conf = predicted_class_and_score(
            defended_prob, defended.threshold
        )
        recovered = defended_pred == clean_pred
        st.markdown(f"### {defended_pred} {'✓' if recovered and flipped else ''}")
        st.caption(f"Arrhythmia score {defended_conf:.3f}")

st.markdown("---")
section_label("WAVEFORM COMPARISON")
st.plotly_chart(
    overlay_figure(beat[:250], x_adv.squeeze().numpy()[:250]), width="stretch"
)
section_label("PERTURBATION")
st.plotly_chart(
    perturbation_figure(beat[:250], x_adv.squeeze().numpy()[:250]), width="stretch"
)

st.markdown("---")
section_label("PERTURBATION STATISTICS")
metric_row(
    [
        ("L∞", f"{stats['linf_mean']:.4f}"),
        ("L2", f"{stats['l2_mean']:.4f}"),
        ("RMS", f"{stats['rms_mean']:.4f}"),
        ("SNR", f"{stats['snr_db_mean']:.1f} dB"),
    ]
)
st.caption(
    "A small constrained perturbation, bounded to the chosen epsilon budget in the same "
    "per-beat normalized units the model trains on -- described as measured here, not assumed imperceptible."
)

st.caption(
    f"Record {groups[idx]} · test beat {idx} · seed {attack_seed} · standard threshold {standard.threshold:.3f}"
)
st.info(
    "The attack targets the standard model. The defended column measures transfer of that attack, not a white-box attack on the defense. Scores are uncalibrated."
)
