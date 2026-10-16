import numpy as np
import pandas as pd
import streamlit as st

from app.charts.robustness_curve_chart import robustness_curve_figure
from app.charts.waveform_chart import overlay_figure
from app.components.loaders import (
    get_attack_examples,
    get_clean_metrics,
    get_robustness_curve,
    tag_for,
)
from app.components.metric_card import metric_row, not_benchmarked, section_label
from app.styles.theme import (
    data_source,
    disclaimer,
    inject_theme,
    page_header,
    research_note,
)
from src.utils.paths import ARTIFACTS_DIR

st.set_page_config(
    page_title="Overview · ECG Research Project", layout="wide", initial_sidebar_state="auto"
)
inject_theme()
with st.sidebar:
    mode = data_source(default="research")
    disclaimer()

page_header(
    "Workbench / Overview",
    "ECG Research Project",
    "ECG Classification & Adversarial Robustness",
)
research_note(mode)
standard_tag, defended_tag = tag_for(mode, "standard"), tag_for(mode, "defended")
clean = get_clean_metrics(standard_tag)
if clean is None:
    not_benchmarked("Standard-model evaluation")
    st.stop()
m = clean["cnn"]
st.caption(
    f"{'MIT-BIH' if mode == 'research' else 'Synthetic fixture'}  /  {clean['n_test_beats']:,} test beats  /  "
    f"{'Context CNN' if mode == 'research' else 'Standard demo CNN'}  /  decision threshold {clean['threshold']:.3f}"
)
metric_row(
    [
        ("Arrhythmia F1", f"{m['f1']:.1%}"),
        ("Arrhythmia recall", f"{m['recall']:.1%}"),
        ("Precision", f"{m['precision']:.1%}"),
        ("Average precision (AP)", f"{m['pr_auc']:.1%}"),
    ]
)

left, right = st.columns([2.1, 1], gap="large")
with left:
    section_label("01 / A saved adversarial example")
    examples = get_attack_examples(standard_tag)
    wave_path = ARTIFACTS_DIR / "experiments" / standard_tag / "attack_waveforms.npz"
    if examples is not None and len(examples) and wave_path.exists():
        with np.load(wave_path, allow_pickle=False) as waveforms:
            fig = overlay_figure(waveforms["original"][0], waveforms["adversarial"][0])
        st.plotly_chart(fig, width="stretch")
        r = examples.iloc[0]
        st.caption(
            f"Saved example · {str(r['attack']).upper()} · ε = {r['epsilon']:g} · "
            f"test beat {int(r['beat_index'])}. This selected example is not an aggregate result."
        )
    else:
        not_benchmarked("Saved attack waveform")
    st.page_link("pages/2_Adversarial_Lab.py", label="Open the adversarial lab →")
with right:
    section_label("What the result tells us")
    st.subheader("Accuracy needs context.")
    counts = clean["test_class_distribution"]
    majority = int(counts.get("0", 0)) / clean["n_test_beats"]
    st.write(
        f"The selected model reaches **{m['accuracy']:.1%} accuracy**, but detects "
        f"**{m['recall']:.1%}** of annotated arrhythmic beats."
    )
    st.write(
        f"Predicting the majority class on every beat would already score **{majority:.1%}** accuracy."
    )
    st.caption(
        "Read F1, recall and precision together. Sigmoid scores have not been probability-calibrated."
    )
    st.page_link("pages/1_ECG_Explorer.py", label="Inspect individual beats →")

st.divider()
left, right = st.columns([2.1, 1], gap="large")
curves = [
    c
    for tag in (standard_tag, defended_tag)
    if (c := get_robustness_curve(tag)) is not None
]
with left:
    section_label("02 / The robustness tradeoff")
    if curves:
        df = pd.concat(curves, ignore_index=True)
        st.plotly_chart(
            robustness_curve_figure(df, "pgd", "adversarial_f1", "Arrhythmia F1"),
            width="stretch",
        )
        st.caption(
            "V2 research: 20 PGD steps, three starts and a BCE check on a fixed 1,408-beat audit. Demo retains its original settings. Empirical evidence, not certified robustness."
        )
    else:
        not_benchmarked("Robustness sweep")
with right:
    section_label("Standard vs adversarial training")
    defended = get_clean_metrics(defended_tag)
    if defended:
        d = defended["cnn"]
        table = pd.DataFrame(
            {
                "Metric": ["Clean F1", "Clean recall", "Clean precision"],
                "Standard": [f"{m[k]:.1%}" for k in ("f1", "recall", "precision")],
                "Defended": [f"{d[k]:.1%}" for k in ("f1", "recall", "precision")],
            }
        )
        st.dataframe(table, hide_index=True, width="stretch")
        st.write(
            f"Adversarial training changes clean recall by **{100 * (d['recall'] - m['recall']):+.1f} percentage points**."
        )
    st.page_link(
        "pages/3_Robustness_Benchmark.py", label="Compare the full benchmark →"
    )

if mode == "research":
    st.caption(
        "Research curves use the fixed 1,408-beat audit subset; Study evidence reports full-test attacks separately."
    )
