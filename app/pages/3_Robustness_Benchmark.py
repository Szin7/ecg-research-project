import pandas as pd
import streamlit as st
from src.utils.paths import ARTIFACTS_DIR

from app.charts.confusion_matrix_chart import confusion_matrix_figure
from app.charts.robustness_curve_chart import robustness_curve_figure
from app.components.loaders import get_clean_metrics, get_robustness_curve, tag_for
from app.components.metric_card import metric_row, not_benchmarked, section_label
from app.styles.theme import (
    data_source,
    disclaimer,
    inject_theme,
    page_header,
    research_note,
)

st.set_page_config(page_title="Benchmarks · ECG Research Project", layout="wide")
inject_theme()
with st.sidebar:
    mode = data_source(default="research")
    disclaimer()
page_header(
    "03 / Evaluate",
    "Robustness benchmarks",
    "Read the performance cost alongside the measured resistance to attack.",
)
research_note(mode)
tags = [tag_for(mode, kind) for kind in ("standard", "defended")]
labels = dict(
    zip(tags, ["Context CNN" if mode == "research" else "Standard CNN", "Defended CNN"])
)
clean_tab, attack_tab, multiclass_tab = st.tabs(
    ["Clean performance", "Attack sweeps", "AAMI / four classes"]
)
with clean_tab:
    clean = {tag: get_clean_metrics(tag) for tag in tags}
    rows = []
    for tag, result in clean.items():
        if result:
            m = result["cnn"]
            rows.append(
                {
                    "Model": labels[tag],
                    "F1": m["f1"],
                    "Recall": m["recall"],
                    "Precision": m["precision"],
                    "Accuracy": m["accuracy"],
                    "AP": m["pr_auc"],
                    "ROC-AUC": m["roc_auc"],
                    "Threshold": result["threshold"],
                }
            )
    if rows:
        st.dataframe(
            pd.DataFrame(rows).style.format(
                {k: "{:.3f}" for k in rows[0] if k != "Model"}
            ),
            width="stretch",
            hide_index=True,
        )
        st.caption(
            "AP is average precision, stored as pr_auc in the original artifacts. Thresholds were selected on validation data."
        )
    for col, tag in zip(st.columns(2), tags):
        with col:
            section_label(labels[tag])
            result = clean[tag]
            if result:
                st.plotly_chart(
                    confusion_matrix_figure(result["cnn"]["confusion_matrix"]),
                    width="stretch",
                    key=f"confusion-{tag}",
                )
                cm = result["cnn"]["confusion_matrix"]
                st.caption(
                    f"{cm[1][0]:,} missed arrhythmic beats · {cm[0][1]:,} false positives · {result['n_test_beats']:,} total beats"
                )
            else:
                not_benchmarked(labels[tag])
    if all(clean.values()):
        base, defended = [clean[t]["cnn"] for t in tags]
        metric_row(
            [
                ("Δ clean F1", f"{100 * (defended['f1'] - base['f1']):+.1f} pp"),
                (
                    "Δ clean recall",
                    f"{100 * (defended['recall'] - base['recall']):+.1f} pp",
                ),
                (
                    "Δ clean accuracy",
                    f"{100 * (defended['accuracy'] - base['accuracy']):+.1f} pp",
                ),
            ]
        )
        st.caption(
            "Change from standard to adversarially trained model. Better accuracy can coexist with worse arrhythmia detection."
        )
with attack_tab:
    curves = [c for tag in tags if (c := get_robustness_curve(tag)) is not None]
    if not curves:
        not_benchmarked("Attack sweep")
    else:
        df = pd.concat(curves, ignore_index=True)
        attack = st.radio("Attack", sorted(df["attack"].unique()), horizontal=True)
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(
                robustness_curve_figure(df, attack, "adversarial_f1", "Arrhythmia F1"),
                width="stretch",
            )
        with c2:
            st.plotly_chart(
                robustness_curve_figure(
                    df, attack, "adversarial_recall", "Arrhythmia recall"
                ),
                width="stretch",
            )
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(
                robustness_curve_figure(
                    df, attack, "robust_accuracy", "Accuracy under attack"
                ),
                width="stretch",
            )
        with c2:
            st.plotly_chart(
                robustness_curve_figure(
                    df, attack, "attack_success_rate", "Attack success rate"
                ),
                width="stretch",
            )
        st.info(
            "Each curve attacks its own model. V2 uses 20 PGD steps, three starts and a BCE check on a fixed 1,408-beat audit. Demo retains its original settings. "
            "Study evidence separates full-test results and 50-step checks. These attacks do not certify robustness."
        )
        st.download_button(
            "Download sweep data",
            df.to_csv(index=False),
            f"{mode}-robustness.csv",
            "text/csv",
        )
        with st.expander("Inspect exact values"):
            st.dataframe(df, width="stretch", hide_index=True)
with multiclass_tab:
    result = get_clean_metrics(
        (
            "v2_aami_context"
            if (ARTIFACTS_DIR / "active_study.json").exists()
            else "aami_cnn"
        )
        if mode == "research"
        else "demo_aami_cnn"
    )
    if not result:
        not_benchmarked("AAMI evaluation")
    else:
        m = result["metrics"]
        metric_row(
            [
                ("Macro-F1", f"{m['macro_f1']:.1%}"),
                ("Weighted F1", f"{m['weighted_f1']:.1%}"),
                ("Accuracy", f"{m['accuracy']:.1%}"),
            ]
        )
        c1, c2 = st.columns([1.3, 1])
        with c1:
            st.dataframe(
                pd.DataFrame(m["per_class"]).T.style.format(
                    {
                        "precision": "{:.3f}",
                        "recall": "{:.3f}",
                        "f1": "{:.3f}",
                        "support": "{:,.0f}",
                    }
                ),
                width="stretch",
            )
            st.caption(
                "N: normal-group beats · S: supraventricular ectopic · V: ventricular ectopic · F: fusion"
            )
        with c2:
            st.plotly_chart(
                confusion_matrix_figure(
                    m["confusion_matrix"], labels=result["classes"]
                ),
                width="stretch",
            )
        st.caption(
            "The four-class experiment remains exploratory. Macro-F1 gives each class equal weight; weighted F1 can hide rare-class failures."
        )

if mode == "research":
    st.caption(
        "V2 curves use the fixed 1,408-beat record-balanced audit. Full-test results and 50-step checks appear on Study evidence. Only one fusion beat is available in validation."
    )
