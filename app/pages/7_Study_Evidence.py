import json
import pandas as pd
import plotly.express as px
import streamlit as st
from app.components.loaders import get_clean_metrics
from app.styles.theme import COLORS, disclaimer, inject_theme, page_header
from src.utils.paths import ARTIFACTS_DIR

st.set_page_config(page_title="Study evidence · ECG Research Project", layout="wide")
inject_theme()
with st.sidebar:
    disclaimer()
page_header(
    "07 / Evidence",
    "What changed, and what held up?",
    "Three fixed seeds, a subject-disjoint split, and adaptive attacks.",
)
path = ARTIFACTS_DIR / "active_study.json"
if not path.exists():
    st.info("The V2 study has not been published yet.")
    st.stop()
active = json.loads(path.read_text())
study = ARTIFACTS_DIR / active["study_dir"]
st.write(
    "**18 training records · 3 validation records · 22 test records.** Record 201 is excluded. Checkpoint and threshold selection use validation scores."
)
seeds = pd.read_csv(study / "seed_results.csv")
clean, attacks, errors = st.tabs(
    ["Across training seeds", "Adaptive attack evidence", "Where the model fails"]
)
with clean:
    st.subheader("How much does the training seed change the result?")
    fig = px.strip(
        seeds[seeds.variant != "aami_context"],
        x="variant",
        y="f1",
        color="variant",
        hover_data=["seed", "recall"],
        color_discrete_map={
            "morphology": COLORS["text_muted"],
            "context": COLORS["accent"],
            "context_defended": COLORS["compare"],
        },
        template="plotly_white",
    )
    fig.update_layout(
        showlegend=False, height=310, xaxis_title="", yaxis_title="Test F1"
    )
    fig.update_yaxes(tickformat=".0%")
    fig.update_xaxes(
        tickvals=["morphology", "context", "context_defended"],
        ticktext=["Waveform", "Waveform + past RR", "Context + adversarial training"],
    )
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "Each dot is one training seed. Seed variation is not a confidence interval across subjects."
    )
    with st.expander("Inspect all 12 training runs"):
        st.dataframe(seeds, hide_index=True, width="stretch")
    st.download_button(
        "Download all seed results",
        seeds.to_csv(index=False),
        "v2-seed-results.csv",
        "text/csv",
    )
    intervals = []
    for variant in ["morphology", "context", "context_defended"]:
        result = get_clean_metrics("v2_" + variant)
        for metric, ci in result["subject_bootstrap_95"].items():
            intervals.append(
                {
                    "variant": variant,
                    "metric": metric,
                    "point": result["cnn"][metric],
                    **ci,
                }
            )
    st.subheader("Uncertainty across test subjects")
    st.dataframe(pd.DataFrame(intervals), hide_index=True, width="stretch")
    st.caption(
        "95% percentile intervals from 2,000 whole-subject resamples, conditional on this model and database."
    )
with attacks:
    frames = []
    for path in sorted(study.glob("attacks_*/metrics.csv")):
        manifest = json.loads((path.parent / "manifest.json").read_text())
        if manifest["status"] == "completed":
            frame = pd.read_csv(path)
            frame["pipeline"] = path.parent.name.removeprefix("attacks_")
            frames.append(frame)
    if frames:
        frame = pd.concat(frames, ignore_index=True)
        st.subheader("Complete test set · 49,685 beats")
        st.dataframe(
            frame[frame.scope == "full_test"], hide_index=True, width="stretch"
        )
        st.subheader("Fixed record-balanced audit · 1,408 beats")
        st.dataframe(
            frame[frame.scope != "full_test"], hide_index=True, width="stretch"
        )
        st.download_button(
            "Download attack evidence",
            frame.to_csv(index=False),
            "v2-attack-evidence.csv",
            "text/csv",
        )
    else:
        st.info("Adaptive attack evaluations are still running.")
    st.warning(
        "Waveform-only threat model: annotated peak positions and past-RR features remain fixed. These tests do not establish certified robustness or cover timing attacks."
    )
with errors:
    variant = st.selectbox("Variant", list(active["selected_runs"]))
    st.dataframe(
        pd.read_csv(study / f"{active['selected_runs'][variant]}_errors.csv"),
        hide_index=True,
        width="stretch",
    )
    st.caption("Binary: 0 = N, 1 = S/V/F. Four-class: 0 N, 1 S, 2 V, 3 F.")
    st.info(
        "Only one fusion beat is present in validation. Minority-class selection and generalization remain weakly established."
    )
