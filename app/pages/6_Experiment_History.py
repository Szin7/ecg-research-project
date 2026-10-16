import json

import pandas as pd
import streamlit as st

from app.styles.theme import (
    data_source,
    disclaimer,
    inject_theme,
    page_header,
    research_note,
)
from src.utils.paths import ARTIFACTS_DIR

st.set_page_config(page_title="Experiment History — ECG Research Project", layout="wide")
inject_theme()

page_header(
    "06 / Reproducibility",
    "Run archive",
    "Compare training runs and export their saved results.",
)

with st.sidebar:
    disclaimer()

rows = []
models_dir = ARTIFACTS_DIR / "models"
experiments_dir = ARTIFACTS_DIR / "experiments"

for history_path in sorted(models_dir.glob("*_history.json")):
    h = json.loads(history_path.read_text())
    tag = h["tag"]
    clean_path = experiments_dir / tag / "clean_metrics.json"
    clean = json.loads(clean_path.read_text()) if clean_path.exists() else None
    # the secondary multi-class (AAMI) experiment tracks macro-F1/argmax
    # predictions instead of F1/a validation-selected threshold -- distinguish by
    # which key is present rather than assuming every history is binary
    is_multiclass = "best_val_macro_f1" in h

    row = {
        "tag": tag,
        "mode": h["mode"],
        "task": "multi-class (AAMI)" if is_multiclass else "binary",
        "seed": h["seed"],
        "n_params": h["n_params"],
        "epochs_trained": h["epochs_trained"],
        "train_seconds": round(h["train_seconds"], 1),
    }
    if is_multiclass:
        row["best_val_metric (macro-F1)"] = round(h["best_val_macro_f1"], 4)
        row["test_metric (macro-F1)"] = (
            round(clean["metrics"]["macro_f1"], 4) if clean else None
        )
        row["test_accuracy"] = round(clean["metrics"]["accuracy"], 4) if clean else None
    else:
        row["best_val_metric (F1)"] = round(h["best_val_f1"], 4)
        row["test_metric (F1)"] = round(clean["cnn"]["f1"], 4) if clean else None
        row["test_accuracy"] = round(clean["cnn"]["accuracy"], 4) if clean else None

    rows.append(row)

for path in sorted((ARTIFACTS_DIR / "runs").glob("*/manifest.json")):
    run = json.loads(path.read_text())
    if run.get("status") == "completed":
        config, result = run["config"], run["result"]
        rows.append(
            {
                "tag": path.parent.name,
                "mode": "research",
                "task": config["variant"],
                "seed": config["seed"],
                "n_params": result["n_params"],
                "epochs_trained": result["epochs"],
                "train_seconds": round(result["train_seconds"], 1),
                "best_val_metric": result["best_val_score"],
                "protocol": "subject-disjoint v2",
            }
        )

if not rows:
    st.info("No trained models found yet. Run `scripts/train.py` to produce one.")
else:
    df = pd.DataFrame(rows)
    mode_filter = st.selectbox("Dataset", ["All", "research", "demo"])
    if mode_filter != "All":
        df = df[df["mode"] == mode_filter]
    st.dataframe(df, width="stretch", hide_index=True)
    st.download_button(
        "Download run summary",
        df.to_csv(index=False),
        "ecg-research-project-runs.csv",
        "text/csv",
    )

st.markdown("---")
st.caption(
    "Full robustness curves are on the Robustness Benchmark page -- this page is for comparing training runs at a glance."
)
