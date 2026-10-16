from html import escape
from pathlib import Path

import streamlit as st
from src.utils.paths import ARTIFACTS_DIR

THEME_CSS = (Path(__file__).parent / "theme.css").read_text()
COLORS = {
    "bg": "#f6f5f0",
    "surface": "#fffefa",
    "border": "#d9ded6",
    "text": "#23373a",
    "text_muted": "#596b69",
    "accent": "#176c64",
    "compare": "#b9573d",
    "good": "#176c64",
    "bad": "#a23e36",
}


def inject_theme():
    st.markdown(f"<style>{THEME_CSS}</style>", unsafe_allow_html=True)
    with st.sidebar:
        st.markdown(
            '<div class="cg-brand">ECG Research Project<span style="color:#176c64">.</span></div>'
            '<div class="cg-brand-sub">ECG Classification &amp; Adversarial Robustness</div>',
            unsafe_allow_html=True,
        )
        for path, label in [
            ("main.py", "Overview"),
            ("pages/1_ECG_Explorer.py", "01  /  ECG explorer"),
            ("pages/2_Adversarial_Lab.py", "02  /  Adversarial lab"),
            ("pages/3_Robustness_Benchmark.py", "03  /  Benchmarks"),
            ("pages/4_Signal_Analysis.py", "04  /  Signal analysis"),
            ("pages/5_Model_Explanation.py", "05  /  Attribution"),
            ("pages/6_Experiment_History.py", "06  /  Run archive"),
            ("pages/7_Study_Evidence.py", "07  /  Study evidence"),
        ]:
            st.page_link(path, label=label)
        st.divider()


def page_header(kicker, title, description):
    st.markdown(
        f'<div class="cg-eyebrow">{escape(kicker)}</div>', unsafe_allow_html=True
    )
    st.title(title)
    st.caption(description)


def data_source(default="demo"):
    current = st.session_state.get("cg_data_source", default)

    def remember():
        st.session_state["cg_data_source"] = st.session_state["_cg_dataset_widget"]

    st.session_state["_cg_dataset_widget"] = current
    mode = st.radio(
        "Dataset",
        ["demo", "research"],
        key="_cg_dataset_widget",
        on_change=remember,
        format_func=lambda v: "Synthetic demo" if v == "demo" else "MIT-BIH research",
    )
    st.session_state["cg_data_source"] = mode
    label = (
        "subject-disjoint v2"
        if (ARTIFACTS_DIR / "active_study.json").exists()
        else "original record split"
    )
    st.caption(
        "Synthetic fixture · pipeline demonstration"
        if mode == "demo"
        else f"MIT-BIH · 360 Hz · {label}"
    )
    return mode


def research_note(mode):
    if mode == "research":
        if (ARTIFACTS_DIR / "active_study.json").exists():
            st.caption(
                "V2 · subject-disjoint split · record 201 excluded. Past-RR features use expert peak annotations and remain fixed during waveform attacks."
            )
        else:
            st.caption(
                "Historical record split: validation 201 and test 202 share a subject."
            )


def disclaimer():
    st.markdown(
        '<div class="cg-disclaimer">Research prototype<br>For signal and model evaluation. '
        "Not for clinical diagnosis.</div>",
        unsafe_allow_html=True,
    )
