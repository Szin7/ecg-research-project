from html import escape

import streamlit as st


def metric_row(items: list, help_texts: dict | None = None) -> None:
    """items: list of (label, value) pairs rendered as a row of metric cards."""
    cols = st.columns(len(items))
    for col, (label, value) in zip(cols, items):
        col.metric(label, value, help=(help_texts or {}).get(label))


def not_benchmarked(message: str = "Not benchmarked yet") -> None:
    st.markdown(
        f'<div class="cg-not-benchmarked">{escape(message)} — run the corresponding script in scripts/.</div>',
        unsafe_allow_html=True,
    )


def section_label(text: str) -> None:
    st.markdown(
        f'<div class="cg-section-label">{escape(text)}</div>', unsafe_allow_html=True
    )
