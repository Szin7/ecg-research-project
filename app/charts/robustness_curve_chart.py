import plotly.graph_objects as go

from app.styles.theme import COLORS

_TAG_STYLE = {
    "v2_context": (COLORS["accent"], "Context CNN"),
    "v2_context_defended": (COLORS["compare"], "Defended context CNN"),
    "v2_morphology": (COLORS["text_muted"], "Waveform CNN"),
    "baseline_cnn": (COLORS["accent"], "Standard CNN"),
    "defended_cnn": (COLORS["compare"], "Defended CNN"),
    "demo_baseline_cnn": (COLORS["accent"], "Standard CNN"),
    "demo_defended_cnn": (COLORS["compare"], "Defended CNN"),
}

_LAYOUT = dict(
    template="plotly_white",
    paper_bgcolor=COLORS["surface"],
    plot_bgcolor=COLORS["surface"],
    font=dict(color=COLORS["text"], size=12),
    margin=dict(l=55, r=20, t=15, b=45),
    legend=dict(orientation="h", y=1.15),
    height=340,
)


def robustness_curve_figure(df, attack: str, y_col: str, y_title: str) -> go.Figure:
    """One line per model tag present in df, for a single attack, epsilon
    on the x-axis."""
    fig = go.Figure()

    for tag in sorted(df["tag"].unique()):
        sub = df[(df["tag"] == tag) & (df["attack"] == attack)].sort_values("epsilon")
        color, label = _TAG_STYLE.get(tag, (COLORS["text_muted"], tag))
        fig.add_trace(
            go.Scatter(
                x=sub["epsilon"],
                y=sub[y_col],
                mode="lines+markers",
                line=dict(color=color, width=2),
                marker=dict(size=6),
                name=label,
            )
        )

    fig.update_layout(
        **_LAYOUT,
        xaxis=dict(title="epsilon (L∞, z-score units)", gridcolor=COLORS["border"]),
        yaxis=dict(
            title=y_title,
            gridcolor=COLORS["border"],
            range=[-0.02, 1.02],
            tickformat=".0%",
        ),
    )
    return fig
