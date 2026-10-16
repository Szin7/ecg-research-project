import numpy as np
import plotly.graph_objects as go

from app.styles.theme import COLORS

_LAYOUT = dict(
    template="plotly_white",
    paper_bgcolor=COLORS["surface"],
    plot_bgcolor=COLORS["surface"],
    font=dict(color=COLORS["text"], size=12),
    margin=dict(l=50, r=20, t=30, b=40),
)


def waveform_figure(
    signal: np.ndarray,
    fs: int = 360,
    title: str = "",
    r_peaks=None,
    amplitude_label="amplitude (z-score)",
) -> go.Figure:
    t = np.arange(len(signal)) / fs

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=t,
            y=signal,
            mode="lines",
            line=dict(color=COLORS["accent"], width=1.5),
            name="signal",
        )
    )

    if r_peaks is not None and len(r_peaks) > 0:
        r_peaks = np.asarray(r_peaks)
        r_peaks = r_peaks[r_peaks < len(signal)]
        fig.add_trace(
            go.Scatter(
                x=r_peaks / fs,
                y=signal[r_peaks],
                mode="markers",
                marker=dict(color=COLORS["compare"], size=8, symbol="x"),
                name="R-peak",
            )
        )

    fig.update_layout(
        **_LAYOUT,
        title=title,
        xaxis=dict(title="time (s)", gridcolor=COLORS["border"], zeroline=False),
        yaxis=dict(title=amplitude_label, gridcolor=COLORS["border"], zeroline=False),
        showlegend=r_peaks is not None,
        height=320,
    )
    return fig


def overlay_figure(
    original: np.ndarray, adversarial: np.ndarray, fs: int = 360
) -> go.Figure:
    """Original vs. adversarial waveform, overlaid -- the centerpiece of
    the Adversarial Lab."""
    t = np.arange(len(original)) / fs

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=t,
            y=original,
            mode="lines",
            line=dict(color=COLORS["accent"], width=1.5),
            name="original",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=t,
            y=adversarial,
            mode="lines",
            line=dict(color=COLORS["compare"], width=1.5, dash="dot"),
            name="adversarial",
        )
    )

    fig.update_layout(
        **_LAYOUT,
        xaxis=dict(title="time (s)", gridcolor=COLORS["border"], zeroline=False),
        yaxis=dict(
            title="amplitude (z-score)", gridcolor=COLORS["border"], zeroline=False
        ),
        legend=dict(orientation="h", y=1.15),
        height=300,
    )
    return fig


def perturbation_figure(
    original: np.ndarray, adversarial: np.ndarray, fs: int = 360
) -> go.Figure:
    """The perturbation itself, on its own scale so a small constrained
    change is actually visible rather than lost against the waveform."""
    t = np.arange(len(original)) / fs
    delta = adversarial - original

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=t,
            y=delta,
            mode="lines",
            line=dict(color=COLORS["bad"], width=1.2),
            fill="tozeroy",
            name="perturbation",
        )
    )

    fig.update_layout(
        **_LAYOUT,
        xaxis=dict(title="time (s)", gridcolor=COLORS["border"], zeroline=False),
        yaxis=dict(
            title="Δ amplitude",
            gridcolor=COLORS["border"],
            zeroline=True,
            zerolinecolor=COLORS["border"],
        ),
        height=180,
    )
    return fig


def attribution_figure(
    signal: np.ndarray, attribution: np.ndarray, fs: int = 360, title: str = ""
) -> go.Figure:
    """Waveform with attribution magnitude shaded underneath -- darker/
    taller shading marks the regions that mattered most to the prediction."""
    t = np.arange(len(signal)) / fs
    weight = np.abs(attribution)
    weight = weight / (weight.max() + 1e-8)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=t,
            y=weight * np.abs(signal).max(),
            marker=dict(color=COLORS["compare"], opacity=0.35),
            name="attribution",
            width=1 / fs,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=t,
            y=signal,
            mode="lines",
            line=dict(color=COLORS["accent"], width=1.5),
            name="signal",
        )
    )

    fig.update_layout(
        **_LAYOUT,
        title=title,
        xaxis=dict(title="time (s)", gridcolor=COLORS["border"], zeroline=False),
        yaxis=dict(
            title="amplitude (z-score)", gridcolor=COLORS["border"], zeroline=False
        ),
        showlegend=False,
        height=280,
        barmode="overlay",
    )
    return fig
