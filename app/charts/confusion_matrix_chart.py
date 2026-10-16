import plotly.graph_objects as go

from app.styles.theme import COLORS


def confusion_matrix_figure(cm: list, labels=("Normal", "Arrhythmia")) -> go.Figure:
    z = cm
    z_text = [[str(v) for v in row] for row in z]

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=list(labels),
            y=list(labels),
            text=z_text,
            texttemplate="%{text}",
            colorscale=[[0, COLORS["surface"]], [1, COLORS["accent"]]],
            showscale=False,
        )
    )
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor=COLORS["surface"],
        plot_bgcolor=COLORS["surface"],
        font=dict(color=COLORS["text"], size=12),
        margin=dict(l=60, r=20, t=10, b=40),
        xaxis=dict(title="predicted", side="bottom"),
        yaxis=dict(title="actual", autorange="reversed"),
        height=280,
    )
    return fig
