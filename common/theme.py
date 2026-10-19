"""Shared Plotly look. Blue (low) to red (high) colour scale used for stress/pressure maps."""
from __future__ import annotations

import plotly.graph_objects as go

# Low (blue) -> mid (green/yellow) -> limit (red)
STRESS_COLORSCALE = [
    [0.0, "#2c7bb6"], [0.25, "#abd9e9"], [0.5, "#ffffbf"], [0.75, "#fdae61"], [1.0, "#d7191c"],
]
PASS_COLOR = "#2e7d32"
FAIL_COLOR = "#c62828"
ACCENT = "#1f4e79"

_FONT = "Inter, Segoe UI, Arial, sans-serif"


def apply_theme(fig: go.Figure, title: str | None = None, height: int = 400) -> go.Figure:
    """Apply the shared layout to a Plotly figure and return it."""
    fig.update_layout(
        template="plotly_white",
        font=dict(family=_FONT, size=13),
        title=dict(text=title, x=0.01) if title else None,
        height=height,
        margin=dict(l=50, r=20, t=50 if title else 20, b=45),
        legend=dict(orientation="h", y=-0.2),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#e8e8e8", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#e8e8e8", zeroline=False)
    return fig
