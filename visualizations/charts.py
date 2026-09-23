"""
charts.py
---------
Plotly visualization helpers for the Streamlit dashboard.

All the chart-building logic is kept here to keep app.py cleaner.
Each function returns a Plotly figure object that can be passed
directly to st.plotly_chart().
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np


# consistent color scheme across all charts — premium dark palette
COLORS = {
    "primary": "#6366f1",       # indigo
    "secondary": "#8b5cf6",     # violet
    "tertiary": "#a78bfa",      # light violet
    "cyan": "#06b6d4",          # cyan accent
    "success": "#10b981",       # emerald
    "warning": "#f59e0b",       # amber
    "danger": "#ef4444",        # red
    "text": "#f1f5f9",          # near white
    "text_secondary": "#94a3b8",# slate
    "text_muted": "#64748b",    # muted
    "bg": "#0a0e1a",            # deep dark navy
    "card_bg": "#111827",       # card background
    "grid": "rgba(99,102,241,0.1)", # subtle grid
    "glow_indigo": "rgba(99,102,241,0.3)",
    "glow_violet": "rgba(139,92,246,0.2)",
}

FONT_FAMILY = "Inter, -apple-system, BlinkMacSystemFont, sans-serif"


def _base_layout(height=300, **kwargs):
    """Return a base layout dict for consistent chart styling."""
    layout = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT_FAMILY, color=COLORS["text"]),
        height=height,
        margin=dict(t=20, b=40, l=20, r=20),
        showlegend=False,
    )
    layout.update(kwargs)
    return layout


def create_match_gauge(score, label="Overall Match Score"):
    """
    Create a gauge/speedometer chart showing the overall match score.
    The color transitions from red (low) to amber (medium) to green (high).
    """
    # determine color based on score
    if score >= 70:
        bar_color = COLORS["success"]
        glow = "rgba(16,185,129,0.3)"
    elif score >= 45:
        bar_color = COLORS["warning"]
        glow = "rgba(245,158,11,0.3)"
    else:
        bar_color = COLORS["danger"]
        glow = "rgba(239,68,68,0.3)"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={
            "suffix": "%",
            "font": {"size": 42, "color": COLORS["text"], "family": FONT_FAMILY},
        },
        title={
            "text": label,
            "font": {"size": 14, "color": COLORS["text_secondary"], "family": FONT_FAMILY},
        },
        gauge={
            "axis": {
                "range": [0, 100],
                "tickwidth": 1,
                "tickcolor": COLORS["grid"],
                "tickfont": {"color": COLORS["text_muted"], "size": 10},
                "dtick": 25,
            },
            "bar": {"color": bar_color, "thickness": 0.8},
            "bgcolor": COLORS["card_bg"],
            "borderwidth": 0,
            "steps": [
                {"range": [0, 45], "color": "rgba(239,68,68,0.08)"},
                {"range": [45, 70], "color": "rgba(245,158,11,0.08)"},
                {"range": [70, 100], "color": "rgba(16,185,129,0.08)"},
            ],
            "threshold": {
                "line": {"color": COLORS["text"], "width": 2},
                "thickness": 0.85,
                "value": score,
            },
        },
    ))

    fig.update_layout(**_base_layout(
        height=300,
        margin=dict(t=60, b=20, l=30, r=30),
    ))

    return fig


def create_score_breakdown(components):
    """
    Horizontal bar chart showing how each component contributes
    to the overall score.

    components: dict like {"TF-IDF Similarity": 65.2, "Skill Match": 72.0, ...}
    """
    names = list(components.keys())
    values = list(components.values())

    # assign gradient colors based on value
    colors = []
    for v in values:
        if v >= 70:
            colors.append(COLORS["success"])
        elif v >= 45:
            colors.append(COLORS["warning"])
        else:
            colors.append(COLORS["danger"])

    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=names,
        x=values,
        orientation='h',
        marker={
            "color": colors,
            "line": {"width": 0},
            "opacity": 0.9,
        },
        text=[f"{v:.1f}%" for v in values],
        textposition="auto",
        textfont={"color": "white", "size": 13, "family": FONT_FAMILY},
        hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
    ))

    fig.update_layout(**_base_layout(
        height=280,
        margin=dict(t=10, b=30, l=140, r=30),
        xaxis=dict(
            range=[0, 100],
            title="Score",
            gridcolor=COLORS["grid"],
            zerolinecolor=COLORS["grid"],
            title_font=dict(size=12, color=COLORS["text_muted"]),
        ),
        yaxis=dict(gridcolor=COLORS["grid"]),
    ))

    return fig


def create_skill_comparison(matched, missing):
    """
    Horizontal grouped bar chart showing matched vs missing skills.
    Each skill gets a green bar (matched) or red bar (missing).
    """
    all_skills = matched + missing
    statuses = ["Matched"] * len(matched) + ["Missing"] * len(missing)
    colors = [COLORS["success"]] * len(matched) + [COLORS["danger"]] * len(missing)

    if not all_skills:
        # return an empty figure with a message
        fig = go.Figure()
        fig.add_annotation(
            text="No skills to compare",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font={"size": 16, "color": COLORS["text_muted"]}
        )
        fig.update_layout(**_base_layout(height=200))
        return fig

    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=all_skills,
        x=[1] * len(all_skills),
        orientation='h',
        marker={
            "color": colors,
            "opacity": 0.85,
            "line": {"width": 0},
        },
        text=statuses,
        textposition="inside",
        textfont={"color": "white", "size": 12, "family": FONT_FAMILY},
        hoverinfo="y+text",
    ))

    fig.update_layout(**_base_layout(
        height=max(250, len(all_skills) * 32),
        margin=dict(t=10, b=10, l=160, r=20),
        xaxis={"visible": False},
        yaxis={"gridcolor": COLORS["grid"], "autorange": "reversed"},
    ))

    return fig


def create_skill_radar(resume_skills_by_category, jd_skills_by_category, categories=None):
    """
    Radar chart comparing skill coverage across categories.
    Shows how well the resume covers each skill category
    relative to what the JD requires.
    """
    if categories is None:
        from config import SKILL_CATEGORIES
        categories = SKILL_CATEGORIES

    resume_counts = []
    jd_counts = []

    for cat in categories:
        resume_counts.append(len(resume_skills_by_category.get(cat, [])))
        jd_counts.append(len(jd_skills_by_category.get(cat, [])))

    # normalize: for each category, compute coverage percentage
    coverage = []
    for r, j in zip(resume_counts, jd_counts):
        if j > 0:
            coverage.append(min(100, (r / j) * 100))
        elif r > 0:
            coverage.append(100)
        else:
            coverage.append(0)

    fig = go.Figure()

    # background reference ring at 100%
    fig.add_trace(go.Scatterpolar(
        r=[100] * (len(categories) + 1),
        theta=categories + [categories[0]],
        fill=None,
        line={"color": "rgba(99,102,241,0.1)", "width": 1, "dash": "dot"},
        name="Target",
        showlegend=False,
    ))

    # actual coverage
    fig.add_trace(go.Scatterpolar(
        r=coverage + [coverage[0]],  # close the polygon
        theta=categories + [categories[0]],
        fill='toself',
        fillcolor="rgba(99,102,241,0.15)",
        line={"color": COLORS["primary"], "width": 2.5},
        marker={"size": 6, "color": COLORS["primary"]},
        name="Skill Coverage",
    ))

    fig.update_layout(
        polar={
            "radialaxis": {
                "visible": True,
                "range": [0, 100],
                "gridcolor": COLORS["grid"],
                "tickfont": {"color": COLORS["text_muted"], "size": 9},
                "ticksuffix": "%",
            },
            "angularaxis": {
                "gridcolor": COLORS["grid"],
                "tickfont": {"color": COLORS["text_secondary"], "size": 10, "family": FONT_FAMILY},
                "linecolor": COLORS["grid"],
            },
            "bgcolor": "rgba(0,0,0,0)",
        },
        **_base_layout(
            height=400,
            margin=dict(t=40, b=40, l=80, r=80),
        ),
    )

    return fig


def create_confusion_matrix_chart(cm_data, labels):
    """
    Heatmap visualization of the confusion matrix from model training.
    """
    cm = np.array(cm_data)

    # normalize to percentages for display
    row_sums = cm.sum(axis=1)[:, np.newaxis]
    cm_pct = np.divide(cm.astype('float'), row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0) * 100

    # build text annotations
    annotations = []
    for i in range(len(labels)):
        for j in range(len(labels)):
            annotations.append(f"{cm[i][j]}\n({cm_pct[i][j]:.1f}%)")

    text_matrix = np.array(annotations).reshape(len(labels), len(labels))

    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=labels,
        y=labels,
        colorscale=[
            [0, COLORS["bg"]],
            [0.5, "rgba(99,102,241,0.5)"],
            [1, COLORS["primary"]],
        ],
        text=text_matrix,
        texttemplate="%{text}",
        textfont={"size": 12, "color": COLORS["text"], "family": FONT_FAMILY},
        hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
        showscale=False,
    ))

    fig.update_layout(**_base_layout(
        height=350,
        margin=dict(t=20, b=60, l=100, r=20),
        xaxis={"title": "Predicted", "side": "bottom"},
        yaxis={"title": "Actual", "autorange": "reversed"},
    ))

    return fig


def create_feature_importance_chart(feature_importances):
    """
    Bar chart showing feature importance from the Random Forest model.

    feature_importances: list of (feature_name, importance_score) tuples
    """
    if not feature_importances:
        return go.Figure()

    names = [f[0] for f in feature_importances]
    values = [f[1] for f in feature_importances]

    # create a smooth gradient color based on position
    n = len(values)
    colors = [
        f"rgba({99 + int((139-99)*i/(n-1))}, {102 + int((92-102)*i/(n-1))}, {241 + int((246-241)*i/(n-1))}, 0.85)"
        if n > 1 else COLORS["primary"]
        for i in range(n)
    ]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=names,
        x=values,
        orientation='h',
        marker={
            "color": colors,
            "line": {"width": 0},
        },
        text=[f"{v:.4f}" for v in values],
        textposition="auto",
        textfont={"color": "white", "family": FONT_FAMILY},
        hovertemplate="%{y}: %{x:.4f}<extra></extra>",
    ))

    fig.update_layout(**_base_layout(
        height=280,
        margin=dict(t=10, b=40, l=180, r=30),
        xaxis=dict(
            title="Importance",
            gridcolor=COLORS["grid"],
            zerolinecolor=COLORS["grid"],
            title_font=dict(size=12, color=COLORS["text_muted"]),
        ),
        yaxis={"autorange": "reversed", "gridcolor": COLORS["grid"]},
    ))

    return fig


def create_probability_chart(probabilities):
    """
    Donut chart showing classification probabilities.

    probabilities: dict like {"Strong Match": 0.75, "Moderate Match": 0.20, ...}
    """
    labels = list(probabilities.keys())
    values = list(probabilities.values())

    color_map = {
        "Strong Match": COLORS["success"],
        "Moderate Match": COLORS["warning"],
        "Weak Match": COLORS["danger"],
    }
    colors = [color_map.get(l, COLORS["primary"]) for l in labels]

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.6,
        marker={
            "colors": colors,
            "line": {"color": COLORS["bg"], "width": 3},
        },
        textinfo="label+percent",
        textfont={"color": COLORS["text"], "size": 11, "family": FONT_FAMILY},
        hovertemplate="%{label}: %{value:.2%}<extra></extra>",
        sort=False,
    )])

    fig.update_layout(**_base_layout(
        height=280,
        margin=dict(t=10, b=10, l=10, r=10),
        showlegend=True,
        legend=dict(
            font=dict(color=COLORS["text_secondary"], size=10, family=FONT_FAMILY),
            orientation="h",
            yanchor="bottom",
            y=-0.1,
            xanchor="center",
            x=0.5,
        ),
    ))

    return fig
