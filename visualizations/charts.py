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


# consistent color scheme across all charts
COLORS = {
    "primary": "#6366f1",       # indigo
    "secondary": "#8b5cf6",     # violet
    "success": "#22c55e",       # green
    "warning": "#f59e0b",       # amber
    "danger": "#ef4444",        # red
    "text": "#e2e8f0",          # light gray
    "bg": "#0f172a",            # dark navy
    "card_bg": "#1e293b",       # slightly lighter navy
    "grid": "#334155",          # grid lines
}


def create_match_gauge(score, label="Overall Match Score"):
    """
    Create a gauge/speedometer chart showing the overall match score.
    The color transitions from red (low) to amber (medium) to green (high).
    """
    # determine color based on score
    if score >= 70:
        bar_color = COLORS["success"]
    elif score >= 45:
        bar_color = COLORS["warning"]
    else:
        bar_color = COLORS["danger"]
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score,
        number={"suffix": "%", "font": {"size": 48, "color": COLORS["text"]}},
        title={"text": label, "font": {"size": 16, "color": COLORS["text"]}},
        gauge={
            "axis": {
                "range": [0, 100],
                "tickwidth": 1,
                "tickcolor": COLORS["grid"],
                "tickfont": {"color": COLORS["text"]},
            },
            "bar": {"color": bar_color, "thickness": 0.75},
            "bgcolor": COLORS["card_bg"],
            "borderwidth": 0,
            "steps": [
                {"range": [0, 45], "color": "rgba(239,68,68,0.15)"},
                {"range": [45, 70], "color": "rgba(245,158,11,0.15)"},
                {"range": [70, 100], "color": "rgba(34,197,94,0.15)"},
            ],
            "threshold": {
                "line": {"color": COLORS["text"], "width": 2},
                "thickness": 0.8,
                "value": score,
            },
        },
    ))
    
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": COLORS["text"]},
        height=280,
        margin=dict(t=50, b=20, l=30, r=30),
    )
    
    return fig


def create_score_breakdown(components):
    """
    Horizontal bar chart showing how each component contributes
    to the overall score.
    
    components: dict like {"TF-IDF Similarity": 65.2, "Skill Match": 72.0, ...}
    """
    names = list(components.keys())
    values = list(components.values())
    
    # assign colors based on value
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
        marker_color=colors,
        text=[f"{v:.1f}%" for v in values],
        textposition="auto",
        textfont={"color": COLORS["text"], "size": 13},
    ))
    
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": COLORS["text"]},
        xaxis={
            "range": [0, 100],
            "title": "Score",
            "gridcolor": COLORS["grid"],
            "zerolinecolor": COLORS["grid"],
        },
        yaxis={"gridcolor": COLORS["grid"]},
        height=300,
        margin=dict(t=20, b=40, l=120, r=20),
        showlegend=False,
    )
    
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
            showarrow=False, font={"size": 16, "color": COLORS["text"]}
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=200,
        )
        return fig
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=all_skills,
        x=[1] * len(all_skills),
        orientation='h',
        marker_color=colors,
        text=statuses,
        textposition="inside",
        textfont={"color": "white", "size": 12},
        hoverinfo="y+text",
    ))
    
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": COLORS["text"]},
        xaxis={"visible": False},
        yaxis={"gridcolor": COLORS["grid"], "autorange": "reversed"},
        height=max(250, len(all_skills) * 30),
        margin=dict(t=10, b=10, l=140, r=20),
        showlegend=False,
    )
    
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
    
    fig.add_trace(go.Scatterpolar(
        r=coverage + [coverage[0]],  # close the polygon
        theta=categories + [categories[0]],
        fill='toself',
        fillcolor="rgba(99,102,241,0.2)",
        line={"color": COLORS["primary"], "width": 2},
        name="Skill Coverage",
    ))
    
    fig.update_layout(
        polar={
            "radialaxis": {
                "visible": True,
                "range": [0, 100],
                "gridcolor": COLORS["grid"],
                "tickfont": {"color": COLORS["text"]},
            },
            "angularaxis": {
                "gridcolor": COLORS["grid"],
                "tickfont": {"color": COLORS["text"], "size": 10},
            },
            "bgcolor": "rgba(0,0,0,0)",
        },
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": COLORS["text"]},
        height=380,
        margin=dict(t=30, b=30, l=60, r=60),
        showlegend=False,
    )
    
    return fig


def create_confusion_matrix_chart(cm_data, labels):
    """
    Heatmap visualization of the confusion matrix from model training.
    """
    cm = np.array(cm_data)
    
    # normalize to percentages for display
    cm_pct = (cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100)
    
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
        colorscale=[[0, COLORS["bg"]], [1, COLORS["primary"]]],
        text=text_matrix,
        texttemplate="%{text}",
        textfont={"size": 12, "color": COLORS["text"]},
        hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
        showscale=False,
    ))
    
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": COLORS["text"]},
        xaxis={"title": "Predicted", "side": "bottom"},
        yaxis={"title": "Actual", "autorange": "reversed"},
        height=350,
        margin=dict(t=20, b=60, l=80, r=20),
    )
    
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
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=names,
        x=values,
        orientation='h',
        marker={
            "color": values,
            "colorscale": [[0, COLORS["secondary"]], [1, COLORS["primary"]]],
        },
        text=[f"{v:.4f}" for v in values],
        textposition="auto",
        textfont={"color": COLORS["text"]},
    ))
    
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": COLORS["text"]},
        xaxis={
            "title": "Importance",
            "gridcolor": COLORS["grid"],
            "zerolinecolor": COLORS["grid"],
        },
        yaxis={"autorange": "reversed", "gridcolor": COLORS["grid"]},
        height=280,
        margin=dict(t=10, b=40, l=160, r=20),
        showlegend=False,
    )
    
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
        hole=0.55,
        marker={"colors": colors, "line": {"color": COLORS["bg"], "width": 2}},
        textinfo="label+percent",
        textfont={"color": COLORS["text"], "size": 12},
        hovertemplate="%{label}: %{value:.2%}<extra></extra>",
    )])
    
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": COLORS["text"]},
        height=300,
        margin=dict(t=20, b=20, l=20, r=20),
        showlegend=True,
        legend={"font": {"color": COLORS["text"]}},
    )
    
    return fig
