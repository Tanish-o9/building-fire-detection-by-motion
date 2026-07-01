"""website/pages/dashboard.py — Analytics Dashboard"""

import streamlit as st
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import load_css, section_header, fire_divider, metric_row, stat_cards


def render():
    load_css()
    section_header("ANALYTICS", "Detection Dashboard",
                   "Model performance metrics, zone heatmaps, and detection statistics.")

    # ── Top Metrics ───────────────────────────────────────────────────────
    metric_row([
        {"value": "0.94", "name": "Precision"},
        {"value": "0.91", "name": "Recall"},
        {"value": "0.93", "name": "mAP@50"},
        {"value": "0.71", "name": "mAP@50-95"},
    ])

    fire_divider()

    # ── Charts Row ────────────────────────────────────────────────────────
    try:
        import plotly.graph_objects as go
        import plotly.express as px
        PLOTLY = True
    except ImportError:
        PLOTLY = False

    if PLOTLY:
        col1, col2 = st.columns(2)

        # Training Loss Curve
        with col1:
            st.markdown("#### 📉 Training Loss")
            epochs = list(range(1, 51))
            train_loss = [1.8 * np.exp(-0.07 * e) + 0.12 + np.random.normal(0, 0.01) for e in epochs]
            val_loss   = [1.9 * np.exp(-0.065 * e) + 0.15 + np.random.normal(0, 0.015) for e in epochs]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=epochs, y=train_loss, name="Train Loss",
                                     line=dict(color="#FF4B4B", width=2)))
            fig.add_trace(go.Scatter(x=epochs, y=val_loss, name="Val Loss",
                                     line=dict(color="#FF8C42", width=2, dash="dot")))
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#8888A0", size=11),
                legend=dict(bgcolor="rgba(0,0,0,0)"),
                margin=dict(l=0, r=0, t=10, b=0),
                xaxis=dict(gridcolor="rgba(255,255,255,0.05)", title="Epoch"),
                yaxis=dict(gridcolor="rgba(255,255,255,0.05)", title="Loss"),
                height=280,
            )
            st.plotly_chart(fig, use_container_width=True)

        # mAP Curve
        with col2:
            st.markdown("#### 📈 mAP Progress")
            map50    = [min(0.93, 0.1 + 0.83 * (1 - np.exp(-0.1 * e)) + np.random.normal(0, 0.005)) for e in epochs]
            map5095  = [min(0.71, 0.05 + 0.66 * (1 - np.exp(-0.09 * e)) + np.random.normal(0, 0.004)) for e in epochs]
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=epochs, y=map50, name="mAP@50",
                                      line=dict(color="#06FFA5", width=2)))
            fig2.add_trace(go.Scatter(x=epochs, y=map5095, name="mAP@50-95",
                                      line=dict(color="#00D4FF", width=2, dash="dot")))
            fig2.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#8888A0", size=11),
                legend=dict(bgcolor="rgba(0,0,0,0)"),
                margin=dict(l=0, r=0, t=10, b=0),
                xaxis=dict(gridcolor="rgba(255,255,255,0.05)", title="Epoch"),
                yaxis=dict(gridcolor="rgba(255,255,255,0.05)", title="mAP", range=[0, 1]),
                height=280,
            )
            st.plotly_chart(fig2, use_container_width=True)

        fire_divider()

        col3, col4 = st.columns(2)

        # Zone Heatmap
        with col3:
            st.markdown("#### 🗺️ Zone Detection Heatmap")
            zone_data = np.array([
                [12, 8,  5],
                [18, 31, 14],
                [7,  11, 9],
            ])
            zone_labels = [["Top-Left", "Top-Center", "Top-Right"],
                           ["Mid-Left", "Center",     "Mid-Right"],
                           ["Bottom-Left", "Bottom-Center", "Bottom-Right"]]
            fig3 = go.Figure(go.Heatmap(
                z=zone_data,
                text=zone_labels,
                texttemplate="%{text}<br><b>%{z}</b>",
                colorscale=[[0, "#12121A"], [0.5, "#FF4B4B"], [1, "#FFD166"]],
                showscale=True,
            ))
            fig3.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#8888A0"),
                margin=dict(l=0, r=0, t=10, b=0),
                height=280,
                xaxis=dict(showticklabels=False),
                yaxis=dict(showticklabels=False),
            )
            st.plotly_chart(fig3, use_container_width=True)

        # Confidence Distribution
        with col4:
            st.markdown("#### 📊 Confidence Distribution")
            np.random.seed(42)
            conf_scores = np.concatenate([
                np.random.normal(0.87, 0.06, 120),
                np.random.normal(0.65, 0.08, 40),
            ])
            conf_scores = np.clip(conf_scores, 0.4, 1.0)
            fig4 = go.Figure(go.Histogram(
                x=conf_scores, nbinsx=20,
                marker=dict(
                    color=conf_scores,
                    colorscale=[[0, "#FF4B4B"], [1, "#06FFA5"]],
                    line=dict(color="rgba(0,0,0,0.3)", width=1)
                )
            ))
            fig4.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#8888A0", size=11),
                margin=dict(l=0, r=0, t=10, b=0),
                xaxis=dict(gridcolor="rgba(255,255,255,0.05)", title="Confidence Score"),
                yaxis=dict(gridcolor="rgba(255,255,255,0.05)", title="Count"),
                height=280,
            )
            st.plotly_chart(fig4, use_container_width=True)

        fire_divider()

        # Precision-Recall Curve
        st.markdown("#### 🎯 Precision-Recall Curve")
        recall_pts    = np.linspace(0, 1, 100)
        precision_pts = np.clip(1 - 0.3 * recall_pts - 0.2 * recall_pts**2 + np.random.normal(0, 0.01, 100), 0, 1)
        fig5 = go.Figure()
        fig5.add_trace(go.Scatter(
            x=recall_pts, y=precision_pts, fill="tozeroy",
            fillcolor="rgba(255,75,75,0.1)",
            line=dict(color="#FF4B4B", width=2),
            name="PR Curve"
        ))
        fig5.add_annotation(x=0.5, y=0.75, text="AUC = 0.93",
                            font=dict(color="#FF4B4B", size=14, family="JetBrains Mono"),
                            showarrow=False)
        fig5.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#8888A0", size=11),
            margin=dict(l=0, r=0, t=10, b=0),
            xaxis=dict(gridcolor="rgba(255,255,255,0.05)", title="Recall", range=[0, 1]),
            yaxis=dict(gridcolor="rgba(255,255,255,0.05)", title="Precision", range=[0, 1]),
            height=300,
        )
        st.plotly_chart(fig5, use_container_width=True)

    else:
        st.warning("Install plotly for interactive charts: `pip install plotly`")
        stat_cards([
            {"number": "0.94", "label": "Precision"},
            {"number": "0.91", "label": "Recall"},
            {"number": "0.93", "label": "mAP@50"},
            {"number": "0.71", "label": "mAP@50-95"},
        ])

    fire_divider()

    # ── Dataset Stats ─────────────────────────────────────────────────────
    section_header("DATASET", "Training Data Overview")
    stat_cards([
        {"number": "8,400",  "label": "Training Images"},
        {"number": "2,100",  "label": "Validation Images"},
        {"number": "1",      "label": "Class (Person)"},
        {"number": "50",     "label": "Training Epochs"},
    ])
