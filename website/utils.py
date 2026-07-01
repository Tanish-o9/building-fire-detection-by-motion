"""website/utils.py — Shared helpers: CSS loader, HTML components"""

import streamlit as st
from pathlib import Path


# ── CSS Loader ─────────────────────────────────────────────────────────────
def load_css():
    css_path = Path(__file__).parent / "styles" / "theme.css"
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# ── Reusable HTML Components ───────────────────────────────────────────────
def hero(title: str, subtitle: str, badge: str = "🔥 AI-POWERED SYSTEM"):
    st.markdown(f"""
    <div class="hero-container fire-bg">
        <div class="hero-badge">{badge}</div>
        <h1 class="hero-title">{title}</h1>
        <p class="hero-subtitle">{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)


def section_header(tag: str, title: str, desc: str = ""):
    st.markdown(f"""
    <div class="section-header">
        <div class="section-tag">{tag}</div>
        <h2 class="section-title">{title}</h2>
        {"<p class='section-desc'>" + desc + "</p>" if desc else ""}
    </div>
    """, unsafe_allow_html=True)


def stat_cards(stats: list[dict]):
    """stats = [{"number": "98%", "label": "Accuracy"}, ...]"""
    cards_html = "".join(f"""
        <div class="stat-card">
            <span class="stat-number">{s['number']}</span>
            <span class="stat-label">{s['label']}</span>
        </div>""" for s in stats)
    st.markdown(f'<div class="stats-grid">{cards_html}</div>', unsafe_allow_html=True)


def feature_cards(features: list[dict]):
    """features = [{"icon": "🎯", "title": "...", "desc": "..."}, ...]"""
    cards_html = "".join(f"""
        <div class="feature-card">
            <span class="feature-icon">{f['icon']}</span>
            <div class="feature-title">{f['title']}</div>
            <div class="feature-desc">{f['desc']}</div>
        </div>""" for f in features)
    st.markdown(f'<div class="features-grid">{cards_html}</div>', unsafe_allow_html=True)


def metric_row(metrics: list[dict]):
    """metrics = [{"value": "0.94", "name": "Precision"}, ...]"""
    cards_html = "".join(f"""
        <div class="metric-card">
            <span class="metric-value">{m['value']}</span>
            <div class="metric-name">{m['name']}</div>
        </div>""" for m in metrics)
    st.markdown(f'<div class="metric-row">{cards_html}</div>', unsafe_allow_html=True)


def alert_banner(icon: str, text: str):
    st.markdown(f"""
    <div class="alert-banner">
        <span class="alert-icon">{icon}</span>
        <span class="alert-text">{text}</span>
    </div>
    """, unsafe_allow_html=True)


def fire_divider():
    st.markdown('<div class="fire-divider"></div>', unsafe_allow_html=True)


def detection_badge(count: int):
    if count == 0:
        cls, icon = "badge-safe", "✅"
        label = "No Persons Detected — Area Clear"
    elif count <= 2:
        cls, icon = "badge-warning", "⚠️"
        label = f"{count} Person{'s' if count > 1 else ''} Detected"
    else:
        cls, icon = "badge-danger", "🚨"
        label = f"CRITICAL — {count} Persons Detected"
    st.markdown(f'<div class="detection-badge {cls}">{icon} {label}</div>',
                unsafe_allow_html=True)


def zone_grid(active_zones: list[str]):
    zone_names = [
        ["Top-Left", "Top-Center", "Top-Right"],
        ["Mid-Left", "Center",     "Mid-Right"],
        ["Bottom-Left", "Bottom-Center", "Bottom-Right"],
    ]
    rows_html = ""
    for row in zone_names:
        for zone in row:
            cls = "zone-cell active" if zone in active_zones else "zone-cell"
            rows_html += f'<div class="{cls}">{zone}</div>'
    st.markdown(f'<div class="zone-grid">{rows_html}</div>', unsafe_allow_html=True)


def arch_flow(nodes: list[dict]):
    """nodes = [{"icon": "📷", "label": "Input"}, ...]"""
    items = ""
    for i, n in enumerate(nodes):
        items += f"""<div class="arch-node">
            <span class="arch-node-icon">{n['icon']}</span>
            <div class="arch-node-label">{n['label']}</div>
        </div>"""
        if i < len(nodes) - 1:
            items += '<span class="arch-arrow">→</span>'
    st.markdown(f'<div class="arch-flow">{items}</div>', unsafe_allow_html=True)


def sidebar_nav():
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center; padding: 20px 0 10px;">
            <div style="font-size:2.5rem;">🔥</div>
            <div style="font-size:1rem; font-weight:800; color:#FF4B4B; letter-spacing:-0.5px;">
                FireDetect AI
            </div>
            <div style="font-size:0.72rem; color:#8888A0; margin-top:4px;">
                Emergency Human Detection
            </div>
        </div>
        <hr style="border-color:rgba(255,255,255,0.08); margin:16px 0;">
        """, unsafe_allow_html=True)

        page = st.radio(
            "Navigation",
            ["🏠  Home", "🎯  Live Detection", "📊  Dashboard", "🧠  How It Works"],
            label_visibility="collapsed"
        )

        st.markdown("""
        <hr style="border-color:rgba(255,255,255,0.08); margin:16px 0;">
        <div style="font-size:0.72rem; color:#8888A0; text-align:center; padding-bottom:10px;">
            Powered by YOLOv8 + OpenCV<br>
            <span style="color:#FF4B4B;">●</span> System Online
        </div>
        """, unsafe_allow_html=True)

        return page
