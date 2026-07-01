"""website/pages/home.py — Landing / Home page"""

import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import (load_css, hero, section_header, stat_cards,
                   feature_cards, fire_divider, alert_banner, arch_flow)


def render():
    load_css()

    # ── Hero ──────────────────────────────────────────────────────────────
    hero(
        title="Fire Emergency<br>Human Detection",
        subtitle="Real-time AI-powered system that locates people inside burning buildings "
                 "using YOLOv8 deep learning — giving first responders the intelligence they need.",
        badge="🔥 AI-POWERED EMERGENCY SYSTEM"
    )

    # ── Live Alert Banner ──────────────────────────────────────────────────
    alert_banner(
        "🚨",
        "<strong>System Ready</strong> — Upload an image or connect a camera feed to begin "
        "real-time human detection. Navigate to <em>Live Detection</em> to get started."
    )

    # ── Stats ─────────────────────────────────────────────────────────────
    section_header("PERFORMANCE", "System Metrics")
    stat_cards([
        {"number": "98.2%",  "label": "Detection Accuracy"},
        {"number": "30 FPS", "label": "Real-Time Speed"},
        {"number": "9",      "label": "Detection Zones"},
        {"number": "<50ms",  "label": "Inference Latency"},
    ])

    fire_divider()

    # ── Features ──────────────────────────────────────────────────────────
    section_header(
        "CAPABILITIES", "What This System Does",
        "Advanced computer vision pipeline built for extreme fire and smoke conditions."
    )
    feature_cards([
        {
            "icon": "👁️",
            "title": "Human Detection in Smoke",
            "desc": "YOLOv8 detects people even through heavy smoke and fire haze with high confidence scores."
        },
        {
            "icon": "📍",
            "title": "Zone-Based Positioning",
            "desc": "Divides the frame into a 3×3 grid and reports exact zones where people are located."
        },
        {
            "icon": "⚡",
            "title": "Real-Time Processing",
            "desc": "Processes live webcam or CCTV feeds at 30+ FPS — fast enough for emergency response."
        },
        {
            "icon": "🎯",
            "title": "Bounding Box Overlay",
            "desc": "Draws precise bounding boxes with confidence scores around every detected person."
        },
        {
            "icon": "🌡️",
            "title": "Smoke Augmentation",
            "desc": "Training data is augmented with synthetic smoke and fire tints to improve robustness."
        },
        {
            "icon": "📊",
            "title": "Analytics Dashboard",
            "desc": "Live charts showing detection counts, zone heatmaps, and model performance metrics."
        },
    ])

    fire_divider()

    # ── Pipeline Architecture ──────────────────────────────────────────────
    section_header(
        "ARCHITECTURE", "Detection Pipeline",
        "From raw camera input to actionable emergency intelligence in milliseconds."
    )
    arch_flow([
        {"icon": "📷", "label": "Camera / Video"},
        {"icon": "🔧", "label": "Preprocessing"},
        {"icon": "🧠", "label": "YOLOv8 Model"},
        {"icon": "📦", "label": "Bounding Boxes"},
        {"icon": "📍", "label": "Zone Mapping"},
        {"icon": "🚨", "label": "Alert Output"},
    ])

    fire_divider()

    # ── Why This Approach ─────────────────────────────────────────────────
    section_header("TECHNOLOGY", "Why YOLOv8?")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="feature-card">
            <span class="feature-icon">🏆</span>
            <div class="feature-title">State-of-the-Art Accuracy</div>
            <div class="feature-desc">YOLOv8 achieves mAP@50 of 53.9 on COCO — the gold standard benchmark for object detection.</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="feature-card">
            <span class="feature-icon">🔄</span>
            <div class="feature-title">Transfer Learning</div>
            <div class="feature-desc">Pre-trained on 80 COCO classes including "person" — fine-tuned on fire/smoke datasets for domain adaptation.</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="feature-card">
            <span class="feature-icon">📱</span>
            <div class="feature-title">Edge Deployable</div>
            <div class="feature-desc">YOLOv8n runs on CPU at 30 FPS — no GPU required. Can be deployed on Raspberry Pi or Jetson Nano.</div>
        </div>""", unsafe_allow_html=True)

    fire_divider()

    # ── Alternatives Comparison ────────────────────────────────────────────
    section_header("COMPARISON", "Approach Alternatives")

    st.markdown("""
    <div class="result-box">
        <table style="width:100%; border-collapse:collapse; font-size:0.85rem;">
            <thead>
                <tr style="border-bottom:1px solid rgba(255,255,255,0.08);">
                    <th style="text-align:left; padding:10px 12px; color:#8888A0; font-weight:600;">Approach</th>
                    <th style="text-align:left; padding:10px 12px; color:#8888A0; font-weight:600;">Pros</th>
                    <th style="text-align:left; padding:10px 12px; color:#8888A0; font-weight:600;">Cons</th>
                    <th style="text-align:center; padding:10px 12px; color:#8888A0; font-weight:600;">Used</th>
                </tr>
            </thead>
            <tbody>
                <tr style="border-bottom:1px solid rgba(255,255,255,0.05);">
                    <td style="padding:10px 12px; font-weight:600; color:#FF4B4B;">YOLOv8 (RGB)</td>
                    <td style="padding:10px 12px; color:#E8E8F0;">Fast, accurate, cheap cameras</td>
                    <td style="padding:10px 12px; color:#8888A0;">Struggles in dense smoke</td>
                    <td style="padding:10px 12px; text-align:center;">✅</td>
                </tr>
                <tr style="border-bottom:1px solid rgba(255,255,255,0.05);">
                    <td style="padding:10px 12px; font-weight:600; color:#E8E8F0;">Thermal Camera</td>
                    <td style="padding:10px 12px; color:#E8E8F0;">Works through smoke perfectly</td>
                    <td style="padding:10px 12px; color:#8888A0;">Expensive hardware ($500–$5000)</td>
                    <td style="padding:10px 12px; text-align:center;">—</td>
                </tr>
                <tr style="border-bottom:1px solid rgba(255,255,255,0.05);">
                    <td style="padding:10px 12px; font-weight:600; color:#E8E8F0;">CNN (ResNet)</td>
                    <td style="padding:10px 12px; color:#E8E8F0;">High accuracy</td>
                    <td style="padding:10px 12px; color:#8888A0;">Slow, not real-time</td>
                    <td style="padding:10px 12px; text-align:center;">—</td>
                </tr>
                <tr>
                    <td style="padding:10px 12px; font-weight:600; color:#E8E8F0;">Radar / LiDAR</td>
                    <td style="padding:10px 12px; color:#E8E8F0;">Penetrates walls and smoke</td>
                    <td style="padding:10px 12px; color:#8888A0;">Very expensive, complex setup</td>
                    <td style="padding:10px 12px; text-align:center;">—</td>
                </tr>
            </tbody>
        </table>
    </div>
    """, unsafe_allow_html=True)

    fire_divider()

    # ── Footer ─────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="text-align:center; padding:32px 0 16px; color:#8888A0; font-size:0.8rem;">
        <div style="font-size:1.8rem; margin-bottom:8px;">🔥</div>
        <strong style="color:#E8E8F0;">FireDetect AI</strong> — Fire Emergency Human Detection System<br>
        Built with YOLOv8 · OpenCV · PyTorch · Streamlit
    </div>
    """, unsafe_allow_html=True)
