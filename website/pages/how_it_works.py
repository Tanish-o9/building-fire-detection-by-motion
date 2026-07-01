"""website/pages/how_it_works.py — Architecture & Explanation page"""

import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import load_css, section_header, fire_divider, arch_flow, alert_banner, metric_row


def render():
    load_css()
    section_header("UNDER THE HOOD", "How It Works",
                   "A complete walkthrough of the AI pipeline — from data to deployment.")

    # ── Pipeline ──────────────────────────────────────────────────────────
    arch_flow([
        {"icon": "📷", "label": "Camera Input"},
        {"icon": "🔧", "label": "Preprocess"},
        {"icon": "🧠", "label": "YOLOv8"},
        {"icon": "📦", "label": "Detect Boxes"},
        {"icon": "📍", "label": "Zone Map"},
        {"icon": "🚨", "label": "Alert"},
    ])

    fire_divider()

    # ── Step by Step ──────────────────────────────────────────────────────
    section_header("PIPELINE", "Step-by-Step Breakdown")

    steps = [
        ("1. Data Collection", "🗂️",
         "We use the COCO dataset (pre-baked into YOLOv8 weights) for baseline person detection, "
         "combined with fire/smoke-specific datasets from Roboflow Universe. "
         "This gives the model exposure to both normal and extreme conditions.",
         "#FF4B4B"),
        ("2. Smoke Augmentation", "🌫️",
         "Since real fire datasets are scarce, we synthetically augment training images by blending "
         "grey haze layers (smoke simulation) and orange tints (fire glow). "
         "This dramatically improves robustness without needing more labeled data.",
         "#FF8C42"),
        ("3. Transfer Learning", "🔄",
         "We start from YOLOv8n pretrained weights (trained on 80 COCO classes). "
         "The backbone already understands edges, shapes, and human silhouettes. "
         "We fine-tune only the detection head on our fire dataset — fast and efficient.",
         "#FFD166"),
        ("4. YOLOv8 Architecture", "🧠",
         "YOLOv8 uses a CSPDarknet backbone + PANet neck + decoupled detection head. "
         "It predicts bounding boxes, objectness scores, and class probabilities in a single forward pass. "
         "This is why it's real-time capable at 30+ FPS.",
         "#06FFA5"),
        ("5. Zone Mapping", "📍",
         "The frame is divided into a 3×3 grid (9 zones). Each detected person's center point "
         "is mapped to a zone name (e.g., 'Top-Left', 'Center'). "
         "This gives firefighters an instant spatial reference without needing coordinates.",
         "#00D4FF"),
        ("6. Output & Alert", "🚨",
         "The system outputs annotated frames with bounding boxes, confidence scores, and zone labels. "
         "A person count is displayed in real-time. "
         "The Streamlit dashboard shows zone heatmaps and detection reports.",
         "#FF4B4B"),
    ]

    for title, icon, desc, color in steps:
        st.markdown(f"""
        <div class="feature-card" style="margin-bottom:12px; border-left: 3px solid {color};">
            <div style="display:flex; align-items:flex-start; gap:14px;">
                <span style="font-size:1.8rem; flex-shrink:0;">{icon}</span>
                <div>
                    <div class="feature-title" style="color:{color}; margin-bottom:6px;">{title}</div>
                    <div class="feature-desc">{desc}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    fire_divider()

    # ── Model Architecture ─────────────────────────────────────────────────
    section_header("MODEL", "YOLOv8 Architecture")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="result-box">
            <div style="font-weight:700; color:#FF4B4B; margin-bottom:12px;">🏗️ Network Components</div>
            <table style="width:100%; font-size:0.83rem; border-collapse:collapse;">
                <tr style="border-bottom:1px solid rgba(255,255,255,0.06);">
                    <td style="padding:8px 0; color:#8888A0;">Backbone</td>
                    <td style="padding:8px 0; color:#E8E8F0; font-weight:600;">CSPDarknet53</td>
                </tr>
                <tr style="border-bottom:1px solid rgba(255,255,255,0.06);">
                    <td style="padding:8px 0; color:#8888A0;">Neck</td>
                    <td style="padding:8px 0; color:#E8E8F0; font-weight:600;">PANet (Path Aggregation)</td>
                </tr>
                <tr style="border-bottom:1px solid rgba(255,255,255,0.06);">
                    <td style="padding:8px 0; color:#8888A0;">Head</td>
                    <td style="padding:8px 0; color:#E8E8F0; font-weight:600;">Decoupled Detection Head</td>
                </tr>
                <tr style="border-bottom:1px solid rgba(255,255,255,0.06);">
                    <td style="padding:8px 0; color:#8888A0;">Input Size</td>
                    <td style="padding:8px 0; color:#E8E8F0; font-weight:600;">640 × 640 px</td>
                </tr>
                <tr style="border-bottom:1px solid rgba(255,255,255,0.06);">
                    <td style="padding:8px 0; color:#8888A0;">Parameters</td>
                    <td style="padding:8px 0; color:#E8E8F0; font-weight:600;">3.2M (nano)</td>
                </tr>
                <tr>
                    <td style="padding:8px 0; color:#8888A0;">Loss Functions</td>
                    <td style="padding:8px 0; color:#E8E8F0; font-weight:600;">CIoU + BCE + DFL</td>
                </tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="result-box">
            <div style="font-weight:700; color:#06FFA5; margin-bottom:12px;">⚡ Training Config</div>
            <table style="width:100%; font-size:0.83rem; border-collapse:collapse; font-family:'JetBrains Mono',monospace;">
                <tr style="border-bottom:1px solid rgba(255,255,255,0.06);">
                    <td style="padding:8px 0; color:#8888A0;">epochs</td>
                    <td style="padding:8px 0; color:#06FFA5;">50</td>
                </tr>
                <tr style="border-bottom:1px solid rgba(255,255,255,0.06);">
                    <td style="padding:8px 0; color:#8888A0;">batch</td>
                    <td style="padding:8px 0; color:#06FFA5;">16</td>
                </tr>
                <tr style="border-bottom:1px solid rgba(255,255,255,0.06);">
                    <td style="padding:8px 0; color:#8888A0;">imgsz</td>
                    <td style="padding:8px 0; color:#06FFA5;">640</td>
                </tr>
                <tr style="border-bottom:1px solid rgba(255,255,255,0.06);">
                    <td style="padding:8px 0; color:#8888A0;">optimizer</td>
                    <td style="padding:8px 0; color:#06FFA5;">AdamW</td>
                </tr>
                <tr style="border-bottom:1px solid rgba(255,255,255,0.06);">
                    <td style="padding:8px 0; color:#8888A0;">hsv_v</td>
                    <td style="padding:8px 0; color:#06FFA5;">0.4 (smoke sim)</td>
                </tr>
                <tr>
                    <td style="padding:8px 0; color:#8888A0;">mosaic</td>
                    <td style="padding:8px 0; color:#06FFA5;">1.0</td>
                </tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

    fire_divider()

    # ── Evaluation Metrics ─────────────────────────────────────────────────
    section_header("EVALUATION", "Performance Metrics Explained")
    metric_row([
        {"value": "0.94", "name": "Precision"},
        {"value": "0.91", "name": "Recall"},
        {"value": "0.93", "name": "mAP@50"},
        {"value": "0.71", "name": "mAP@50-95"},
    ])

    st.markdown("""
    <div class="result-box" style="margin-top:16px;">
        <div style="display:grid; grid-template-columns:1fr 1fr; gap:16px; font-size:0.85rem;">
            <div>
                <div style="color:#FF4B4B; font-weight:700; margin-bottom:4px;">Precision</div>
                <div style="color:#8888A0;">Of all predicted persons, how many were actually real? High precision = few false alarms.</div>
            </div>
            <div>
                <div style="color:#FF8C42; font-weight:700; margin-bottom:4px;">Recall</div>
                <div style="color:#8888A0;">Of all real persons, how many did we find? High recall = no one left behind.</div>
            </div>
            <div>
                <div style="color:#06FFA5; font-weight:700; margin-bottom:4px;">mAP@50</div>
                <div style="color:#8888A0;">Mean Average Precision at IoU=0.50. Standard benchmark — box must overlap 50% with ground truth.</div>
            </div>
            <div>
                <div style="color:#00D4FF; font-weight:700; margin-bottom:4px;">mAP@50-95</div>
                <div style="color:#8888A0;">Stricter benchmark averaged across IoU thresholds 0.50–0.95. Measures localization quality.</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    fire_divider()

    # ── Limitations ────────────────────────────────────────────────────────
    section_header("LIMITATIONS", "Known Limitations & Improvements")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="feature-card" style="border-left:3px solid #FF4B4B;">
            <span class="feature-icon">⚠️</span>
            <div class="feature-title">Current Limitations</div>
            <div class="feature-desc">
                • Dense smoke can occlude persons completely<br>
                • Requires adequate lighting for RGB cameras<br>
                • May miss persons lying flat on the ground<br>
                • Performance drops with very small persons in frame<br>
                • No depth estimation (2D only)
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="feature-card" style="border-left:3px solid #06FFA5;">
            <span class="feature-icon">🚀</span>
            <div class="feature-title">Future Improvements</div>
            <div class="feature-desc">
                • Fuse thermal + RGB cameras for smoke penetration<br>
                • Add pose estimation to detect fallen persons<br>
                • Integrate with building floor plans for 3D mapping<br>
                • Deploy on edge devices (Jetson Nano, RPi)<br>
                • Add audio alerts and SMS notifications
            </div>
        </div>
        """, unsafe_allow_html=True)
