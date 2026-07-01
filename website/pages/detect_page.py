"""website/pages/detect_page.py — Live Detection page"""

import streamlit as st
import cv2
import numpy as np
import tempfile
import sys
import time
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils import load_css, section_header, fire_divider, detection_badge, zone_grid, alert_banner

from ultralytics import YOLO


@st.cache_resource
def load_model(path):
    return YOLO(path)


ZONE_NAMES = [
    ["Top-Left", "Top-Center", "Top-Right"],
    ["Mid-Left", "Center", "Mid-Right"],
    ["Bottom-Left", "Bottom-Center", "Bottom-Right"],
]


def get_zone(cx, cy, w, h):
    col = min(int(cx / w * 3), 2)
    row = min(int(cy / h * 3), 2)
    return ZONE_NAMES[row][col]


def run_detection(frame, model, conf):
    h, w = frame.shape[:2]
    results = model(frame, verbose=False)[0]
    detections = []
    for box in results.boxes:
        if int(box.cls[0]) != 0 or float(box.conf[0]) < conf:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        zone = get_zone(cx, cy, w, h)
        conf_val = float(box.conf[0])

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 255), 2)
        label = f"Person {conf_val:.2f} | {zone}"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (x1, y1 - lh - 8), (x1 + lw + 4, y1), (0, 200, 255), -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
        cv2.circle(frame, (cx, cy), 5, (255, 50, 50), -1)
        detections.append({"bbox": (x1, y1, x2, y2), "conf": conf_val, "zone": zone})

    for i in range(1, 3):
        cv2.line(frame, (w * i // 3, 0), (w * i // 3, h), (80, 80, 80), 1)
        cv2.line(frame, (0, h * i // 3), (w, h * i // 3), (80, 80, 80), 1)

    cv2.putText(frame, f"Persons: {len(detections)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2)
    return frame, detections


def render():
    load_css()

    section_header("LIVE DETECTION", "Human Detection Engine",
                   "Upload an image or video — the AI will locate every person and report their zone.")

    # ── Sidebar controls ──────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### ⚙️ Detection Settings")
        model_path = st.text_input("Model", "yolov8n.pt")
        conf = st.slider("Confidence", 0.1, 1.0, 0.4, 0.05)
        mode = st.radio("Input Source", ["🖼️ Image", "🎬 Video", "📷 Webcam"])
        st.markdown("---")
        st.markdown("**Tips:**\n- Lower confidence = more detections\n- Use `best.pt` after training")

    model = load_model(model_path)

    # ── Image ─────────────────────────────────────────────────────────────
    if "Image" in mode:
        uploaded = st.file_uploader("Drop an image here", type=["jpg", "jpeg", "png", "bmp"],
                                    label_visibility="collapsed")
        if uploaded:
            img = Image.open(uploaded).convert("RGB")
            frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

            with st.spinner("🔍 Analyzing..."):
                annotated, detections = run_detection(frame.copy(), model, conf)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Original**")
                st.image(img, use_container_width=True)
            with col2:
                st.markdown("**Detected**")
                st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)

            fire_divider()
            detection_badge(len(detections))

            if detections:
                active_zones = list({d["zone"] for d in detections})
                st.markdown("#### 📍 Zone Map")
                zone_grid(active_zones)

                st.markdown("#### 📋 Detection Report")
                for i, d in enumerate(detections, 1):
                    st.markdown(f"""
                    <div class="result-box" style="margin:8px 0;">
                        <strong style="color:#FF4B4B;">Person {i}</strong> &nbsp;|&nbsp;
                        Zone: <code>{d['zone']}</code> &nbsp;|&nbsp;
                        Confidence: <code>{d['conf']:.2f}</code> &nbsp;|&nbsp;
                        BBox: <code>{d['bbox']}</code>
                    </div>""", unsafe_allow_html=True)

                cv2.imwrite("website/last_output.jpg", annotated)
                with open("website/last_output.jpg", "rb") as f:
                    st.download_button("⬇️ Download Annotated Image", f,
                                       file_name="fire_detection_output.jpg", mime="image/jpeg")
        else:
            alert_banner("📂", "No image uploaded yet. Drop a JPG/PNG above to start detection.")

    # ── Video ─────────────────────────────────────────────────────────────
    elif "Video" in mode:
        uploaded = st.file_uploader("Drop a video here", type=["mp4", "avi", "mov"],
                                    label_visibility="collapsed")
        if uploaded:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded.read())
            tfile.close()

            cap = cv2.VideoCapture(tfile.name)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

            col1, col2 = st.columns([3, 1])
            with col1:
                stframe = st.empty()
            with col2:
                info = st.empty()
                progress = st.progress(0)

            stop = st.button("⏹ Stop")
            frame_n = 0
            all_detections = []

            while cap.isOpened() and not stop:
                ret, frame = cap.read()
                if not ret:
                    break
                annotated, detections = run_detection(frame, model, conf)
                all_detections.extend(detections)
                stframe.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                              channels="RGB", use_container_width=True)
                frame_n += 1
                progress.progress(min(frame_n / total, 1.0))
                zones = list({d["zone"] for d in detections})
                info.markdown(f"""
                <div class="result-box">
                    <div style="font-size:1.6rem; font-weight:800; color:#FF4B4B;">{len(detections)}</div>
                    <div style="font-size:0.75rem; color:#8888A0;">PERSONS</div>
                    <hr style="border-color:rgba(255,255,255,0.08); margin:8px 0;">
                    <div style="font-size:0.75rem; color:#E8E8F0;">{'<br>'.join(zones) or 'None'}</div>
                </div>""", unsafe_allow_html=True)

            cap.release()
            Path(tfile.name).unlink(missing_ok=True)

            if all_detections:
                fire_divider()
                st.success(f"✅ Video processed — {len(all_detections)} total person detections across all frames.")
        else:
            alert_banner("🎬", "No video uploaded yet. Drop an MP4/AVI above to start detection.")

    # ── Webcam ────────────────────────────────────────────────────────────
    elif "Webcam" in mode:
        alert_banner("📷", "Webcam streams from your local machine. Click <strong>Start</strong> to begin.")
        run = st.checkbox("▶ Start Webcam")

        if run:
            stframe = st.empty()
            info = st.empty()
            cap = cv2.VideoCapture(0)

            if not cap.isOpened():
                st.error("❌ Cannot access webcam. Make sure it is connected and not in use.")
            else:
                while run:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    annotated, detections = run_detection(frame, model, conf)
                    stframe.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                                  channels="RGB", use_container_width=True)
                    zones = list({d["zone"] for d in detections})
                    info.markdown(f"**Persons:** {len(detections)} | **Zones:** {', '.join(zones) or 'None'}")
                cap.release()
