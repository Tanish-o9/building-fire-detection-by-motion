"""
app.py — Streamlit web app for Fire Emergency Human Detection

Run with:  streamlit run app.py
"""

import streamlit as st
import cv2
import numpy as np
import tempfile
from pathlib import Path
from PIL import Image
from ultralytics import YOLO
from detect import draw_detections


@st.cache_resource
def load_model(model_path: str):
    return YOLO(model_path)


def process_frame(frame: np.ndarray, model, conf: float) -> tuple[np.ndarray, list]:
    results = model(frame, verbose=False)[0]
    detections = draw_detections(frame, results.boxes, conf)
    return frame, detections


# ── UI ─────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Fire Emergency Human Detection", page_icon="🔥", layout="wide")
st.title("🔥 Fire Emergency Human Detection System")
st.caption("Detects humans in fire/smoke conditions using YOLOv8")

with st.sidebar:
    st.header("Settings")
    model_path = st.text_input("Model path", value="yolov8n.pt",
                               help="Use 'yolov8n.pt' for pretrained, or path to your fine-tuned best.pt")
    conf_threshold = st.slider("Confidence threshold", 0.1, 1.0, 0.4, 0.05)
    mode = st.radio("Input mode", ["Upload Image", "Upload Video", "Webcam"])

model = load_model(model_path)

# ── Image mode ─────────────────────────────────────────────────────────────
if mode == "Upload Image":
    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        annotated, detections = process_frame(frame, model, conf_threshold)
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

        col1, col2 = st.columns(2)
        col1.image(img, caption="Original", use_container_width=True)
        col2.image(annotated_rgb, caption="Detected", use_container_width=True)

        st.subheader(f"Found {len(detections)} person(s)")
        for i, d in enumerate(detections, 1):
            st.write(f"**Person {i}** — Zone: `{d['zone']}` | Confidence: `{d['conf']:.2f}` | BBox: `{d['bbox']}`")

# ── Video mode ─────────────────────────────────────────────────────────────
elif mode == "Upload Video":
    uploaded = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    if uploaded:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded.read())
        tfile.close()

        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()
        info_box = st.empty()

        stop = st.button("Stop")
        while cap.isOpened() and not stop:
            ret, frame = cap.read()
            if not ret:
                break
            annotated, detections = process_frame(frame, model, conf_threshold)
            stframe.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                          channels="RGB", use_container_width=True)
            zones = [d["zone"] for d in detections]
            info_box.info(f"Persons: {len(detections)} | Zones: {', '.join(zones) or 'None'}")

        cap.release()
        Path(tfile.name).unlink(missing_ok=True)

# ── Webcam mode ────────────────────────────────────────────────────────────
elif mode == "Webcam":
    st.warning("Webcam mode streams from your local machine. Click 'Start' then 'Stop' to end.")
    run = st.checkbox("Start webcam")
    stframe = st.empty()
    info_box = st.empty()

    if run:
        cap = cv2.VideoCapture(0)
        while run:
            ret, frame = cap.read()
            if not ret:
                st.error("Cannot access webcam.")
                break
            annotated, detections = process_frame(frame, model, conf_threshold)
            stframe.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                          channels="RGB", use_container_width=True)
            zones = [d["zone"] for d in detections]
            info_box.info(f"Persons: {len(detections)} | Zones: {', '.join(zones) or 'None'}")
            run = st.session_state.get("Start webcam", True)
        cap.release()
