"""
webapp/server.py — Flask backend
Handles image/video upload, runs YOLOv8 detection, returns results to browser
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
from ultralytics import YOLO
import cv2
import numpy as np
import os
import uuid
import base64
from pathlib import Path
import webbrowser
import threading

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["OUTPUT_FOLDER"] = "outputs"
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200MB

BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_PATH = BASE_DIR.parent / "yolov8n.pt"

UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Load model once at startup
model = YOLO(str(MODEL_PATH))

ZONE_NAMES = [
    ["Top-Left",    "Top-Center",    "Top-Right"],
    ["Mid-Left",    "Center",        "Mid-Right"],
    ["Bottom-Left", "Bottom-Center", "Bottom-Right"],
]


def get_zone(cx, cy, w, h):
    col = min(int(cx / w * 3), 2)
    row = min(int(cy / h * 3), 2)
    return ZONE_NAMES[row][col]


def run_detection(frame, conf=0.4):
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

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 255), 2)
        label = f"Person {conf_val:.2f} | {zone}"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - lh - 10), (x1 + lw + 6, y1), (0, 200, 255), -1)
        cv2.putText(frame, label, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        cv2.circle(frame, (cx, cy), 6, (255, 50, 50), -1)
        detections.append({"bbox": [x1, y1, x2, y2], "conf": round(conf_val, 2), "zone": zone})

    # Draw 3x3 grid
    for i in range(1, 3):
        cv2.line(frame, (w * i // 3, 0), (w * i // 3, h), (60, 60, 60), 1)
        cv2.line(frame, (0, h * i // 3), (w, h * i // 3), (60, 60, 60), 1)

    cv2.putText(frame, f"Persons: {len(detections)}", (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 255), 2)
    return frame, detections


def frame_to_base64(frame):
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return base64.b64encode(buf).decode("utf-8")


# ── Routes ─────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/detect/image", methods=["POST"])
def detect_image():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    conf = float(request.form.get("conf", 0.4))

    img_array = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({"error": "Invalid image"}), 400

    annotated, detections = run_detection(frame, conf)

    # Save output
    out_name = f"{uuid.uuid4().hex}.jpg"
    out_path = OUTPUT_DIR / out_name
    cv2.imwrite(str(out_path), annotated)

    return jsonify({
        "image": frame_to_base64(annotated),
        "detections": detections,
        "count": len(detections),
        "output_file": out_name,
    })


@app.route("/detect/video", methods=["POST"])
def detect_video():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    conf = float(request.form.get("conf", 0.4))

    tmp_path = UPLOAD_DIR / f"{uuid.uuid4().hex}.mp4"
    file.save(str(tmp_path))

    cap = cv2.VideoCapture(str(tmp_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

    out_name = f"{uuid.uuid4().hex}.mp4"
    out_path = OUTPUT_DIR / out_name
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    all_detections = []
    frames_processed = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        annotated, detections = run_detection(frame, conf)
        all_detections.extend(detections)
        writer.write(annotated)
        frames_processed += 1

    cap.release()
    writer.release()
    tmp_path.unlink(missing_ok=True)

    zones = list({d["zone"] for d in all_detections})
    return jsonify({
        "count": len(all_detections),
        "frames": frames_processed,
        "zones": zones,
        "output_file": out_name,
        "detections": all_detections[:50],  # first 50 for display
    })


@app.route("/outputs/<filename>")
def download_output(filename):
    return send_from_directory(str(OUTPUT_DIR), filename, as_attachment=True)


if __name__ == "__main__":
    # Auto-open browser after 1.5s
    def open_browser():
        import time
        time.sleep(1.5)
        webbrowser.open("http://localhost:5000")
    threading.Thread(target=open_browser, daemon=True).start()

    print("\n🔥 FireDetect AI is running at http://localhost:5000\n")
    app.run(host="0.0.0.0", port=5000, debug=False)
