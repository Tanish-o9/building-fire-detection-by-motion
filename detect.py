"""
detect.py — Real-time human detection on image / video / webcam

HOW IT WORKS:
  1. Load YOLOv8 model (pretrained or fine-tuned)
  2. Run inference on each frame
  3. Draw bounding boxes + zone labels
  4. Print coordinates of detected persons

ZONE SYSTEM:
  The frame is divided into a 3×3 grid (like a tic-tac-toe board).
  Each detected person is assigned a zone name (e.g., "Top-Left", "Center").
  This gives firefighters a quick spatial reference.
"""

import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path


ZONE_NAMES = [
    ["Top-Left",    "Top-Center",    "Top-Right"],
    ["Mid-Left",    "Center",        "Mid-Right"],
    ["Bottom-Left", "Bottom-Center", "Bottom-Right"],
]


def get_zone(cx: int, cy: int, frame_w: int, frame_h: int) -> str:
    col = min(int(cx / frame_w * 3), 2)
    row = min(int(cy / frame_h * 3), 2)
    return ZONE_NAMES[row][col]


def draw_detections(frame: np.ndarray, boxes, conf_threshold: float = 0.4) -> list[dict]:
    """
    Draw bounding boxes on frame and return list of detected person info.
    Returns: [{"bbox": (x1,y1,x2,y2), "conf": float, "zone": str}, ...]
    """
    h, w = frame.shape[:2]
    detections = []

    for box in boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        if cls != 0 or conf < conf_threshold:   # class 0 = person in COCO
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        zone = get_zone(cx, cy, w, h)

        # Bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Label background
        label = f"Person {conf:.2f} | {zone}"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (x1, y1 - lh - 6), (x1 + lw, y1), (0, 255, 0), -1)
        cv2.putText(frame, label, (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)

        # Center dot
        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

        detections.append({"bbox": (x1, y1, x2, y2), "conf": conf, "zone": zone})

    # Draw 3×3 grid overlay
    for i in range(1, 3):
        cv2.line(frame, (w * i // 3, 0), (w * i // 3, h), (100, 100, 100), 1)
        cv2.line(frame, (0, h * i // 3), (w, h * i // 3), (100, 100, 100), 1)

    # Person count
    cv2.putText(frame, f"Persons detected: {len(detections)}", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    return detections


def detect_image(image_path: str, model_path: str = "yolov8n.pt",
                 output_path: str = "output.jpg", conf: float = 0.4):
    """Run detection on a single image and save result."""
    model = YOLO(model_path)
    frame = cv2.imread(image_path)
    if frame is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    results = model(frame, verbose=False)[0]
    detections = draw_detections(frame, results.boxes, conf)

    cv2.imwrite(output_path, frame)
    print(f"Saved to {output_path}")
    for i, d in enumerate(detections, 1):
        print(f"  Person {i}: bbox={d['bbox']}, conf={d['conf']:.2f}, zone={d['zone']}")
    return detections


def detect_video(source, model_path: str = "yolov8n.pt",
                 output_path: str = None, conf: float = 0.4):
    """
    Run detection on a video file or webcam.

    Args:
        source: 0 for webcam, or path to video file (str)
        output_path: if provided, saves annotated video
    """
    model = YOLO(model_path)
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {source}")

    writer = None
    if output_path:
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(output_path,
                                 cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    print("Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False)[0]
        detections = draw_detections(frame, results.boxes, conf)

        if writer:
            writer.write(frame)

        cv2.imshow("Fire Emergency Human Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import sys

    # Usage examples:
    #   python detect.py                        → webcam
    #   python detect.py video.mp4              → video file
    #   python detect.py image.jpg              → single image
    #   python detect.py video.mp4 best.pt      → custom model

    source = sys.argv[1] if len(sys.argv) > 1 else 0
    model  = sys.argv[2] if len(sys.argv) > 2 else "yolov8n.pt"

    if isinstance(source, str) and Path(source).suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
        detect_image(source, model_path=model)
    else:
        detect_video(source if source == 0 else source, model_path=model)
