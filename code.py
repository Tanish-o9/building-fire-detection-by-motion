"""
code.py — Fire Emergency Human Detection System
Master entry point

Usage:
    python code.py                        # webcam real-time detection
    python code.py --image path/to/img   # detect on a single image
    python code.py --video path/to/vid   # detect on a video file
    python code.py --app                 # launch Streamlit web app
    python code.py --train               # fine-tune on your dataset
    python code.py --demo                # download sample image and run detection
"""

import argparse
import subprocess
import sys
import os


def run_demo():
    """Download a sample street image and run person detection on it."""
    import urllib.request
    import cv2
    from ultralytics import YOLO
    from detect import draw_detections

    sample_url = (
        "https://ultralytics.com/images/bus.jpg"
    )
    sample_path = "demo_input.jpg"
    output_path = "demo_output.jpg"

    print("Downloading sample image...")
    urllib.request.urlretrieve(sample_url, sample_path)
    print(f"Saved to {sample_path}")

    print("Loading YOLOv8n (downloads ~6 MB on first run)...")
    model = YOLO("yolov8n.pt")

    frame = cv2.imread(sample_path)
    results = model(frame, verbose=False)[0]
    detections = draw_detections(frame, results.boxes, conf_threshold=0.4)

    cv2.imwrite(output_path, frame)
    print(f"\nAnnotated image saved → {output_path}")
    print(f"Persons detected: {len(detections)}")
    for i, d in enumerate(detections, 1):
        print(f"  Person {i}: zone={d['zone']}, conf={d['conf']:.2f}, bbox={d['bbox']}")

    # Show result (opens a window; press any key to close)
    cv2.imshow("Fire Emergency Human Detection — Demo", frame)
    print("\nPress any key in the image window to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Fire Emergency Human Detection System")
    parser.add_argument("--demo",   action="store_true", help="Run demo on a sample image")
    parser.add_argument("--image",  type=str,            help="Path to input image")
    parser.add_argument("--video",  type=str,            help="Path to input video")
    parser.add_argument("--webcam", action="store_true", help="Run real-time webcam detection")
    parser.add_argument("--app",    action="store_true", help="Launch Streamlit web app")
    parser.add_argument("--train",  action="store_true", help="Fine-tune YOLOv8 on your dataset")
    parser.add_argument("--model",  type=str, default="yolov8n.pt", help="Model path")
    parser.add_argument("--conf",   type=float, default=0.4,        help="Confidence threshold")
    args = parser.parse_args()

    if args.demo:
        run_demo()

    elif args.image:
        from detect import detect_image
        detect_image(args.image, model_path=args.model, conf=args.conf)

    elif args.video:
        from detect import detect_video
        detect_video(args.video, model_path=args.model, conf=args.conf)

    elif args.webcam:
        from detect import detect_video
        detect_video(0, model_path=args.model, conf=args.conf)

    elif args.app:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])

    elif args.train:
        from train import train, evaluate
        train(data_yaml="dataset/data.yaml", model_size="n", epochs=50)
        evaluate("runs/train/fire_human_detect/weights/best.pt")

    else:
        # Default: run demo
        print("No argument provided. Running demo mode...\n")
        run_demo()


if __name__ == "__main__":
    main()
