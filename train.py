"""
train.py — Fine-tune YOLOv8 for human detection in fire/smoke conditions

MODEL CHOICE — YOLOv8n (nano):
  - 'n' = nano: fastest, smallest; good for real-time on CPU/GPU
  - Upgrade to 'yolov8s' or 'yolov8m' for better accuracy if you have a GPU
  - Pre-trained on COCO → already detects 'person' (class 0)

TRAINING STRATEGY:
  - Start from pretrained weights (transfer learning)
  - Fine-tune on fire/smoke dataset so the model learns to see through haze
  - Freeze backbone for first few epochs to preserve learned features (optional)
"""

from ultralytics import YOLO
import matplotlib.pyplot as plt
import json
from pathlib import Path


def train(data_yaml: str = "dataset/data.yaml",
          model_size: str = "n",
          epochs: int = 50,
          img_size: int = 640,
          batch: int = 16,
          project: str = "runs/train",
          name: str = "fire_human_detect"):
    """
    Fine-tune YOLOv8 on your dataset.

    Args:
        data_yaml:   path to data.yaml created by data_prep.py
        model_size:  'n' | 's' | 'm' | 'l' | 'x'  (nano → xlarge)
        epochs:      number of training epochs (50 is a good start)
        img_size:    input resolution (640 standard; 416 for speed)
        batch:       images per batch (reduce if OOM)
        project:     output folder
        name:        run name
    """
    model = YOLO(f"yolov8{model_size}.pt")  # downloads pretrained weights automatically

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=img_size,
        batch=batch,
        project=project,
        name=name,
        # Augmentation — helps simulate fire/smoke if not already in dataset
        hsv_h=0.015,   # hue shift
        hsv_s=0.7,     # saturation (simulates smoke desaturation)
        hsv_v=0.4,     # brightness (simulates dark/smoky rooms)
        fliplr=0.5,    # horizontal flip
        mosaic=1.0,    # mosaic augmentation (combines 4 images)
        degrees=10,    # rotation
        translate=0.1,
        scale=0.5,
        verbose=True,
    )
    return results


def evaluate(model_path: str, data_yaml: str = "dataset/data.yaml", img_size: int = 640):
    """
    Evaluate trained model and print precision, recall, mAP50, mAP50-95.

    METRICS EXPLAINED:
      Precision  = of all predicted persons, how many were real?
      Recall     = of all real persons, how many did we find?
      mAP50      = mean Average Precision at IoU=0.50 (standard benchmark)
      mAP50-95   = stricter benchmark across multiple IoU thresholds
    """
    model = YOLO(model_path)
    metrics = model.val(data=data_yaml, imgsz=img_size)

    print("\n── Evaluation Results ──────────────────────")
    print(f"  Precision : {metrics.box.mp:.4f}")
    print(f"  Recall    : {metrics.box.mr:.4f}")
    print(f"  mAP@50    : {metrics.box.map50:.4f}")
    print(f"  mAP@50-95 : {metrics.box.map:.4f}")
    print("────────────────────────────────────────────\n")
    return metrics


def plot_training_curves(run_dir: str = "runs/train/fire_human_detect"):
    """Plot loss and mAP curves from the training results CSV."""
    import pandas as pd

    csv_path = Path(run_dir) / "results.csv"
    if not csv_path.exists():
        print(f"No results.csv found at {csv_path}")
        return

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss curves
    axes[0].plot(df["epoch"], df["train/box_loss"], label="Box Loss (train)")
    axes[0].plot(df["epoch"], df["val/box_loss"], label="Box Loss (val)")
    axes[0].set_title("Box Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    # mAP curves
    axes[1].plot(df["epoch"], df["metrics/mAP50(B)"], label="mAP@50")
    axes[1].plot(df["epoch"], df["metrics/mAP50-95(B)"], label="mAP@50-95")
    axes[1].set_title("mAP")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(f"{run_dir}/training_curves.png")
    plt.show()
    print(f"Curves saved to {run_dir}/training_curves.png")


if __name__ == "__main__":
    # Step 1: Train
    train(
        data_yaml="dataset/data.yaml",
        model_size="n",
        epochs=50,
        batch=16,
    )

    # Step 2: Evaluate best checkpoint
    evaluate(
        model_path="runs/train/fire_human_detect/weights/best.pt",
        data_yaml="dataset/data.yaml",
    )

    # Step 3: Plot curves
    plot_training_curves("runs/train/fire_human_detect")
