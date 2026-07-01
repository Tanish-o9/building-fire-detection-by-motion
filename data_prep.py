"""
data_prep.py — Dataset collection and preprocessing

DATASETS TO USE:
1. COCO (persons in normal conditions) — pre-baked into YOLOv8 pretrained weights
2. Fire-Smoke-Person dataset from Roboflow Universe:
   https://universe.roboflow.com/search?q=fire+smoke+person
3. AIDER (Aerial Image Dataset for Emergency Response) — optional

WHY THESE DATASETS?
- COCO gives strong baseline person detection
- Fire/smoke datasets teach the model to detect humans under occlusion
- Combining both prevents catastrophic forgetting
"""

import os
import shutil
import urllib.request
from pathlib import Path


# ── Option A: Download via Roboflow (recommended) ──────────────────────────
def download_roboflow(api_key: str, workspace: str, project: str, version: int):
    """
    Download a labeled dataset from Roboflow.
    Sign up free at https://roboflow.com to get an API key.
    Recommended project: 'fire-smoke-detection-2' or 'person-in-fire'
    """
    from roboflow import Roboflow
    rf = Roboflow(api_key=api_key)
    proj = rf.workspace(workspace).project(project)
    dataset = proj.version(version).download("yolov8")
    print(f"Dataset downloaded to: {dataset.location}")
    return dataset.location


# ── Option B: Use a local folder of images (no API needed) ─────────────────
def prepare_local_dataset(raw_images_dir: str, output_dir: str = "dataset"):
    """
    If you have your own images, this sets up the YOLOv8 folder structure.
    You still need to label them with a tool like Label Studio or Roboflow.

    Expected output structure:
        dataset/
          images/train/   images/val/
          labels/train/   labels/val/
          data.yaml
    """
    for split in ["train", "val"]:
        Path(f"{output_dir}/images/{split}").mkdir(parents=True, exist_ok=True)
        Path(f"{output_dir}/labels/{split}").mkdir(parents=True, exist_ok=True)

    # Write data.yaml — tells YOLO where data lives and class names
    yaml_content = f"""path: {os.path.abspath(output_dir)}
train: images/train
val: images/val

nc: 1
names: ['person']
"""
    with open(f"{output_dir}/data.yaml", "w") as f:
        f.write(yaml_content)

    print(f"Dataset scaffold created at '{output_dir}/'")
    print("Next: copy images into images/train & images/val,")
    print("      then add YOLO-format .txt labels into labels/train & labels/val")


# ── Preprocessing: augment images to simulate fire/smoke ───────────────────
def augment_with_smoke(image_path: str, output_path: str, intensity: float = 0.4):
    """
    Simulate smoke by blending a grey haze over an image.
    This is a cheap way to make a normal-camera dataset look like a fire scene.

    WHY: Real fire datasets are scarce. Augmentation expands training diversity.
    """
    import cv2
    import numpy as np

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read {image_path}")

    smoke_layer = np.full_like(img, fill_value=180, dtype=np.uint8)  # grey haze
    blended = cv2.addWeighted(img, 1 - intensity, smoke_layer, intensity, 0)
    cv2.imwrite(output_path, blended)


def augment_with_fire_tint(image_path: str, output_path: str, intensity: float = 0.3):
    """
    Simulate fire glow by adding an orange tint.
    """
    import cv2
    import numpy as np

    img = cv2.imread(image_path)
    fire_layer = np.zeros_like(img, dtype=np.uint8)
    fire_layer[:, :, 2] = 200  # red channel
    fire_layer[:, :, 1] = 80   # green channel (orange = red+green)
    blended = cv2.addWeighted(img, 1 - intensity, fire_layer, intensity, 0)
    cv2.imwrite(output_path, blended)


def batch_augment(input_dir: str, output_dir: str):
    """Apply both smoke and fire augmentations to all images in a folder."""
    import cv2

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    for img_file in Path(input_dir).glob("*.jpg"):
        stem = img_file.stem
        augment_with_smoke(str(img_file), f"{output_dir}/{stem}_smoke.jpg")
        augment_with_fire_tint(str(img_file), f"{output_dir}/{stem}_fire.jpg")
    print(f"Augmented images saved to '{output_dir}/'")


if __name__ == "__main__":
    # Example: scaffold a local dataset
    prepare_local_dataset(raw_images_dir="raw_images", output_dir="dataset")

    # Example: augment images in a folder
    # batch_augment("dataset/images/train", "dataset/images/train_augmented")

    # Example: Roboflow download (fill in your credentials)
    # download_roboflow(
    #     api_key="YOUR_API_KEY",
    #     workspace="your-workspace",
    #     project="fire-smoke-person",
    #     version=1
    # )
