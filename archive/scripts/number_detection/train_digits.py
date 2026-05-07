import os
import torch
from ultralytics import YOLO

def train_digit_model():
    # path to the latest dataset version
    base_path = os.path.abspath(os.path.join("archive", "datasets", "digit_detection", "Version2"))
    yaml_path = os.path.join(base_path, "data.yaml")

    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"data.yaml not found at: {yaml_path}")

    # use GPU if available
    selected_device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Training on: {selected_device}")
    print(f"Dataset: {yaml_path}")

    # start from scratch with base YOLOv8 nano model
    model = YOLO("yolov8n.pt")

    model.train(
        data=yaml_path,
        epochs=100,
        imgsz=640,
        batch=16,
        device=selected_device,
        project="runs/digit_detection",
        name="v1",

        # --- AUGMENTATIONS ---
        # no rotation — angle model already straightens the crop before this model sees it
        degrees=0.0,
        # mild scale variation to handle different tag sizes
        scale=0.3,
        # very slight perspective for natural camera angle variation
        perspective=0.0001,
        # mosaic combines 4 images — helps detect small digits in context
        mosaic=1.0,
        # no mixup or copy_paste — these blur digit identity and confuse 6 vs 9
        mixup=0.0,
        copy_paste=0.0,
        # mild flip disabled — flipping digits creates wrong classes (6 becomes 9)
        fliplr=0.0,
        flipud=0.0,

        # --- TRAINING SETTINGS ---
        # stop early if no improvement for 15 consecutive epochs
        patience=15,
        # dropout for regularization to reduce overfitting
        dropout=0.1,
        save=True,
        plots=True,
        verbose=True
    )

if __name__ == "__main__":
    train_digit_model()