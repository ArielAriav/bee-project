import os
from ultralytics import YOLO
import torch

def train_digit_model():
    # 1. Load the YOLO11 Nano model
    model = YOLO("yolo11n.pt")

    # 2. Check for GPU availability
    # The output '0' confirmed your GPU is active
    selected_device = 0 if torch.cuda.is_available() else "cpu"
    print(f"--- 🚀 Training starting on: {selected_device} ---")

    # 3. Path setup
    base_path = os.path.abspath("archive/datasets/digit_recognition")
    yaml_path = os.path.join(base_path, "data.yaml")

    # 4. Start training with corrected arguments
    model.train(
        data=yaml_path,
        epochs=150,
        imgsz=640,
        batch=16,
        device=selected_device,
        name="bee_digit_v2_rgb",
        
        # --- AUGMENTATIONS ---
        # Removed 'blur' to prevent SyntaxError
        degrees=180.0,    # Random rotation for bee orientation
        scale=0.5,        # Random scaling
        perspective=0.0001,
        mosaic=1.0,       # Combines 4 images to improve small object detection
        mixup=0.1,
        copy_paste=0.1,
        
        # --- TRAINING SETTINGS ---
        patience=20,      # Early stopping if no improvement
        save=True,
        plots=True
    )

if __name__ == "__main__":
    train_digit_model()