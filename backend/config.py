# config.py
import os

# Using os.path.join for cross-platform compatibility (Windows & Mac)
MODEL_PATH = os.path.join("models", "tag_detection", "best.pt")
DIGIT_MODEL_PATH = os.path.join("models", "number_detection", "best.pt")
ANGLE_MODEL_PATH = os.path.join("models", "angle_detection", "best.pt")

DET_CONF = 0.5
MAX_DIGITS = 3

OCR_EVERY = 4

# Locking Logic
LOCK_COUNT = 8

CROP_EXPAND = 50
SAVE_CROPS_DIR = "angleModelData"
