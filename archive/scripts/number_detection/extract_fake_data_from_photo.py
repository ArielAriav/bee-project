import cv2
import numpy as np
import os

INPUT_IMAGE = "archive/datasets/digit_detection/fake_stickers_2.jfif"  # שני לנתיב התמונה שלך
OUTPUT_DIR = os.path.join("archive", "datasets", "digit_detection", "synthetic_crops_from_photo2")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_stickers(image_path, output_dir):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not open image: {image_path}")
        return

    original = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # threshold to find the red circles
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_blue = np.array([100, 100, 50])
    upper_blue = np.array([130, 255, 255])

    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # find contours of the circles
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    print(f"Found {len(contours)} contours")

    saved = 0
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)

        # filter out noise — only keep circles of reasonable size
        if area < 1000:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        # filter out non-square shapes (circles should be roughly square)
        aspect_ratio = w / h
        if not (0.7 < aspect_ratio < 1.3):
            continue

        # add small padding
        pad = 5
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(img.shape[1], x + w + pad)
        y2 = min(img.shape[0], y + h + pad)

        crop = original[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        cv2.imwrite(os.path.join(output_dir, f"sticker_{saved:04d}.jpg"), crop)
        saved += 1

    print(f"Saved {saved} stickers to: {output_dir}")

if __name__ == "__main__":
    extract_stickers(INPUT_IMAGE, OUTPUT_DIR)