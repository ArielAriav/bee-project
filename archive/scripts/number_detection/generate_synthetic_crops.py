import cv2
import numpy as np
import os
import random
from PIL import Image, ImageDraw, ImageFont

OUTPUT_DIR = os.path.join("archive", "datasets", "digit_detection", "synthetic_crops")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- הגדרות ---
IMAGES_PER_DIGIT = 100  # כמה תמונות לייצר לכל ספרה נדירה
STICKER_SIZE = 120

# ספרות שחסרות — שני את הרשימה הזו בלבד
RARE_DIGITS = ["0", "6", "1"]

def get_numbers_containing(digit):
    """
    Generates a list of random numbers (0-999) that contain the given digit.
    """
    result = []
    for num in range(0, 1000):
        if digit in str(num):
            result.append(str(num))
    return result

def generate_sticker(digit_str, idx):
    """
    Generates a synthetic sticker image with the given number.
    """
    size = STICKER_SIZE
    img = np.ones((size, size, 3), dtype=np.uint8) * 255

    # white circle
    cv2.circle(img, (size//2, size//2), size//2 - 2, (220, 220, 220), -1)
    cv2.circle(img, (size//2, size//2), size//2 - 2, (160, 160, 160), 2)

    # draw text
    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)

    font_size = 55 if len(digit_str) == 1 else 42 if len(digit_str) == 2 else 32
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), digit_str, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x = (size - text_w) // 2
    y = (size - text_h) // 2

    # slight random position offset for variation
    x += random.randint(-4, 4)
    y += random.randint(-4, 4)

    draw.text((x, y), digit_str, fill=(20, 20, 20), font=font)
    img = np.array(pil_img)

    # random blur to simulate motion
    blur_amount = random.choice([0, 0, 1, 1, 3])
    if blur_amount > 0:
        img = cv2.GaussianBlur(img, (blur_amount*2+1, blur_amount*2+1), 0)

    # random brightness variation
    brightness = random.randint(-30, 30)
    img = np.clip(img.astype(int) + brightness, 0, 255).astype(np.uint8)

    # random slight rotation (-10 to 10 degrees) to simulate imperfect sticker placement
    angle = random.uniform(-10, 10)
    center = (size//2, size//2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    img = cv2.warpAffine(img, matrix, (size, size), borderValue=(255, 255, 255))

    filename = f"synthetic_{digit_str}_{idx:04d}.jpg"
    cv2.imwrite(os.path.join(OUTPUT_DIR, filename), img)

def main():
    total = 0

    for rare_digit in RARE_DIGITS:
        # get all numbers 0-999 that contain this digit
        candidates = get_numbers_containing(rare_digit)
        print(f"Digit '{rare_digit}': {len(candidates)} possible numbers → generating {IMAGES_PER_DIGIT} images")

        for i in range(IMAGES_PER_DIGIT):
            # pick a random number from the candidates
            chosen = random.choice(candidates)
            generate_sticker(chosen, i)
            total += 1

    print(f"\nDone. Generated {total} synthetic images.")
    print(f"Saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()