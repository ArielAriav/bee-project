import cv2
import os
import shutil
import numpy as np

INPUT_DIR = os.path.join("archive", "datasets", "digit_recognition", "straightened_crops")
OUTPUT_DIR = os.path.join("archive", "datasets", "digit_recognition", "filtered_crops")

# how different a crop must be to be kept (0-1, higher = stricter filtering)
SIMILARITY_THRESHOLD = 0.95
# max similar crops to keep per "group"
MAX_SIMILAR = 8

def image_hash(img, size=16):
    """
    Converts image to a small grayscale thumbnail for fast comparison.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (size, size))
    return resized.flatten().astype(np.float32)

def similarity(h1, h2):
    """
    Returns cosine similarity between two image hashes (0-1).
    """
    norm1 = np.linalg.norm(h1)
    norm2 = np.linalg.norm(h2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return np.dot(h1, h2) / (norm1 * norm2)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_files = sorted(os.listdir(INPUT_DIR))
    image_files = [f for f in all_files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"Total input crops: {len(image_files)}")

    # list of (hash, similar_count) for every kept image
    kept_hashes = []
    kept = 0
    skipped = 0

    for file_name in image_files:
        path = os.path.join(INPUT_DIR, file_name)
        img = cv2.imread(path)
        if img is None:
            continue

        h = image_hash(img)

        # check similarity against all kept images
        too_similar = False
        for i, (kept_hash, count) in enumerate(kept_hashes):
            sim = similarity(h, kept_hash)
            if sim >= SIMILARITY_THRESHOLD:
                if count >= MAX_SIMILAR:
                    # this group is full — skip the image
                    too_similar = True
                else:
                    # group has room — count it and keep
                    kept_hashes[i] = (kept_hash, count + 1)
                break

        if too_similar:
            skipped += 1
            continue

        shutil.copy(path, os.path.join(OUTPUT_DIR, file_name))
        if not any(similarity(h, kh) >= SIMILARITY_THRESHOLD for kh, _ in kept_hashes):
            kept_hashes.append((h, 1))
        kept += 1

        if kept % 100 == 0:
            print(f"Progress: kept {kept} so far...")

    print(f"\nDone.")
    print(f"Kept:    {kept}")
    print(f"Skipped: {skipped}")
    print(f"Saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()