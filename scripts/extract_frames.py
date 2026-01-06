import cv2
from pathlib import Path

VIDEO_PATH = Path("data/raw/entrance/beeVideo.mp4") # Change the video name here
OUT_DIR = Path("data/frames/beeVideo") # Change the video name here
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Every few frames save one imageת
# Save 1 out of every 5 frames
EVERY_N_FRAMES = 5

# Limit to not produce too many images the first time
MAX_SAVED = 800

def main():
    if not VIDEO_PATH.exists():
        raise FileNotFoundError(f"Video not found: {VIDEO_PATH.resolve()}")

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise RuntimeError("Could not open video")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"FPS: {fps:.2f}, Total frames: {total}")

    frame_idx = 0
    saved = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % EVERY_N_FRAMES == 0:
            out_path = OUT_DIR / f"{saved:06d}.jpg"
            cv2.imwrite(str(out_path), frame)
            saved += 1

            if saved >= MAX_SAVED:
                break

        frame_idx += 1

    cap.release()
    print(f"Saved {saved} frames to: {OUT_DIR.resolve()}")

if __name__ == "__main__":
    main()
