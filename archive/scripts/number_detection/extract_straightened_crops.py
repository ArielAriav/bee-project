import cv2
import os
from ultralytics import YOLO

# Path configuration
TAG_MODEL_PATH = os.path.join("backend", "models", "tag_detection", "best.pt")
ANGLE_MODEL_PATH = os.path.join("backend", "models", "angle_detection", "best.pt")
VIDEO_DIR = os.path.join("data", "raw", "entrance")
OUTPUT_DIR = os.path.join("archive", "datasets", "digit_recognition", "straightened_crops")

# Logic parameters
CONF_THRESHOLD = 0.5
FRAME_SKIP = 15
MAX_CROPS_PER_VIDEO = 300
CROP_EXPAND = 50

ROTATION_MAP = {
    "Down":  cv2.ROTATE_180,
    "Left":  cv2.ROTATE_90_CLOCKWISE,
    "Right": cv2.ROTATE_90_COUNTERCLOCKWISE,
    "Up":    None  # already upright
}

def straighten_crop(crop, direction):
    rotation = ROTATION_MAP.get(direction)
    if rotation is not None:
        return cv2.rotate(crop, rotation)
    return crop

def process_video(tag_model, angle_model, video_path, output_dir):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Could not open: {video_path}")
        return 0

    total_saved = 0
    frame_idx = 0
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    print(f"Processing: {video_path}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if total_saved >= MAX_CROPS_PER_VIDEO:
            print(f"Reached quota of {MAX_CROPS_PER_VIDEO}. Stopping.")
            break

        if frame_idx % FRAME_SKIP == 0:
            # detect tag location
            tag_results = tag_model(frame, conf=CONF_THRESHOLD, verbose=False)

            if tag_results[0].boxes is not None:
                for i, box in enumerate(tag_results[0].boxes.xyxy):
                    x1, y1, x2, y2 = map(int, box)

                    # expanded crop for angle detection
                    h, w = frame.shape[:2]
                    ey1 = max(0, y1 - CROP_EXPAND)
                    ey2 = min(h, y2 + CROP_EXPAND)
                    ex1 = max(0, x1 - CROP_EXPAND)
                    ex2 = min(w, x2 + CROP_EXPAND)
                    expanded_crop = frame[ey1:ey2, ex1:ex2]

                    if expanded_crop.size == 0:
                        continue

                    # detect direction
                    angle_results = angle_model.predict(expanded_crop, verbose=False)
                    direction = angle_results[0].names[angle_results[0].probs.top1]

                    # tight crop for saving
                    tight_crop = frame[y1:y2, x1:x2]
                    if tight_crop.size == 0:
                        continue

                    # straighten and save
                    straightened = straighten_crop(tight_crop, direction)
                    file_name = f"{video_name}_f{frame_idx}_t{i}_dir{direction}.jpg"
                    cv2.imwrite(os.path.join(output_dir, file_name), straightened)
                    total_saved += 1

                    if total_saved >= MAX_CROPS_PER_VIDEO:
                        break

        frame_idx += 1

    cap.release()
    print(f"Done: {total_saved} crops saved.")
    return total_saved

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    tag_model = YOLO(TAG_MODEL_PATH)
    angle_model = YOLO(ANGLE_MODEL_PATH)

    video_extensions = [".mp4", ".avi", ".mov", ".mkv"]
    video_files = [
        f for f in os.listdir(VIDEO_DIR)
        if os.path.splitext(f)[1].lower() in video_extensions
    ]

    if not video_files:
        print(f"No videos found in {VIDEO_DIR}")
        return

    grand_total = 0
    for video_file in video_files:
        video_path = os.path.join(VIDEO_DIR, video_file)
        grand_total += process_video(tag_model, angle_model, video_path, OUTPUT_DIR)

    print(f"\nFinished. Total crops: {grand_total}")
    print(f"Saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()