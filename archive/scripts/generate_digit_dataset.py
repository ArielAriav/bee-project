import cv2
import os
from ultralytics import YOLO

# Path configuration (Internal paths kept as requested)
MODEL_PATH = "backend/models/tag_detection/best.pt"
VIDEO_PATH = "data/raw/entrance/beeVideo16.mp4" 
OUTPUT_DIR = "archive/datasets/digit_recognition/raw_crops/beeVideo16"

# Logic Parameters
MAX_IMAGES_PER_VIDEO = 500  # Hard limit to prevent massive datasets from long videos
FRAME_SKIP = 1  # Process 1 frame every 0.5 seconds (at 30fps) to ensure movement
MAX_TAGS_PER_FRAME = 3      # Don't take too many tags from a single frame to ensure time diversity
CONF_THRESHOLD = 0.65       # Balanced confidence for digit quality

def generate_digit_dataset():
    # Folder logic: Create if not exists
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        print(f"Directory created/verified: {OUTPUT_DIR}")

    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(VIDEO_PATH)
    
    if not cap.isOpened():
        print(f"Error: Could not open video at {VIDEO_PATH}")
        return

    total_saved = 0
    frame_idx = 0

    print(f"Processing: {VIDEO_PATH}")
    print(f"Settings: Max {MAX_IMAGES_PER_VIDEO} crops, Skipping {FRAME_SKIP} frames.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Stop if we reached the video quota
        if total_saved >= MAX_IMAGES_PER_VIDEO:
            print(f"Reached quota of {MAX_IMAGES_PER_VIDEO} images. Stopping.")
            break

        # Dynamic sampling logic
        if frame_idx % FRAME_SKIP == 0:
            # Detection only (avoiding track warnings)
            results = model(frame, conf=CONF_THRESHOLD, verbose=False)

            for result in results:
                boxes = result.boxes.xyxy.cpu().numpy()
                
                # Limit how many tags we take from this specific moment
                tags_to_take = boxes[:MAX_TAGS_PER_FRAME] 

                for i, box in enumerate(tags_to_take):
                    if total_saved >= MAX_IMAGES_PER_VIDEO:
                        break
                        
                    x1, y1, x2, y2 = map(int, box)
                    crop = frame[y1:y2, x1:x2]
                    
                    if crop.size > 0:
                        file_name = f"f{frame_idx}_t{i}.jpg"
                        cv2.imwrite(os.path.join(OUTPUT_DIR, file_name), crop)
                        total_saved += 1

        frame_idx += 1
        if total_saved > 0 and total_saved % 50 == 0:
            print(f"Status: Collected {total_saved}/{MAX_IMAGES_PER_VIDEO} crops...")

    cap.release()
    print(f"Done! Final count for this video: {total_saved}")

if __name__ == "__main__":
    generate_digit_dataset()