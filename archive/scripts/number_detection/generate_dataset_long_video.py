import cv2
import os
from ultralytics import YOLO

# Path configuration
MODEL_PATH = "backend/models/tag_detection/best.pt"
VIDEO_PATH = "data/raw/entrance/video_for_valid.mp4" 
OUTPUT_DIR = "archive/datasets/digit_recognition/raw_crops/video_for_valid"

# Logic Parameters
TOTAL_MAX = 75           # Absolute limit of images
TOTAL_MIN = 50            # Target minimum
IMAGES_PER_BEE = 20         # Take multiple frames per unique bee to hit the target
SAVE_INTERVAL = 15          # Frames to wait between saving the same bee (approx 0.5s)
FRAME_SKIP = 2              # Scan every 2nd frame to ensure no sticker is missed
CONF_THRESHOLD = 0.5        # Confidence threshold for detection

def generate_dataset_long_video():
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load the model and video
    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(VIDEO_PATH)
    
    if not cap.isOpened():
        print(f"Error: Could not open {VIDEO_PATH}")
        return

    total_saved = 0
    bee_history = {} # Tracks: {track_id: {'count': 0, 'last_save': frame_index}}
    frame_idx = 0

    print(f"Starting extraction from: {VIDEO_PATH}")
    print(f"Targeting {TOTAL_MIN}-{TOTAL_MAX} images...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Stop immediately if we reached the maximum limit
        if total_saved >= TOTAL_MAX:
            break

        # Scan every X frames to ensure we don't miss new stickers
        if frame_idx % FRAME_SKIP == 0:
            # Using ByteTrack to avoid motion compensation warnings
            results = model.track(frame, persist=True, tracker="bytetrack.yaml", conf=CONF_THRESHOLD, verbose=False)

            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.int().cpu().numpy()

                for box, track_id in zip(boxes, track_ids):
                    if total_saved >= TOTAL_MAX:
                        break
                    
                    # Initialize tracking for new bee IDs
                    if track_id not in bee_history:
                        bee_history[track_id] = {'count': 0, 'last_save': -SAVE_INTERVAL}
                    
                    bee = bee_history[track_id]
                    
                    # Logic: Save if we haven't reached the per-bee limit AND enough frames passed for a new shot
                    if bee['count'] < IMAGES_PER_BEE:
                        if (frame_idx - bee['last_save']) >= SAVE_INTERVAL:
                            x1, y1, x2, y2 = map(int, box)
                            crop = frame[y1:y2, x1:x2]
                            
                            if crop.size > 0:
                                file_name = f"bee{track_id}_f{frame_idx}.jpg"
                                cv2.imwrite(os.path.join(OUTPUT_DIR, file_name), crop)
                                
                                bee['count'] += 1
                                bee['last_save'] = frame_idx
                                total_saved += 1

        frame_idx += 1
        if total_saved > 0 and total_saved % 20 == 0:
            print(f"Status: {total_saved} images collected...")

    cap.release()
    print("-" * 30)
    print(f"Extraction complete. Total images saved: {total_saved}")
    print(f"Total unique bees identified: {len(bee_history)}")
    
    if total_saved < TOTAL_MIN:
        print(f"Warning: Only {total_saved} images collected. The video may have very few bees.")
    print("-" * 30)

if __name__ == "__main__":
    generate_dataset_long_video()