# backend/batch_extract.py
import os
import glob
import cv2
from ultralytics import YOLO
from processor import BeeProcessor
import config

def process_all_videos():
    videos_dir = "videos"
    
    # Search for all mp4 files in the directory (supports both upper and lowercase extensions)
    video_files = glob.glob(os.path.join(videos_dir, "*.mp4")) + \
                  glob.glob(os.path.join(videos_dir, "*.MP4"))

    if not video_files:
        print(f"❌ No videos found in directory: {videos_dir}")
        return

    print(f"📦 Found {len(video_files)} videos. Loading models...")

    # Load models only once to save time
    yolo_tag = YOLO(config.MODEL_PATH)
    yolo_digits = YOLO(config.DIGIT_MODEL_PATH)

    total_videos = len(video_files)

    for idx, video_path in enumerate(video_files, 1):
        video_name = os.path.basename(video_path)
        print(f"\n🎥 Starting processing video [{idx}/{total_videos}]: {video_name}")

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Important: create a new BeeProcessor object for each video
        # so that bee tracking IDs reset between videos
        processor = BeeProcessor(yolo_tag, yolo_digits)

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break # end of video

            # Send the frame to the processor.
            # The processor will detect, crop and save images to angleModelData
            processor.process_and_annotate(frame)

            frame_count += 1

            # Print progress every 100 frames to avoid flooding the terminal
            if frame_count % 100 == 0:
                percent = (frame_count / total_frames) * 100 if total_frames > 0 else 0
                print(f"  ⏳ Processed {frame_count}/{total_frames} frames ({percent:.1f}%)")

        cap.release()
        print(f"✅ Video {video_name} finished successfully!")

    print(f"\n🎉 All videos processed! Images are waiting in: {config.SAVE_CROPS_DIR}")

if __name__ == "__main__":
    process_all_videos()