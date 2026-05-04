# backend/batch_extract.py
import os
import glob
import cv2
from ultralytics import YOLO
from processor import BeeProcessor
import config

def process_all_videos():
    videos_dir = "videos"
    
    # חיפוש כל קבצי ה-mp4 בתיקייה (תומך גם באותיות גדולות וקטנות)
    video_files = glob.glob(os.path.join(videos_dir, "*.mp4")) + \
                  glob.glob(os.path.join(videos_dir, "*.MP4"))
    
    if not video_files:
        print(f"❌ לא נמצאו סרטונים בתיקייה: {videos_dir}")
        return

    print(f"📦 נמצאו {len(video_files)} סרטונים. טוען מודלים...")
    
    # טעינת המודלים פעם אחת בלבד כדי לחסוך זמן
    yolo_tag = YOLO(config.MODEL_PATH)
    yolo_digits = YOLO(config.DIGIT_MODEL_PATH)

    total_videos = len(video_files)

    for idx, video_path in enumerate(video_files, 1):
        video_name = os.path.basename(video_path)
        print(f"\n🎥 מתחיל עיבוד סרטון [{idx}/{total_videos}]: {video_name}")
        
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # חשוב מאוד: ניצור אובייקט BeeProcessor חדש עבור כל סרטון!
        # ככה מזהי המעקב (IDs) של הדבורים מתאפסים בין סרטון לסרטון
        processor = BeeProcessor(yolo_tag, yolo_digits)
        
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break # הסתיים הסרטון
                
            # שליחת הפריים לפרוססור.
            # הפרוססור כבר יזהה, יחתוך וישמור את התמונות ל-angleModelData לפי הקוד שכתבנו
            processor.process_and_annotate(frame)
            
            frame_count += 1
            
            # הדפסת התקדמות כל 100 פריימים כדי שלא יציף את הטרמינל
            if frame_count % 100 == 0:
                percent = (frame_count / total_frames) * 100 if total_frames > 0 else 0
                print(f"  ⏳ עובדו {frame_count}/{total_frames} פריימים ({percent:.1f}%)")

        cap.release()
        print(f"✅ סרטון {video_name} הסתיים בהצלחה!")

    print(f"\n🎉 כל הסרטונים עובדו! התמונות מחכות לך בתיקייה: {config.SAVE_CROPS_DIR}")

if __name__ == "__main__":
    process_all_videos()