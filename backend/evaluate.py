import cv2
import json
import os
import glob
from ultralytics import YOLO
from paddleocr import PaddleOCR
from processor import BeeProcessor
import config

def evaluate_folder(videos_folder, ground_truth_path):
    print("--- Loading Ground Truth ---")
    with open(ground_truth_path, 'r') as f:
        ground_truth = json.load(f)
        # נוודא שכל המספרים הם מטיפוס טקסט (String) למניעת באגים
        ground_truth = {k: str(v) for k, v in ground_truth.items()}
    
    print("--- Loading AI Models ---")
    yolo_model = YOLO(config.MODEL_PATH)
    paddle_model = PaddleOCR(
        lang="en",
        text_recognition_model_name="en_PP-OCRv5_mobile_rec",
        use_doc_orientation_classify=False,
        use_doc_unwarping=config.USE_DOC_UNWARPING,
        use_textline_orientation=config.USE_TEXTLINE_ORIENTATION,
        show_log=False
    )
    
    search_pattern = os.path.join(videos_folder, "*.mp4")
    video_files = glob.glob(search_pattern)
    
    if not video_files:
        print(f"No .mp4 files found in {videos_folder}")
        return

    global_stats = {}
    
    for video_path in video_files:
        video_name = os.path.basename(video_path)
        
        if video_name not in ground_truth:
            print(f"Skipping {video_name} - No target number found in JSON.")
            continue
            
        target_number = ground_truth[video_name]
        print(f"\n---> Processing Video: {video_name} | Target Number: {target_number} <---")
        
        processor = BeeProcessor(yolo_model, paddle_model)
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        
        # מדדים ייעודיים לסרטון הנוכחי
        video_stats = {"TP": 0, "FP": 0, "FN": 0}
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            processor.process_and_annotate(frame)
            frame_count += 1
            
            # איסוף כל התחזיות בפריים הנוכחי (מכל ה-IDs שיש עכשיו)
            frame_predictions = []
            has_active_bees = False
            
            for bee_id, bee_state in processor.bees.items():
                if bee_state.is_active:
                    has_active_bees = True
                    if bee_state.current_num:
                        frame_predictions.append(str(bee_state.current_num))
            
            # חישוב המדדים לפריים
            if has_active_bees:
                if target_number in frame_predictions:
                    video_stats["TP"] += 1
                    # אם המודל גם זיהה מספרים שגויים אחרים באותו פריים, נחשיב אותם כשגיאה
                    wrong_preds = [num for num in frame_predictions if num != target_number]
                    video_stats["FP"] += len(wrong_preds)
                else:
                    if len(frame_predictions) > 0:
                        # זיהינו מספרים, אבל אף אחד מהם הוא לא מספר המטרה
                        video_stats["FP"] += len(frame_predictions)
                    else:
                        # עקבנו אחרי דבורה, אבל לא הצלחנו לקרוא שום מספר
                        video_stats["FN"] += 1

            if frame_count % 100 == 0:
                print(f"  Processed {frame_count} frames...")
                
        cap.release()
        global_stats[video_name] = video_stats
        print(f"Finished {video_name}. TP:{video_stats['TP']} | FP:{video_stats['FP']} | FN:{video_stats['FN']}")

    print("\n--- All Videos Processed ---\n")
    calculate_video_metrics(global_stats)

def calculate_video_metrics(stats):
    if not stats:
        print("No evaluation data generated. Check your videos and ground_truth.json")
        return

    print("================ VIDEO-LEVEL REPORT ================")
    total_TP = total_FP = total_FN = 0
    
    for video_name, data in stats.items():
        tp = data["TP"]
        fp = data["FP"]
        fn = data["FN"]
        
        total_TP += tp
        total_FP += fp
        total_FN += fn
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"Video: {video_name}")
        print(f"  Total Frames Tracked: {tp + fp + fn}")
        print(f"  TP: {tp:^4} | FP: {fp:^4} | FN: {fn:^4}")
        print(f"  Precision: {precision:.2f} | Recall: {recall:.2f} | F1: {f1:.2f}\n")
        
    print("=========================================================")
    print("               OVERALL SYSTEM PERFORMANCE                ")
    print("=========================================================")
    
    macro_precision = total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else 0
    macro_recall = total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 0
    macro_f1 = 2 * (macro_precision * macro_recall) / (macro_precision + macro_recall) if (macro_precision + macro_recall) > 0 else 0
    
    print(f"Total Videos Evaluated: {len(stats)}")
    print(f"Global Precision:     {macro_precision:.2%}")
    print(f"Global Recall:        {macro_recall:.2%}")
    print(f"Global F1-Score:      {macro_f1:.2%}")
    print("=========================================================")

if __name__ == "__main__":
    VIDEOS_FOLDER = r"C:\Users\User1\Desktop\bee-project\backend\videos_test" 
    GT_FILE = "ground_truth.json"
    
    evaluate_folder(VIDEOS_FOLDER, GT_FILE)