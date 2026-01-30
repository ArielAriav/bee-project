import os
# Disable specific hardware acceleration flags that can sometimes cause crashes on certain systems
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_enable_pir_api"] = "0"

from collections import deque, defaultdict, Counter
from ultralytics import YOLO
from config import * # Importing variables like MODEL_PATH, VIDEO_PATH, etc.
import cv2
import re
import numpy as np
from paddleocr import PaddleOCR

# --- OCR INITIALIZATION ---
# Initialize PaddleOCR with English language.
# det=None: We disable the internal text detector because YOLO is already finding the 'tag'.
# rec_image_shape: Optimized for the mobile recognition model (faster and lighter).
paddle_ocr = PaddleOCR(
    lang="en",
    text_detection_model_name=None, 
    text_recognition_model_name="en_PP-OCRv5_mobile_rec",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    show_log=False
)

def rotate_image(img, angle):
    """
    Rotates the image by fixed 90-degree increments.
    Bees can be oriented in any direction, so we check 4 main angles to find upright text.
    """
    if angle == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    elif angle == 270:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img

def enhance_image_for_ocr(img_bgr):
    """
    Preprocesses the small bee tag image to make the number clearer for the AI.
    """
    if img_bgr is None or img_bgr.size == 0:
        return None
    
    # 1. Upscale: OCR models perform much better when the text is larger.
    scale = UPSCALE if 'UPSCALE' in globals() else 3
    img_large = cv2.resize(img_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    
    # 2. Grayscale: Removes color noise that might confuse the recognition model.
    gray = cv2.cvtColor(img_large, cv2.COLOR_BGR2GRAY)
    
    # 3. CLAHE: Improves contrast locally. Useful if the tag has glare or uneven lighting.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    
    # 4. Denoising: Smooths out pixel grain while keeping the edges of the numbers sharp.
    gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

def paddle_readtext(img):
    """
    The wrapper function to call PaddleOCR and safely extract the detected text and confidence score.
    """
    try:
        # We set det=False because we are providing the 'tag' crop directly.
        res = paddle_ocr.ocr(img, det=False, cls=False)
        
        if not res or not res[0]:
            return []

        results = []
        for line in res[0]:
            if isinstance(line, tuple) and len(line) == 2:
                text, conf = line
                results.append((str(text).strip(), float(conf)))
        return results
    except Exception as e:
        print(f"OCR Runtime Error: {e}")
        return []

def read_digits(crop_bgr):
    """
    The main logic for reading the number:
    1. Preprocess the tag.
    2. Try reading it at 0, 90, 180, and 270 degrees.
    3. Filter for valid numbers between 0-999.
    """
    _digit_re = re.compile(r"\d+") # Regex to find any sequence of digits

    if crop_bgr is None or crop_bgr.size == 0:
        return None, 0.0

    processed = enhance_image_for_ocr(crop_bgr)
    best_text, best_conf = None, 0.0
    
    # Check 4 orientations because bees walk in random directions
    for angle in [0, 90, 180, 270]:
        rotated = rotate_image(processed, angle)
        results = paddle_readtext(rotated)
        
        for text, conf in results:
            nums = _digit_re.findall(text)
            # Filter: We only care about sequences that are 1 to 3 digits long (0-999)
            valid_nums = [n for n in nums if 1 <= len(n) <= 3]
            
            if not valid_nums:
                continue
            
            # Pick the longest sequence (e.g., if it sees '12' and '120', pick '120')
            cand = max(valid_nums, key=len)
            
            # Keep the result that the AI is most confident about across all rotations
            if conf > best_conf:
                best_conf = conf
                best_text = cand
                
                # Performance optimization: if we are 98% sure, stop checking other rotations
                if best_conf >= 0.98:
                    return best_text, best_conf
                    
    return best_text, best_conf

def vote_number(hist):
    """
    Temporal Stabilization:
    Since the number might flicker in the video, we collect results from several 
    frames and choose the one with the highest total confidence.
    """
    if not hist:
        return None, 0.0
    weights = defaultdict(float)
    for num, conf in hist:
        weights[num] += conf
    total = sum(weights.values())
    if total == 0: return None, 0.0
    
    num, w = max(weights.items(), key=lambda x: x[1])
    dominance = w / total # Percentage of confidence this number has compared to others
    return (num, dominance) if dominance >= VOTE_DOMINANCE else (None, 0.0)

def main():
    # Load the YOLO model (for tag detection) and the video file
    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print(f"Error: Could not open video: {VIDEO_PATH}")
        return

    # --- STABILIZATION BUFFERS ---
    # ocr_hist: Stores (number, confidence) for weighted voting.
    # recent_digits: Stores last 30 raw numbers to find the most common one.
    ocr_hist = deque(maxlen=VOTE_HISTORY)
    recent_digits = deque(maxlen=30)
    locked_digit = None # Stores the final "stable" number once it's confirmed
    frame_idx = 0
    
    current_num, current_conf = None, 0.0

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_idx += 1

        # --- STEP 1: DETECTION ---
        # Run YOLO to find the 'tag' on the bee
        res = model(frame, conf=DET_CONF, verbose=False)
        boxes = getattr(res[0], "boxes", None)
        annotated = frame.copy()

        best_bbox = None
        best_det_conf = 0.0

        if boxes is not None:
            for b in boxes:
                # Ensure the detected object is actually a 'tag'
                if model.names[int(b.cls[0])].lower() == "tag":
                    conf = float(b.conf[0])
                    # If multiple tags are found, we focus on the most confident detection
                    if conf > best_det_conf:
                        best_det_conf = conf
                        best_bbox = tuple(map(int, b.xyxy[0]))

       # --- STEP 2: RECOGNITION (OCR) ---
        if best_bbox:
            x1, y1, x2, y2 = best_bbox
            h_f, w_f = frame.shape[:2]
            
            # Apply padding around the tag to give the OCR model a bit more context
            p = PADDING
            x1_p, y1_p = max(0, x1 - p), max(0, y1 - p)
            x2_p, y2_p = min(w_f - 1, x2 + p), min(h_f - 1, y2 + p)

            crop = frame[y1_p:y2_p, x1_p:x2_p]
            
            # Only run OCR every few frames (OCR_EVERY) to keep the processing fast
            if frame_idx % OCR_EVERY == 0 and crop.size > 0:
                num, ocr_c = read_digits(crop)
                
                if num is not None and ocr_c >= MIN_ACCEPT_CONF:
                    # --- NEW ANTI-DOWNGRADE RULE ---
                    # Define what is our current best guess (locked or current voted)
                    best_guess = locked_digit if locked_digit else current_num
                    
                    # If we already have a multi-digit number, ignore any shorter proposals
                    if best_guess and len(str(num)) < len(str(best_guess)):
                        pass # Ignore the shorter number
                    else:
                        # Process the result normally only if it's not shorter
                        ocr_hist.append((num, ocr_c))
                        recent_digits.append(num)

                        # LOCKING LOGIC: 
                        # If the same number appears 8 times in our buffer, we consider it 
                        # "Locked" and won't change it easily.
                        if locked_digit is None:
                            counts = Counter(recent_digits)
                            if counts:
                                most_common, count = counts.most_common(1)[0]
                                if count >= 8:
                                    locked_digit = most_common

                # Calculate the weighted "Winner" number from history
                voted_num, voted_dom = vote_number(ocr_hist)
                if voted_num:
                    current_num, current_conf = voted_num, voted_dom

            # --- STEP 3: VISUALIZATION ---
            # Draw the box around the tag
            cv2.rectangle(annotated, (x1, y1), (x2, y2), COLOR, 2)
            
            # Display either the 'Locked' number or the current 'Voted' number
            final_display_num = locked_digit if locked_digit else current_num
            label = f"Tag:{best_det_conf:.2f}"
            if final_display_num:
                label += f" | Num:{final_display_num} ({current_conf:.2f})"
            
            cv2.putText(annotated, label, (x1, max(y1 - 10, 20)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR, 2)

        # Display the result in a window
        cv2.imshow("Tag & Number Detection", annotated)
        # Press 'q' to stop the video processing early
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # --- STEP 4: CLEANUP & SUMMARY ---
    cap.release()
    cv2.destroyAllWindows()

    print("\n" + "="*40)
    print("       FINAL DETECTION SUMMARY")
    print("="*40)

    # Output the final results to the console for your report
    if locked_digit:
        print(f"FINAL GUESS: {locked_digit}")
        print(f"STATUS: Stable (locked by majority vote)")
    elif current_num:
        print(f"FINAL GUESS: {current_num}")
        print(f"STATUS: Estimated (based on recent frames)")
        print(f"CONFIDENCE: {current_conf:.2f}")
    else:
        print("FINAL GUESS: None")
        print("STATUS: No reliable number was detected during this run.")
    
    print("="*40 + "\n")

if __name__ == "__main__":
    main()