from collections import deque, defaultdict, Counter
from ultralytics import YOLO
from config import *
import cv2
import re
import easyocr

reader = easyocr.Reader(["en"], gpu=True)

def center_crop(img, ratio):
    """Crop the center region of the image using the given ratio."""
    h, w = img.shape[:2]
    ch, cw = int(h * ratio), int(w * ratio)
    y1 = (h - ch) // 2
    x1 = (w - cw) // 2
    return img[y1:y1 + ch, x1:x1 + cw]

def blur_score(gray):
    """Calculate Laplacian variance as a measure of image sharpness (blur score)."""
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def rotate(img, ang):
    """Rotate the input image by a given angle around its center."""
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

def deskew_gray(gray):
    """Auto-correct skew in a grayscale image using edge detection and minimum bounding rectangle."""
    edges = cv2.Canny(gray, 50, 150)
    pts = cv2.findNonZero(edges)
    if pts is None:
        return gray
    rect = cv2.minAreaRect(pts)
    angle = rect[-1]
    if angle < -45:
        angle = 90 + angle
    return rotate(gray, angle)

def sharpen(gray):
    """Apply unsharp masking to enhance edges in a grayscale image."""
    blur = cv2.GaussianBlur(gray, (0, 0), 1.0)
    return cv2.addWeighted(gray, 1.6, blur, -0.6, 0)

def prep_variants(gray):
    """Generate sharpened and thresholded variants of a grayscale image for OCR."""
    g = sharpen(gray)
    _, thr = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return (g, thr, cv2.bitwise_not(thr))

def read_digits(crop_bgr):
    """
    Run OCR on a cropped BGR image to detect up to 3-digit numbers (0–999).
    Tries multiple variants and angles to improve recognition accuracy.
    Returns: (number_string, confidence)
    """
    _digit_re = re.compile(r"\d{1,3}")

    if crop_bgr is None or crop_bgr.size == 0:
        return None, 0.0

    crop = cv2.resize(crop_bgr, None, fx=UPSCALE, fy=UPSCALE, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = deskew_gray(gray)

    best_text, best_conf = None, 0.0
    for v in prep_variants(gray):
        # Try different image variants and angles for OCR robustness
        for a in ANGLES:
            img = rotate(v, a) if a else v
            for _, text, conf in reader.readtext(img, **OCR_PARAMS):
                m = _digit_re.findall(text.strip())
                if not m:
                    continue
                cand = max(m, key=len)
                c = float(conf)
                if c > best_conf:
                    best_text, best_conf = cand, c
                    if best_conf >= 0.98:
                        return best_text, best_conf
                    
    return best_text, best_conf

def vote_number(hist):
    """
    Perform weighted voting over recent OCR results to stabilize predictions.
    Returns: (dominant_number, confidence) if dominant enough, else (None, 0.0)
    """
    if not hist:
        return None, 0.0
    weights = defaultdict(float)
    for num, conf in hist:
        weights[num] += conf
    total = sum(weights.values())
    num, w = max(weights.items(), key=lambda x: x[1])
    dom = w / total if total > 0 else 0.0
    return (num, w / total) if dom >= VOTE_DOMINANCE else (None, 0.0)

def main():
    """Main function: loads video, runs detection and OCR pipeline, and displays results frame-by-frame."""
    # === Load model and video ===
    final_number = None
    final_conf = 0.0
    recent_digits = deque(maxlen=30)  # Saves the last 30 readings
    locked_digit = None # Lock number if repeated enough times across frames (stabilization heuristic)
    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(VIDEO_PATH)

    # === Frame processing setup ===
    if not cap.isOpened():
        print("Error: cannot open video:", VIDEO_PATH)
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    delay_ms = int((1000 / fps) * 1.5)

    frame_idx = 0
    ocr_hist = deque(maxlen=VOTE_HISTORY)
    crop_buf = deque(maxlen=6)  # to improve blur: choose the sharpest

    current_num, current_conf = None, 0.0

    while True:
        # === Read next frame ===
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # === Run YOLO tag detection ===
        res = model(frame, conf=DET_CONF, verbose=False, device=0)
        boxes = getattr(res[0], "boxes", None)
        annotated = frame.copy()

        best_bbox, best_det_conf = None, 0.0
        if boxes is not None and len(boxes) > 0:
            for b in boxes:
                cls_id = int(b.cls[0])
                name = model.names[cls_id].lower()
                if name != "tag":
                    continue
                c = float(b.conf[0])
                if c > best_det_conf:
                    best_det_conf = c
                    best_bbox = tuple(map(int, b.xyxy[0]))

        if best_bbox:
            # Prepare crop and decide whether to run OCR on this frame
            x1, y1, x2, y2 = best_bbox
            h, w = frame.shape[:2]
            x1 = max(0, x1 - PADDING); y1 = max(0, y1 - PADDING)
            x2 = min(w - 1, x2 + PADDING); y2 = min(h - 1, y2 + PADDING)

            crop = frame[y1:y2, x1:x2]
            crop = center_crop(crop, CROP_CENTER_RATIO)
            gray_small = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            crop_buf.append(crop)

            if current_num is None:
                run_ocr = (frame_idx % 2 == 0)
            elif current_conf >= SKIP_IF_CONF_GE:
                run_ocr = (frame_idx % (OCR_EVERY * 2) == 0)
            else:
                run_ocr = (frame_idx % OCR_EVERY == 0)

            # Blur filter
            if run_ocr:
                # === Prepare crop and blur check ===
                blur_min = BLUR_MIN_FIND if current_num is None else BLUR_MIN_LOCK
                if blur_score(gray_small) < blur_min:
                    run_ocr = False

            if run_ocr and crop_buf:
                best_crop = max(crop_buf, key=lambda c: blur_score(cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)))
                num, conf = read_digits(best_crop)
                if num is not None:
                    recent_digits.append(num)
                    if locked_digit is None:
                        counts = Counter(recent_digits)
                        most_common, count = counts.most_common(1)[0]
                        if count >= 20:
                            locked_digit = most_common

                # Save valid OCR result to history for voting
                if num is not None and conf >= MIN_ACCEPT_CONF:
                    ocr_hist.append((num, conf))

                voted_num, voted_conf = vote_number(ocr_hist)
                if voted_num is not None:
                    current_num, current_conf = voted_num, max(current_conf, voted_conf)
                    final_number = current_num
                    final_conf = current_conf

            # === Draw bounding box and label ===
            cv2.rectangle(annotated, (x1, y1), (x2, y2), COLOR, 2)
            label = f"tag {best_det_conf:.2f}"
            if current_num is not None and current_conf >= DISPLAY_MIN_CONF:
                shown_num = locked_digit if locked_digit is not None else current_num
                label += f"  num={shown_num} ({current_conf:.2f})"

            cv2.putText(annotated, label, (x1, max(20, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR, 2)

        cv2.imshow("Tag + Number Detection", annotated)
        if cv2.waitKey(delay_ms) & 0xFF == ord("q"):
            break

    # === Final result output ===
    print("\n=== FINAL RESULT ===")
   
    if locked_digit is not None:
        print(f"Detected number: {locked_digit} (locked by majority)")
    elif final_number is not None:
        print(f"Detected number: {final_number} (confidence ~ {final_conf:.2f})")
    else:
        print("No stable number detected")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
