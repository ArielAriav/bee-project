from ultralytics import YOLO
import cv2
import re
import easyocr

MODEL_PATH = r"runs\tag_detection\train\weights\best.pt"
VIDEO_PATH = r"data\raw\entrance\beeVideo2.mp4"

CONF = 0.4
PADDING = 6
COLOR = (0, 0, 255)  # red

# --- Performance knobs ---
OCR_EVERY_N_FRAMES = 4     # run OCR once every N frames (try 4-10)
OCR_SKIP_IF_CONF_GE = 0.95 # if we already have very confident number, OCR less often
UPSCALE_FX = 4             # was 6, try 3-4 to speed up
ANGLES = [0, -10, 10]      # was 7 angles, reduce to 3
USE_VARIANTS = True        # keep variants but fewer
DISPLAY_MIN_CONF = 0.70  # show number only if confidence >= 70%
LEN_LOCK = True
LEN_STRICT_CONF = 0.92      # allow different length only if very confident
LEN_STRICT_STREAK = 12      # and it persists

ANGLES_FAST = [0, -10, 10]
ANGLES_WIDE = [0, -30, -20, -10, 10, 20, 30, 45, -45]
WIDE_TRIGGER_CONF = 0.75  # below this confidence, try wide angles

# --- EasyOCR init (do this ONCE, outside the loop) ---
# gpu=True will use CUDA if your torch sees the GPU
reader = easyocr.Reader(['en'], gpu=True)

def center_crop(img, ratio=0.7):
    h, w = img.shape[:2]
    ch, cw = int(h * ratio), int(w * ratio)
    y1 = (h - ch) // 2
    x1 = (w - cw) // 2
    return img[y1:y1+ch, x1:x1+cw]


def deskew_binary(bin_img):
    """
    Estimate text angle from binary image and rotate to straighten.
    Works best when digits are dominant in the crop.
    """
    coords = cv2.findNonZero(255 - bin_img)  # focus on dark pixels
    if coords is None:
        return bin_img

    rect = cv2.minAreaRect(coords)
    angle = rect[-1]

    # Normalize angle
    if angle < -45:
        angle = 90 + angle

    return _rotate_image(bin_img, angle)


def _prep_variants(gray):
    """
    Create multiple preprocessing variants (color-agnostic) to handle
    different lighting / blur / black-white tags.
    Returns a list of grayscale images (uint8).
    """
    variants = [gray]

    # Otsu threshold + inverted
    _, thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(thr)
    variants.append(cv2.bitwise_not(thr))

    # Optional: CLAHE (can help, but costs time)
    if USE_VARIANTS:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        variants.append(clahe.apply(gray))

    return variants


def _rotate_image(img, angle_deg):
    """Rotate image around center without changing size too much."""
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def read_digits_easyocr(crop_bgr, reader, max_digits=3):
    """
    Robust digit reading with two-stage rotation search + deskew.
    Returns: (best_text, best_conf) or (None, 0.0)
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return None, 0.0

    crop = cv2.resize(
        crop_bgr, None,
        fx=UPSCALE_FX, fy=UPSCALE_FX,
        interpolation=cv2.INTER_CUBIC
    )

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    variants = _prep_variants(gray)

    best_text = None
    best_conf = 0.0

    def try_angles(angle_list, vars_list):
        nonlocal best_text, best_conf

        for v in vars_list:
            for ang in angle_list:
                img = _rotate_image(v, ang)

                results = reader.readtext(
                    img,
                    detail=1,
                    allowlist="0123456789",
                    paragraph=False
                )

                for _, text, conf in results:
                    text = text.strip()
                    m = re.findall(r"\d{1,%d}" % max_digits, text)
                    if not m:
                        continue

                    candidate = max(m, key=len)
                    c = float(conf)

                    if c > best_conf:
                        best_conf = c
                        best_text = candidate

                        if best_conf >= 0.98:
                            return True
        return False

    # Stage 1: fast angles
    if try_angles(ANGLES_FAST, variants):
        return best_text, best_conf

    # Stage 2: if confidence is low, deskew + wide angles
    if best_conf < WIDE_TRIGGER_CONF:
        deskewed = []
        for v in variants:
            deskewed.append(deskew_binary(v))

        try_angles(ANGLES_WIDE, deskewed)

    return best_text, best_conf


def blur_score(gray):
    # Higher is sharper, lower is blurrier
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def main():
    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("Error: cannot open video:", VIDEO_PATH)
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1:
        fps = 30  # fallback
    frame_delay_ms = int(1000 / fps)  # show at original speed
    SLOW_FACTOR = 1.5                 # 1.0 normal, 1.5 slower, 2.0 much slower
    frame_delay_ms = int(frame_delay_ms * SLOW_FACTOR)

    print("Running tag detection + number reading, press Q to quit")

    # --- Stabilization (anti-jumps) ---
    current_num = None            # number currently shown
    current_conf = 0.0
    frame_idx = 0

    candidate_num = None        # A new number trying to replace current_num
    candidate_streak = 0        # How many consecutive frames it appeared
    STREAK_THRESHOLD = 6        # Frames needed to accept a new number (tune)
    MIN_ACCEPT_CONF = 0.45      # Ignore OCR below this confidence (tune)

    while True:
        ret, frame = cap.read()
        frame_idx += 1

        if not ret:
            break

        results = model(frame, conf=CONF, verbose=False, device=0)
        annotated = frame.copy()

        boxes = getattr(results[0], "boxes", None)
        if boxes is not None and len(boxes) > 0:
            best = None

            for box in boxes:
                cls_id = int(box.cls[0])
                class_name = model.names[cls_id]
                box_conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                if class_name.lower() != "tag":
                    continue

                if best is None or box_conf > best["conf"]:
                    best = {"conf": box_conf, "bbox": (x1, y1, x2, y2), "class": class_name}

            if best is not None:
                x1, y1, x2, y2 = best["bbox"]
                h, w = frame.shape[:2]
                x1p = max(0, x1 - PADDING)
                y1p = max(0, y1 - PADDING)
                x2p = min(w - 1, x2 + PADDING)
                y2p = min(h - 1, y2 + PADDING)

                crop = frame[y1p:y2p, x1p:x2p]
                crop = center_crop(crop, 0.7)

                gray_small = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                
                # --- OCR scheduling ---
                # Run OCR frequently until we lock a number, then slow down
                if current_num is None:
                    run_ocr = (frame_idx % 2 == 0)  # fast lock at the beginning
                else:
                    run_ocr = (frame_idx % OCR_EVERY_N_FRAMES == 0)
                    if current_conf >= OCR_SKIP_IF_CONF_GE:
                        run_ocr = (frame_idx % (OCR_EVERY_N_FRAMES * 2) == 0)

                if run_ocr:
                    blur_thresh = 30 if current_num is None else 40
                    if blur_score(gray_small) < blur_thresh:
                        run_ocr = False

                did_ocr = False
                number, num_conf = None, 0.0

                if run_ocr:
                    did_ocr = True
                    number, num_conf = read_digits_easyocr(crop, reader, max_digits=3)

                # --- Stability gate ---
                # Update stability ONLY when we actually ran OCR
                if did_ocr:
                    # 1) Reject very low confidence reads
                    if number is None or num_conf < MIN_ACCEPT_CONF:
                        number = None

                    # 2) Length lock: prevents 38 -> 8 flips when one digit is lost
                    # If we already have a stable number, prefer keeping the same digit-count
                    if number is not None and LEN_LOCK and (current_num is not None):
                        if len(number) != len(current_num):
                            # Allow different length only if it's VERY confident
                            if num_conf < LEN_STRICT_CONF:
                                number = None

                    # 3) If this frame has no reliable number, do not update stability
                    if number is None:
                        candidate_num = None
                        candidate_streak = 0

                    else:
                        # 4) First stable number
                        if current_num is None:
                            current_num = number
                            current_conf = num_conf
                            candidate_num = None
                            candidate_streak = 0

                        else:
                            # 5) Same as current: keep it, maybe raise confidence
                            if number == current_num:
                                current_conf = max(current_conf, num_conf)
                                candidate_num = None
                                candidate_streak = 0

                            # 6) Different number: require consistency over time
                            else:
                                if candidate_num != number:
                                    candidate_num = number
                                    candidate_streak = 1
                                else:
                                    candidate_streak += 1

                            # If candidate has different digit-length than current,
                            # require a longer streak before switching
                            needed = STREAK_THRESHOLD
                            if candidate_num is not None and len(candidate_num) != len(current_num):
                                needed = max(needed, LEN_STRICT_STREAK)

                            if candidate_streak >= needed:
                                current_num = candidate_num
                                current_conf = num_conf
                                candidate_num = None
                                candidate_streak = 0
                # --- end stability gate ---

                # Draw bounding box
                cv2.rectangle(annotated, (x1p, y1p), (x2p, y2p), COLOR, 2)

                # Draw label
                label = f"{best['class']} {best['conf']:.2f}"
                if current_num is not None and current_conf >= DISPLAY_MIN_CONF:
                    label += f"  num={current_num} ({current_conf:.2f})"

                cv2.putText(
                    annotated,
                    label,
                    (x1p, max(20, y1p - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    COLOR,
                    2
                )

        cv2.imshow("Tag Detection + Number", annotated)
        if cv2.waitKey(frame_delay_ms) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
