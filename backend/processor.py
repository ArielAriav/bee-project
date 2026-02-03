# backend/processor.py
import cv2
import re
import numpy as np
import os
from collections import deque, Counter, defaultdict
from ultralytics import YOLO
from paddleocr import PaddleOCR
import config

class BeeProcessor:
    def __init__(self, yolo_instance, paddle_instance):
        """
        Modified __init__ to accept pre-loaded models.
        This allows us to destroy and recreate this class instantly 
        on every 'New Scan' without reloading heavy AI models.
        """
        # We use the models passed from main.py
        self.model = yolo_instance
        self.paddle_ocr = paddle_instance
        
        # --- PRESERVED LOGIC VARIABLES ---
        self.recent_digits = deque(maxlen=30)
        # Using getattr to be safe if config is missing keys, but defaults match your script
        self.ocr_hist = deque(maxlen=getattr(config, "VOTE_HISTORY", 12))
        self.locked_digit = None
        self.frame_idx = 0
        self.current_num = None
        self.current_conf = 0.0

    def reset_state(self):
        """Resets all detection memory for a fresh start."""
        self.recent_digits.clear()
        self.ocr_hist.clear()
        self.locked_digit = None
        self.frame_idx = 0
        self.current_num = None
        self.current_conf = 0.0

    def rotate_image(self, img, angle):
        """Standard rotations as defined in the script."""
        if angle == 90: return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        if angle == 180: return cv2.rotate(img, cv2.ROTATE_180)
        if angle == 270: return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return img

    def enhance_image_for_ocr(self, img_bgr):
        """1:1 Script Enhancement (Upscale + CLAHE + Denoising)."""
        if img_bgr is None or img_bgr.size == 0: return None
        scale = getattr(config, "UPSCALE", 4)
        img_large = cv2.resize(img_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(img_large, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    def paddle_readtext(self, img):
        """1:1 Paddle wrapper from script."""
        try:
            res = self.paddle_ocr.ocr(img, det=False, cls=False)
            if not res or not res[0]: return []
            results = []
            for line in res[0]:
                if isinstance(line, tuple) and len(line) == 2:
                    text, conf = line
                    results.append((str(text).strip(), float(conf)))
            return results
        except Exception:
            return []

    def read_digits(self, crop_bgr):
        """1:1 Number detection logic with 4 orientations."""
        _digit_re = re.compile(r"\d+")
        if crop_bgr is None or crop_bgr.size == 0: return None, 0.0
        processed = self.enhance_image_for_ocr(crop_bgr)
        best_text, best_conf = None, 0.0
        
        for angle in [0, 90, 180, 270]:
            rotated = self.rotate_image(processed, angle)
            results = self.paddle_readtext(rotated)
            for text, conf in results:
                nums = _digit_re.findall(text)
                valid_nums = [n for n in nums if 1 <= len(n) <= 3]
                if not valid_nums: continue
                cand = max(valid_nums, key=len)
                if conf > best_conf:
                    best_conf = conf
                    best_text = cand
                    if best_conf >= 0.98: return best_text, best_conf
        return best_text, best_conf

    def vote_number(self, hist):
        """1:1 Weighted Voting logic."""
        if not hist: return None, 0.0
        weights = defaultdict(float)
        for num, conf in hist: weights[num] += conf
        total = sum(weights.values())
        if total == 0: return None, 0.0
        num, w = max(weights.items(), key=lambda x: x[1])
        dominance = w / total
        vote_dominance = getattr(config, "VOTE_DOMINANCE", 0.6)
        return (num, dominance) if dominance >= vote_dominance else (None, 0.0)

    def process_and_annotate(self, frame):
        """Main processing loop 1:1 with detect_tag_and_number.py."""
        self.frame_idx += 1
        annotated = frame.copy()
        
        # Detection
        res = self.model(frame, conf=config.DET_CONF, verbose=False)
        boxes = getattr(res[0], "boxes", None)
        best_bbox = None
        best_det_conf = 0.0

        if boxes is not None:
            for b in boxes:
                if self.model.names[int(b.cls[0])].lower() == "tag":
                    conf = float(b.conf[0])
                    if conf > best_det_conf:
                        best_det_conf = conf
                        best_bbox = tuple(map(int, b.xyxy[0]))

        if best_bbox:
            x1, y1, x2, y2 = best_bbox
            h_f, w_f = frame.shape[:2]
            p = config.PADDING
            x1_p, y1_p = max(0, x1 - p), max(0, y1 - p)
            x2_p, y2_p = min(w_f - 1, x2 + p), min(h_f - 1, y2 + p)
            crop = frame[y1_p:y2_p, x1_p:x2_p]

            if self.frame_idx % config.OCR_EVERY == 0 and crop.size > 0:
                num, ocr_c = self.read_digits(crop)
                if num is not None and ocr_c >= config.MIN_ACCEPT_CONF:
                    best_guess = self.locked_digit if self.locked_digit else self.current_num
                    # 1:1 Anti-downgrade logic
                    if not (best_guess and len(str(num)) < len(str(best_guess))):
                        self.ocr_hist.append((num, ocr_c))
                        self.recent_digits.append(num)
                        
                        if self.locked_digit is None:
                            counts = Counter(self.recent_digits)
                            if counts and counts.most_common(1)[0][1] >= 8:
                                self.locked_digit = counts.most_common(1)[0][0]

                voted_num, voted_dom = self.vote_number(self.ocr_hist)
                if voted_num:
                    self.current_num, self.current_conf = voted_num, voted_dom

            # 1:1 Visualization
            COLOR = getattr(config, "COLOR", (0, 0, 255))
            cv2.rectangle(annotated, (x1, y1), (x2, y2), COLOR, 2)
            final_display_num = self.locked_digit if self.locked_digit else self.current_num
            label = f"Tag:{best_det_conf:.2f}"
            if final_display_num:
                label += f" | Num:{final_display_num} ({self.current_conf:.2f})"
            cv2.putText(annotated, label, (x1, max(y1 - 10, 20)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR, 2)

        return annotated, {"id": self.locked_digit, "is_locked": self.locked_digit is not None}