# backend/processor.py
import cv2
import re
import numpy as np
from collections import deque, Counter, defaultdict
from ultralytics import YOLO
import config

class BeeState:
    """
    Holds memory for a specific bee.
    Now includes position history for Re-ID logic.
    """
    def __init__(self, track_id, initial_pos):
        self.original_id = track_id # The ID we show the user
        self.yolo_id = track_id     # The current ID YOLO thinks it is
        
        # Position logic
        self.last_center = initial_pos
        self.frames_lost = 0
        self.is_active = True

        # OCR Logic
        self.recent_digits = deque(maxlen=30)
        self.ocr_hist = deque(maxlen=getattr(config, "VOTE_HISTORY", 12))
        self.locked_digit = None
        self.current_num = None
        self.current_conf = 0.0

    def update_pos(self, pos, yolo_id):
        self.last_center = pos
        self.yolo_id = yolo_id
        self.frames_lost = 0
        self.is_active = True

class BeeProcessor:
    def __init__(self, yolo_instance, paddle_instance):
        self.model = yolo_instance
        self.paddle_ocr = paddle_instance
        self.frame_idx = 0
        
        # Maps: Stable ID -> BeeState
        self.bees = {} 
        
        # Configuration for Re-ID
        self.MAX_LOST_FRAMES = 30  # Keep in memory for ~1 second (assuming 30fps)
        self.MAX_DIST_REID = 100   # Pixel distance to match a lost bee

    def reset_state(self):
        self.bees.clear()
        self.frame_idx = 0

    # --- QUALITY CONTROL HELPERS ---
    def is_valid_crop(self, frame, x1, y1, x2, y2, conf):
        """
        Filters out bad crops to save resources and improve accuracy.
        Based on the logic you requested.
        """
        h, w = frame.shape[:2]
        
        # 1. Edge Clearance: Reject if touching image boundaries
        margin = 5
        if x1 < margin or y1 < margin or x2 > w - margin or y2 > h - margin:
            return False

        # 2. Size Check: Reject tiny boxes (noise)
        box_w, box_h = x2 - x1, y2 - y1
        if box_w < 20 or box_h < 20: 
            return False

        # 3. Confidence Check: Only accept high confidence detections for OCR
        if conf < 0.6: # Configurable threshold
            return False

        return True

    def calculate_distance(self, p1, p2):
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    # --- OCR HELPERS (UNCHANGED) ---
    def rotate_image(self, img, angle):
        if angle == 90: return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        if angle == 180: return cv2.rotate(img, cv2.ROTATE_180)
        if angle == 270: return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return img

    def enhance_image_for_ocr(self, img_bgr):
        if img_bgr is None or img_bgr.size == 0: return None
        scale = getattr(config, "UPSCALE", 4)
        img_large = cv2.resize(img_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(img_large, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    def read_digits(self, crop_bgr):
        _digit_re = re.compile(r"\d+")
        if crop_bgr is None or crop_bgr.size == 0:
            return None, 0.0

        # Preprocess image for OCR
        processed = self.enhance_image_for_ocr(crop_bgr)

        best_text, best_conf = None, 0.0

        # Try different rotations
        for angle in [0, 90, 180, 270]:
            rotated = self.rotate_image(processed, angle)

            try:
                res = self.paddle_ocr.ocr(rotated, det=False, cls=False)
                if not res or not res[0]:
                    continue

                for line in res[0]:
                    raw_text, conf = line[0], line[1]

                    # Normalize common OCR confusions
                    t = str(raw_text).strip()

                    # Characters often confused with "1"
                    t = (
                        t.replace("I", "1")
                        .replace("l", "1")
                        .replace("|", "1")
                        .replace("!", "1")
                    )

                    # Optional fixes
                    t = (
                        t.replace("O", "0")
                        .replace("o", "0")
                        .replace("S", "5")
                    )

                    nums = _digit_re.findall(t)

                    valid_nums = [n for n in nums if 1 <= len(n) <= 3]
                    if not valid_nums:
                        continue

                    # Prefer longer numbers (e.g., 15 over 5)
                    cand = max(valid_nums, key=len)

                    if conf > best_conf:
                        best_conf = conf
                        best_text = cand

                        if best_conf >= 0.98:
                            return best_text, best_conf

            except Exception as e:
                continue

        return best_text, best_conf

    def vote_number(self, hist):
        if not hist: return None, 0.0
        weights = defaultdict(float)
        for num, conf in hist: weights[num] += conf
        total = sum(weights.values())
        if total == 0: return None, 0.0
        num, w = max(weights.items(), key=lambda x: x[1])
        dominance = w / total
        if dominance >= getattr(config, "VOTE_DOMINANCE", 0.6):
            return num, dominance
        return None, 0.0

    # --- MAIN PROCESSING LOOP ---
    def process_and_annotate(self, frame):
        self.frame_idx += 1
        annotated = frame.copy()
        
        # 1. Run YOLO Tracking
        results = self.model.track(frame, conf=config.DET_CONF, persist=True, verbose=False)
        
        current_yolo_ids = set()
        
        if results and results[0].boxes and results[0].boxes.id is not None:
            boxes = results[0].boxes
            ids = boxes.id.cpu().numpy().astype(int)
            coords = boxes.xyxy.cpu().numpy().astype(int)
            confs = boxes.conf.cpu().numpy()

            for i, yolo_id in enumerate(ids):
                current_yolo_ids.add(yolo_id)
                x1, y1, x2, y2 = coords[i]
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                det_conf = confs[i]

                # --- RE-IDENTIFICATION LOGIC (Fixing lost IDs) ---
                # Check if this YOLO ID is already mapped to a known bee
                found_bee = None
                
                # Case A: We already know this YOLO ID
                for bee in self.bees.values():
                    if bee.yolo_id == yolo_id:
                        found_bee = bee
                        break
                
                # Case B: New YOLO ID -> Check if it's actually an old lost bee
                if not found_bee:
                    possible_matches = []
                    for bee in self.bees.values():
                        # Only check bees that are currently "lost" (not tracked by current YOLO ID)
                        if not bee.is_active:
                            dist = self.calculate_distance((cx, cy), bee.last_center)
                            if dist < self.MAX_DIST_REID:
                                possible_matches.append((dist, bee))
                    
                    if possible_matches:
                        # Match with the closest lost bee
                        possible_matches.sort(key=lambda x: x[0])
                        found_bee = possible_matches[0][1]
                        # Merge: Update the old bee with the new YOLO ID
                        print(f"🔄 Re-ID: Merged new ID {yolo_id} into stable ID {found_bee.original_id}")
                    else:
                        # Truly a new bee
                        found_bee = BeeState(yolo_id, (cx, cy))
                        self.bees[yolo_id] = found_bee

                # Update the bee's status
                found_bee.update_pos((cx, cy), yolo_id)
                
                # --- VISUALIZATION & OCR ---
                # Default Color
                COLOR = getattr(config, "COLOR", (0, 0, 255))
                rect_color = (0, 215, 255) if found_bee.locked_digit else COLOR 
                
                # Draw Box
                cv2.rectangle(annotated, (x1, y1), (x2, y2), rect_color, 2)

                # --- SMART CROP SELECTION ---
                # Only run OCR if the crop is high quality
                if self.is_valid_crop(frame, x1, y1, x2, y2, det_conf):
                    # Only run OCR every N frames to save FPS
                    if self.frame_idx % config.OCR_EVERY == 0:
                        p = config.PADDING
                        h_f, w_f = frame.shape[:2]
                        x1_p, y1_p = max(0, x1 - p), max(0, y1 - p)
                        x2_p, y2_p = min(w_f - 1, x2 + p), min(h_f - 1, y2 + p)
                        crop = frame[y1_p:y2_p, x1_p:x2_p]

                        num, ocr_c = self.read_digits(crop)
                        
                        if num is not None and ocr_c >= config.MIN_ACCEPT_CONF:
                            best_guess = found_bee.locked_digit if found_bee.locked_digit else found_bee.current_num
                            
                            # Anti-downgrade logic
                            if not (best_guess and len(str(num)) < len(str(best_guess))):
                                found_bee.ocr_hist.append((num, ocr_c))
                                found_bee.recent_digits.append(num)
                                
                                if found_bee.locked_digit is None:
                                    counts = Counter(found_bee.recent_digits)
                                    if counts and counts.most_common(1)[0][1] >= 8:
                                        found_bee.locked_digit = counts.most_common(1)[0][0]

                            voted_num, voted_dom = self.vote_number(found_bee.ocr_hist)
                            if voted_num:
                                found_bee.current_num, found_bee.current_conf = voted_num, voted_dom

                # Draw Label
                display_num = found_bee.locked_digit if found_bee.locked_digit else found_bee.current_num
                label = f"ID:{found_bee.original_id}"
                if display_num:
                    label += f" | #{display_num}"
                
                cv2.putText(annotated, label, (x1, max(y1 - 10, 20)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, rect_color, 2)

        # --- CLEANUP LOST TRACKS ---
        # Mark bees not found in this frame as inactive
        for bee in self.bees.values():
            if bee.yolo_id not in current_yolo_ids:
                bee.is_active = False
                bee.frames_lost += 1

        # Optional: Remove bees lost for too long (to keep memory clean)
        # We copy keys to avoid runtime error during deletion
        active_keys = list(self.bees.keys())
        for key in active_keys:
            if self.bees[key].frames_lost > self.MAX_LOST_FRAMES:
                # Only remove if it never found a number (noise), 
                # otherwise keep it in history for final report
                if not self.bees[key].current_num: 
                    del self.bees[key]

        return annotated, None