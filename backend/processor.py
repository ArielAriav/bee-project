# backend/processor.py
import cv2
import re
import numpy as np
from collections import deque, Counter, defaultdict
import config

class BeeState:
    def __init__(self, track_id, initial_pos):
        self.original_id = track_id
        self.yolo_id = track_id
        self.last_center = initial_pos
        self.frames_lost = 0
        self.is_active = True
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
        self.bees = {} 
        self.MAX_LOST_FRAMES = 30 
        self.MAX_DIST_REID = 100

    def is_valid_crop(self, frame, x1, y1, x2, y2, conf):
        h, w = frame.shape[:2]
        margin = 5
        if x1 < margin or y1 < margin or x2 > w - margin or y2 > h - margin:
            return False
        if (x2 - x1) < 20 or (y2 - y1) < 20: 
            return False
        if conf < getattr(config, "OCR_CROP_CONF", 0.4):
            return False
        return True

    def calculate_distance(self, p1, p2):
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def rotate_image(self, img, angle):
        if angle == 90: return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        if angle == 180: return cv2.rotate(img, cv2.ROTATE_180)
        if angle == 270: return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return img

    def enhance_image_for_ocr(self, img_bgr):
        if img_bgr is None or img_bgr.size == 0: return None
        scale = getattr(config, "UPSCALE", 3)
        img_large = cv2.resize(img_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(img_large, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    def read_digits(self, crop_bgr):
        _digit_re = re.compile(r"\d+")
        if crop_bgr is None or crop_bgr.size == 0:
            return None, 0.0

        processed = self.enhance_image_for_ocr(crop_bgr)
        best_text, best_conf = None, 0.0

        for angle in [0, 90, 180, 270]:
            rotated = self.rotate_image(processed, angle)
            try:
                res = self.paddle_ocr.ocr(rotated, det=False, cls=False)
                if not res or not res[0]: continue

                for line in res[0]:
                    raw_text, conf = line[0], line[1]
                    t = str(raw_text).strip().replace(" ", "")
                    # Visual normalization
                    t = t.replace("I", "1").replace("l", "1").replace("|", "1")
                    t = t.replace("O", "0").replace("o", "0").replace("S", "5")

                    nums = _digit_re.findall(t)
                    if not nums: continue
                    
                    cand = "".join(nums)
                    if not (1 <= len(cand) <= getattr(config, "MAX_DIGITS", 3)): continue

                    if conf > best_conf:
                        best_conf = conf
                        best_text = cand
            except:
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

    def process_and_annotate(self, frame):
        self.frame_idx += 1
        annotated = frame.copy()
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
                
                # Re-ID and tracking logic
                found_bee = None
                for bee in self.bees.values():
                    if bee.yolo_id == yolo_id:
                        found_bee = bee
                        break
                
                if not found_bee:
                    possible_matches = [(self.calculate_distance((cx, cy), b.last_center), b) 
                                        for b in self.bees.values() if not b.is_active]
                    if possible_matches and min(possible_matches, key=lambda x: x[0])[0] < self.MAX_DIST_REID:
                        found_bee = min(possible_matches, key=lambda x: x[0])[1]
                    else:
                        found_bee = BeeState(yolo_id, (cx, cy))
                        self.bees[yolo_id] = found_bee

                found_bee.update_pos((cx, cy), yolo_id)
                
                # OCR Processing
                if self.is_valid_crop(frame, x1, y1, x2, y2, confs[i]):
                    if self.frame_idx % config.OCR_EVERY == 0:
                        p = config.PADDING
                        h_f, w_f = frame.shape[:2]
                        crop = frame[max(0, y1-p):min(h_f-1, y2+p), max(0, x1-p):min(w_f-1, x2+p)]
                        num, ocr_c = self.read_digits(crop)
                        
                        if num is not None and ocr_c >= config.MIN_ACCEPT_CONF:
                            # --- PROMOTION LOGIC ---
                            # If we see a longer number (e.g., '15' vs '5'), unlock to allow update
                            if found_bee.locked_digit and len(str(num)) > len(str(found_bee.locked_digit)):
                                found_bee.locked_digit = None
                                found_bee.recent_digits.clear()

                            # Anti-downgrade check
                            current_best = found_bee.locked_digit or found_bee.current_num
                            if not (current_best and len(str(num)) < len(str(current_best))):
                                found_bee.ocr_hist.append((num, ocr_c))
                                found_bee.recent_digits.append(num)
                                
                                # Locking check
                                if found_bee.locked_digit is None:
                                    counts = Counter(found_bee.recent_digits)
                                    top_num, count = counts.most_common(1)[0]
                                    if count >= getattr(config, "LOCK_COUNT", 8):
                                        found_bee.locked_digit = top_num

                            v_num, v_dom = self.vote_number(found_bee.ocr_hist)
                            if v_num:
                                found_bee.current_num, found_bee.current_conf = v_num, v_dom

                # Drawing
                rect_color = (0, 215, 255) if found_bee.locked_digit else config.COLOR 
                cv2.rectangle(annotated, (x1, y1), (x2, y2), rect_color, 2)
                display_num = found_bee.locked_digit or found_bee.current_num
                label = f"ID:{found_bee.original_id}" + (f" | #{display_num}" if display_num else "")
                cv2.putText(annotated, label, (x1, max(y1-10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, rect_color, 2)

        # Cleanup
        for bee in self.bees.values():
            if bee.yolo_id not in current_yolo_ids:
                bee.is_active = False
                bee.frames_lost += 1
        
        for k in list(self.bees.keys()):
            if self.bees[k].frames_lost > self.MAX_LOST_FRAMES and not self.bees[k].current_num:
                del self.bees[k]

        return annotated, None