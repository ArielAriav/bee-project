# backend/processor.py
import cv2
import re
import numpy as np
from collections import deque, Counter
import config

class BeeState:
    """
    Internal state for a single bee tracking instance.
    """
    def __init__(self, track_id, initial_pos):
        self.original_id = track_id
        self.yolo_id = track_id
        self.last_center = initial_pos
        self.frames_lost = 0
        self.is_active = True
        self.recent_digits = deque(maxlen=30)
        self.locked_digit = None
        self.current_num = None
        self.current_conf = 0.0

    def update_pos(self, pos, yolo_id):
        self.last_center = pos
        self.yolo_id = yolo_id
        self.frames_lost = 0
        self.is_active = True

class BeeProcessor:
    def __init__(self, yolo_instance, digit_model_instance):
        """
        Initializes models. model = tag detection, digit_model = digit recognition.
        """
        self.model = yolo_instance
        self.digit_model = digit_model_instance
        self.frame_idx = 0
        self.bees = {} 

    def read_digits(self, crop):
        """
        Uses the second YOLO model to identify digits within the tag crop.
        """
        if crop is None or crop.size == 0:
            return None, 0.0

        results = self.digit_model.predict(crop, conf=0.25, verbose=False)
        detected = []
        
        if results and len(results[0].boxes) > 0:
            for box in results[0].boxes:
                # Save (X-coordinate, Digit Class, Confidence)
                detected.append((box.xyxy[0][0].item(), int(box.cls[0]), box.conf[0].item()))

            # Sort by X-axis to maintain digit sequence (0-999)
            detected.sort(key=lambda x: x[0])
            digit_str = "".join([str(d[1]) for d in detected])
            avg_conf = sum([d[2] for d in detected]) / len(detected)
            return digit_str, avg_conf
            
        return None, 0.0

    def process_and_annotate(self, frame):
        """
        Main logic: Detect tags, then detect digits inside each tag.
        """
        self.frame_idx += 1
        annotated = frame.copy()
        
        # 1. Track tag location
        results = self.model.track(frame, conf=config.DET_CONF, persist=True, verbose=False)
        current_yolo_ids = set()

        if results and results[0].boxes and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            ids = results[0].boxes.id.cpu().numpy().astype(int)
            
            for box, track_id in zip(boxes, ids):
                current_yolo_ids.add(track_id)
                x1, y1, x2, y2 = box
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                
                # Bee Lifecycle Management
                if track_id not in self.bees:
                    self.bees[track_id] = BeeState(track_id, (cx, cy))
                
                bee = self.bees[track_id]
                bee.update_pos((cx, cy), track_id)
                
                # 2. Digit Detection (every N frames)
                if self.frame_idx % config.OCR_EVERY == 0:
                    crop = frame[max(0,y1):min(frame.shape[0],y2), max(0,x1):min(frame.shape[1],x2)]
                    res_str, res_conf = self.read_digits(crop)
                    
                    if res_str and 1 <= len(res_str) <= config.MAX_DIGITS:
                        bee.recent_digits.append(res_str)
                        
                        # Number Confirmation (Locking)
                        counts = Counter(bee.recent_digits)
                        top_num, freq = counts.most_common(1)[0]
                        if freq >= getattr(config, "LOCK_COUNT", 8):
                            bee.locked_digit = top_num
                        
                        bee.current_num = res_str
                        bee.current_conf = res_conf

                # 3. UI Drawing
                color = (0, 215, 255) if bee.locked_digit else (0, 255, 0)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                display_num = bee.locked_digit or bee.current_num or ""
                label = f"ID:{track_id} | #{display_num}"
                cv2.putText(annotated, label, (x1, max(y1-10, 20)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return annotated