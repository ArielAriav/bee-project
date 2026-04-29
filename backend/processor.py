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
        self.motion_history = deque(maxlen=10)
        self.direction_votes = Counter()
        self.locked_direction = None

    def update_pos(self, pos, yolo_id):
        self.last_center = pos
        self.yolo_id = yolo_id
        self.frames_lost = 0
        self.is_active = True
        self.motion_history.append(pos[0])

        # vote on direction based on horizontal movement
        if len(self.motion_history) >= 2:
            delta = self.motion_history[-1] - self.motion_history[-2]
            if abs(delta) > 3:  # ignore tiny movements
                vote = "ltr" if delta > 0 else "rtl"
                self.direction_votes[vote] += 1

            # lock direction once one side has enough votes
            total = sum(self.direction_votes.values())
            if total >= 10:
                top = self.direction_votes.most_common(1)[0]
                if top[1] / total > 0.7:  # 70% agreement
                    self.locked_direction = top[0]

    @property
    def moving_direction(self):
        # if we have a locked direction from history, use it
        if self.locked_direction:
            return self.locked_direction

        if len(self.motion_history) < 3:
            return None

        delta = self.motion_history[-1] - self.motion_history[0]
        if abs(delta) < 5:
            return None  # truly stationary, no signal

        return "ltr" if delta > 0 else "rtl"
    
    @property
    def dominant_axis(self):
        """
        Returns whether the bee moves more horizontally or vertically.
        Diagonal movement biases digit reading — we only care about
        the horizontal component for direction.
        """
        if len(self.motion_history) < 3:
            return None
    
        # compare total horizontal vs vertical displacement
        # (requires storing Y as well — see note below)
        h_delta = abs(self.motion_history_x[-1] - self.motion_history_x[0])
        v_delta = abs(self.motion_history_y[-1] - self.motion_history_y[0])
    
        if h_delta < 5 and v_delta < 5:
            return None  # stationary
    
        return "horizontal" if h_delta >= v_delta else "vertical"

class BeeProcessor:
    def __init__(self, yolo_instance, digit_model_instance):
        """
        Initializes models. model = tag detection, digit_model = digit recognition.
        """
        self.model = yolo_instance
        self.digit_model = digit_model_instance
        self.frame_idx = 0
        self.bees = {} 

    def read_digits(self, crop, bee: BeeState):
        """
        Reads digits from the tag crop using the digit YOLO model.
        Digit order is corrected based on the bee's movement direction:
        if the bee moves right-to-left, the digit string is reversed.
        """
        if crop is None or crop.size == 0:
            return None, 0.0

        results = self.digit_model.predict(crop, conf=0.25, verbose=False)
        detected = []

        if results and len(results[0].boxes) > 0:
            for box in results[0].boxes:
                # Save (X-coordinate, digit class, confidence)
                detected.append((box.xyxy[0][0].item(), int(box.cls[0]), box.conf[0].item()))

            x_coords = [d[0] for d in detected]
            y_coords = [box.xyxy[0][1].item() for box in results[0].boxes]

            x_spread = max(x_coords) - min(x_coords)
            y_spread = max(y_coords) - min(y_coords)

            if x_spread >= y_spread:
                # digits arranged horizontally — sort by X
                detected.sort(key=lambda x: x[0])
            else:
                # digits arranged vertically — sort by Y
                detected.sort(key=lambda x: x[2])  # x[2] יהיה ה-Y coordinate

            digit_str = "".join([str(d[1]) for d in detected])
            avg_conf = sum([d[2] for d in detected]) / len(detected)

            # reverse the digit string to get the correct reading order
            if bee.moving_direction == "rtl":
                digit_str = digit_str[::-1]

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
                    # pass bee so read_digits can apply direction-aware digit ordering
                    res_str, res_conf = self.read_digits(crop, bee)
                    
                    if res_str and 1 <= len(res_str) <= config.MAX_DIGITS:
                        # normalize to canonical form before voting —
                        canonical = res_str
                        if bee.moving_direction == "rtl":
                            canonical = res_str[::-1]
                        bee.recent_digits.append(canonical)
                        
                        # Number Confirmation (Locking)
                        counts = Counter(bee.recent_digits)
                        top_num, freq = counts.most_common(1)[0]
                        if freq >= getattr(config, "LOCK_COUNT", 8):
                            bee.locked_digit = top_num
                        
                        # always store the raw result for confidence tracking
                        bee.current_conf = res_conf

                        # display the most voted number so far, not the latest guess
                        counts = Counter(bee.recent_digits)
                        if counts:
                            bee.current_num = counts.most_common(1)[0][0]
                        else:
                            bee.current_num = res_str

                # 3. UI Drawing
                color = (0, 215, 255) if bee.locked_digit else (0, 255, 0)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                display_num = bee.locked_digit or bee.current_num or ""
                label = f"ID:{track_id} | #{display_num}"
                cv2.putText(annotated, label, (x1, max(y1-10, 20)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return annotated