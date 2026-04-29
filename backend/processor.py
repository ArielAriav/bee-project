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
        self.recent_digits = deque(maxlen=60)
        self.locked_digit = None
        self.current_num = None
        self.current_conf = 0.0
        self.motion_history = deque(maxlen=10)
        self.direction_votes = Counter()
        self.locked_direction = None
        self.current_num_peak_conf = 0.0
        self.stable_count = 0  # counts consecutive frames where current_num didn't change

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
    
class BeeProcessor:
    def __init__(self, yolo_instance, digit_model_instance):
        """
        Initializes models. model = tag detection, digit_model = digit recognition.
        """
        self.model = yolo_instance
        self.digit_model = digit_model_instance
        self.frame_idx = 0
        self.bees = {} 

    def find_nearby_bee(self, cx, cy, new_id, max_dist=80):
        """
        Looks for an existing bee near the given position.
        If found, it means YOLO assigned a new ID to an already-known bee.
        Returns the existing BeeState or None.
        """
        for existing_id, bee in self.bees.items():
            if existing_id == new_id:
                continue
            if not bee.is_active:
                continue
            ex, ey = bee.last_center
            dist = ((cx - ex) ** 2 + (cy - ey) ** 2) ** 0.5
            if dist < max_dist:
                return existing_id, bee
        return None, None

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

                if track_id not in self.bees:
                    # check if this is a known bee that got a new YOLO ID
                    existing_id, existing_bee = self.find_nearby_bee(cx, cy, track_id)
                    if existing_bee is not None:
                        # reuse the existing state under the new ID
                        self.bees[track_id] = existing_bee
                        self.bees[track_id].yolo_id = track_id
                        # remove old ID to avoid duplicates
                        del self.bees[existing_id]
                    else:
                        self.bees[track_id] = BeeState(track_id, (cx, cy))

                bee = self.bees[track_id]
                bee.update_pos((cx, cy), track_id)
                
                # 2. Digit Detection (every N frames)
                if self.frame_idx % config.OCR_EVERY == 0 and bee.locked_digit is None:
                    crop = frame[max(0,y1):min(frame.shape[0],y2), max(0,x1):min(frame.shape[1],x2)]
                    # pass bee so read_digits can apply direction-aware digit ordering
                    res_str, res_conf = self.read_digits(crop, bee)
                    
                    if res_str and 1 <= len(res_str) <= config.MAX_DIGITS:
                        # normalize to canonical form before voting —
                        canonical = res_str
                        if bee.moving_direction == "rtl":
                            canonical = res_str[::-1]
                        bee.recent_digits.append(canonical)

                        counts = Counter(bee.recent_digits)
                        best_num, best_freq = counts.most_common(1)[0]

                        # --- Display logic ---
                        if bee.current_num is None:
                            bee.current_num = best_num
                            bee.current_num_peak_conf = res_conf
                            bee.stable_count = 1

                        elif canonical == bee.current_num:
                            # same number — strengthen
                            bee.current_num_peak_conf = max(bee.current_num_peak_conf, res_conf)
                            bee.stable_count += 1

                        elif (best_num != bee.current_num
                                and best_freq >= 5
                                and res_conf > bee.current_num_peak_conf):
                            # switch — reset stability counter
                            bee.current_num = best_num
                            bee.current_num_peak_conf = res_conf
                            bee.stable_count = 1  # reset! must re-earn stability

                        bee.current_conf = res_conf

                        # --- Locking logic ---
                        # lock only after current_num has been stable for LOCK_COUNT consecutive readings
                        if bee.stable_count >= config.LOCK_COUNT:
                            bee.locked_digit = bee.current_num

                # 3. UI Drawing
                color = (0, 215, 255) if bee.locked_digit else (0, 255, 0)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                display_num = bee.locked_digit or bee.current_num or ""
                label = f"ID:{bee.original_id} | #{display_num}"
                cv2.putText(annotated, label, (x1, max(y1-10, 20)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        return annotated