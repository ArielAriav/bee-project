# backend/processor.py
import cv2
import numpy as np
from collections import deque, Counter
import config
import os
import time

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
        self.current_num_peak_conf = 0.0
        self.stable_count = 0
        self.current_direction = "Up"
        self.velocity = (0, 0)  # NEW: estimated velocity based on last two positions
        self.prev_center = initial_pos  # NEW: previous center for velocity calculation

    def update_pos(self, pos, yolo_id):
        # NEW: calculate velocity from last known position before updating
        self.velocity = (
            pos[0] - self.last_center[0],
            pos[1] - self.last_center[1]
        )
        self.prev_center = self.last_center
        self.last_center = pos
        self.yolo_id = yolo_id
        self.frames_lost = 0
        self.is_active = True

    @property
    def predicted_next_pos(self):
        """
        Predicts where this bee will be in the next frame based on current velocity.
        """
        return (
            self.last_center[0] + self.velocity[0],
            self.last_center[1] + self.velocity[1]
        )

class BeeProcessor:
    def __init__(self, yolo_instance, digit_model_instance, angle_model_instance):
        """
        Initializes models. model = tag detection, digit_model = digit recognition, angle_model = direction classification.
        """
        self.model = yolo_instance
        self.digit_model = digit_model_instance
        self.angle_model = angle_model_instance
        self.frame_idx = 0
        self.bees = {}

        os.makedirs(config.SAVE_CROPS_DIR, exist_ok=True)

    def find_nearby_bee(self, cx, cy, new_id, max_dist=120):
        """
        Looks for an existing bee near the given position.
        Checks both current position and predicted next position to handle fast-moving bees.
        """
        for existing_id, bee in self.bees.items():
            if existing_id == new_id:
                continue
            if not bee.is_active:
                continue

            ex, ey = bee.last_center
            px, py = bee.predicted_next_pos

            # distance to current position
            dist_current = ((cx - ex) ** 2 + (cy - ey) ** 2) ** 0.5
            # distance to predicted next position
            dist_predicted = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5

            # match if either distance is within threshold
            if min(dist_current, dist_predicted) < max_dist:
                return existing_id, bee
        return None, None

    def read_digits(self, crop, direction):
        """
        Reads digits by mathematically rotating the crop so the digits are upright,
        then feeds the upright image to the OCR model.
        """
        if crop is None or crop.size == 0:
            return None, 0.0

        # rotate the crop according to the detected direction
        rotated_crop = crop

        if direction == "Down":
            # rotate 180 degrees
            rotated_crop = cv2.rotate(crop, cv2.ROTATE_180)
        elif direction == "Left":
            # rotate 90 degrees clockwise
            rotated_crop = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
        elif direction == "Right":
            # rotate 90 degrees counter-clockwise
            rotated_crop = cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # send the rotated (upright) crop to the digit model
        results = self.digit_model.predict(rotated_crop, conf=0.25, verbose=False)
        detected = []

        if results and len(results[0].boxes) > 0:
            for box in results[0].boxes:
                detected.append((box.xyxy[0][0].item(), int(box.cls[0]), box.conf[0].item()))

            x_coords = [d[0] for d in detected]
            y_coords = [box.xyxy[0][1].item() for box in results[0].boxes]

            x_spread = max(x_coords) - min(x_coords)
            y_spread = max(y_coords) - min(y_coords)

            # since the image is now upright, digits are always arranged horizontally
            if x_spread >= y_spread:
                detected.sort(key=lambda x: x[0])
            else:
                detected.sort(key=lambda x: x[2])

            digit_str = "".join([str(d[1]) for d in detected])
            avg_conf = sum([d[2] for d in detected]) / len(detected)

            # string reversal removed — image is upright so model always reads left to right

            return digit_str, avg_conf

        return None, 0.0

    def process_and_annotate(self, frame):
        self.frame_idx += 1
        annotated = frame.copy()

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
                    existing_id, existing_bee = self.find_nearby_bee(cx, cy, track_id)
                    if existing_bee is not None:
                        # reuse the existing state under the new ID
                        self.bees[track_id] = existing_bee
                        self.bees[track_id].yolo_id = track_id
                        del self.bees[existing_id]
                    else:
                        self.bees[track_id] = BeeState(track_id, (cx, cy))

                bee = self.bees[track_id]
                bee.update_pos((cx, cy), track_id)

                # run every OCR_EVERY frames to reduce CPU load, regardless of lock state
                if self.frame_idx % config.OCR_EVERY == 0:

                    # 1. create expanded crop for direction detection
                    ey1 = max(0, y1 - config.CROP_EXPAND)
                    ey2 = min(frame.shape[0], y2 + config.CROP_EXPAND)
                    ex1 = max(0, x1 - config.CROP_EXPAND)
                    ex2 = min(frame.shape[1], x2 + config.CROP_EXPAND)
                    expanded_crop = frame[ey1:ey2, ex1:ex2]

                    # 2. direction detection — always runs and updates the bee state
                    if expanded_crop is not None and expanded_crop.size > 0:
                        angle_results = self.angle_model.predict(expanded_crop, verbose=False)
                        if angle_results:
                            top_class_index = angle_results[0].probs.top1
                            bee.current_direction = angle_results[0].names[top_class_index]

                    # 3. standard crop and digit OCR — runs only if number is not yet locked
                    if bee.locked_digit is None:
                        crop = frame[max(0,y1):min(frame.shape[0],y2), max(0,x1):min(frame.shape[1],x2)]
                        res_str, res_conf = self.read_digits(crop, bee.current_direction)

                        if res_str and 1 <= len(res_str) <= config.MAX_DIGITS:
                            bee.recent_digits.append(res_str)
                            counts = Counter(bee.recent_digits)
                            best_num, best_freq = counts.most_common(1)[0]

                            if bee.current_num is None:
                                bee.current_num = best_num
                                bee.current_num_peak_conf = res_conf
                                bee.stable_count = 1
                            elif res_str == bee.current_num:
                                bee.current_num_peak_conf = max(bee.current_num_peak_conf, res_conf)
                                bee.stable_count += 1
                            elif (best_num != bee.current_num and best_freq >= 5 and res_conf > bee.current_num_peak_conf):
                                bee.current_num = best_num
                                bee.current_num_peak_conf = res_conf
                                bee.stable_count = 1

                            bee.current_conf = res_conf

                            if bee.stable_count >= config.LOCK_COUNT:
                                bee.locked_digit = bee.current_num

                # UI Drawing
                color = (0, 215, 255) if bee.locked_digit else (0, 255, 0)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                display_num = bee.locked_digit or bee.current_num or ""
                label = f"ID:{bee.original_id} | #{display_num} | {bee.current_direction}"
                cv2.putText(annotated, label, (x1, max(y1-10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return annotated