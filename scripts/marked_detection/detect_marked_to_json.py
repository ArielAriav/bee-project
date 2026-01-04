from ultralytics import YOLO
import cv2
import json

# Loading the model
model = YOLO("models/marked_detection/best.pt")

# Opening a video
video_path = "data/raw/entrance/beeVideo2.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error - The video cannot be opened.")
    exit()

# JSON structure
detections_json = {}
frame_index = 0

print("Starts detection - press Q to exit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Running the model
    results = model(frame, conf=0.4)

    frame_detections = []
    bee_count = 0

    # Skip all detections in the frame
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        class_name = model.names[cls_id]
        confidence = float(box.conf[0])

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Bee counting
        if class_name.lower() in ["bee", "queen"]:
            bee_count += 1

        frame_detections.append({
            "class": class_name,
            "confidence": round(confidence, 3),
            "bbox": [x1, y1, x2, y2]
        })

    # Saving to JSON 
    detections_json[f"frame_{frame_index}"] = {
        "bee_count": bee_count,
        "detections": frame_detections
    }

    # Drawing on the frame
    annotated_frame = results[0].plot()
    cv2.putText(
        annotated_frame,
        f"Bees detected: {bee_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("Bee Detection + Count", annotated_frame)

    frame_index += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Saving JSON to a file
with open("detections.json", "w") as f:
    json.dump(detections_json, f, indent=4)

print("JSON saved successfully: detections.json")

