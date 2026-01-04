from ultralytics import YOLO
import cv2

# Loading the trained model
model = YOLO("models/marked_detection/best.pt")

# Opens the video
video_path = "data/raw/entrance/beeVideo2.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error - The video cannot be opened.")
    exit()

print("Starting recognition... Press Q to exit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Running the model on frame
    results = model(frame, conf=0.4)

    # Drawing the results boxes on the frame
    annotated_frame = results[0].plot()

    # Display the video with the identification
    cv2.imshow("Bee Detection - QueenTrack", annotated_frame)

    # Exit by pressing Q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

