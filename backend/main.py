# backend/main.py
import os
import time
import shutil
import cv2
import uuid
import glob
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from processor import BeeProcessor
import config

# --- INITIALIZATION ---
print("--- Loading AI Models ---")
# Load both YOLO models globally
GLOBAL_YOLO_TAG = YOLO(config.MODEL_PATH)
GLOBAL_YOLO_DIGITS = YOLO(config.DIGIT_MODEL_PATH)
GLOBAL_YOLO_ANGLE = YOLO(config.ANGLE_MODEL_PATH)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared state for the active session
active_session = {
    "id": None,
    "processor": None,
    "is_finished": False,
    "current_file": None,
    "input_source": None,  # "file" | "camera"
}

def cleanup_temporary_files():
    """
    Cleans up old video files from previous runs to save space.
    """
    for temp_file in glob.glob("temp_*"):
        try:
            if active_session["current_file"] and temp_file in active_session["current_file"]:
                continue
            os.remove(temp_file)
        except Exception as e:
            print(f"Cleanup Error: {e}")

@app.post("/upload-video")
async def upload(file: UploadFile = File(...)):
    global active_session
    
    cleanup_temporary_files()
    
    new_session_id = str(uuid.uuid4())
    filename = f"temp_{int(time.time())}_{file.filename}"
    temp_path = os.path.join(os.getcwd(), filename)

    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Initialize processor with ALL THREE models
    active_session["processor"] = BeeProcessor(GLOBAL_YOLO_TAG, GLOBAL_YOLO_DIGITS, GLOBAL_YOLO_ANGLE)
    active_session["id"] = new_session_id
    active_session["is_finished"] = False
    active_session["current_file"] = temp_path
    active_session["input_source"] = "file"

    return {"filename": filename, "session_id": new_session_id}


@app.post("/start-camera")
async def start_camera(camera_index: Optional[int] = Query(default=None)):
    """Start a live USB camera session using the same inference pipeline as uploaded video."""
    global active_session

    cleanup_temporary_files()

    device_index = camera_index if camera_index is not None else config.CAMERA_INDEX
    new_session_id = str(uuid.uuid4())

    active_session["processor"] = BeeProcessor(GLOBAL_YOLO_TAG, GLOBAL_YOLO_DIGITS, GLOBAL_YOLO_ANGLE)
    active_session["id"] = new_session_id
    active_session["is_finished"] = False
    active_session["current_file"] = None
    active_session["input_source"] = "camera"

    return {"session_id": new_session_id, "camera_index": device_index}


@app.post("/stop-session")
async def stop_session():
    """Stop the active stream (camera or video) and mark the session finished."""
    global active_session
    active_session["is_finished"] = True
    active_session["id"] = None
    return {"status": "stopped"}

def frame_generator(path, session_id):
    """
    Handles frame-by-frame processing and maintains the video stream.
    """
    cap = cv2.VideoCapture(path)
    try:
        while cap.isOpened():
            if session_id != active_session["id"]:
                break
                
            ret, frame = cap.read()
            if not ret or frame is None:
                active_session["is_finished"] = True
                break
            
            proc = active_session["processor"]
            try:
                # Core call: Get annotated frame from processor
                annotated_frame = proc.process_and_annotate(frame) if proc else frame
            except Exception as e:
                print(f"Frame Processing Exception: {e}")
                annotated_frame = frame
            
            # Safe encoding to prevent black screen on encoding failures
            if annotated_frame is not None:
                success, buffer = cv2.imencode('.jpg', annotated_frame)
                if success:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    finally:
        cap.release()
        # PRESERVED: Automatic file cleanup after stream ends
        if os.path.exists(path):
            try:
                os.remove(path)
            except:
                pass

@app.get("/video-feed")
async def video_feed(filename: str = Query(...), session_id: str = Query(...)):
    path = os.path.join(os.getcwd(), filename)
    return StreamingResponse(
        frame_generator(path, session_id), 
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


def camera_frame_generator(session_id, camera_index):
    """
    Reads frames from a USB camera and runs the same per-frame processing as file input.
    """
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        active_session["is_finished"] = True
        return

    try:
        while cap.isOpened():
            if session_id != active_session["id"] or active_session["is_finished"]:
                break

            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            proc = active_session["processor"]
            try:
                annotated_frame = proc.process_and_annotate(frame) if proc else frame
            except Exception as e:
                print(f"Camera Frame Processing Exception: {e}")
                annotated_frame = frame

            if annotated_frame is not None:
                success, buffer = cv2.imencode('.jpg', annotated_frame)
                if success:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    finally:
        cap.release()
        active_session["is_finished"] = True


@app.get("/camera-feed")
async def camera_feed(
    session_id: str = Query(...),
    camera_index: Optional[int] = Query(default=None),
):
    device_index = camera_index if camera_index is not None else config.CAMERA_INDEX
    return StreamingResponse(
        camera_frame_generator(session_id, device_index),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )

@app.get("/get-result")
async def get_result():
    try:
        proc = active_session["processor"]
        if not proc:
            return {"bees": [], "video_ended": active_session["is_finished"], "status": "Idle"}

        active_bees = []
        for bee_id, bee_state in list(proc.bees.items()):
            if bee_state.original_id is None:
                continue
            active_bees.append({
                "track_id": int(bee_state.original_id),
                "number": bee_state.locked_digit,
                "is_locked": True,
                "confidence": float(bee_state.current_conf or 0.0)
            })

        active_bees.sort(key=lambda x: (not x["is_locked"], -x["confidence"]))
        return {
            "bees": active_bees,
            "video_ended": active_session["is_finished"],
            "status": "Processing" if not active_session["is_finished"] else "Finished"
        }
    except Exception as e:
        print(f"❌ Error in /get-result: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)