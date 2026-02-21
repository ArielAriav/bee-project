# backend/main.py
import os
import time
import shutil
import cv2
import uuid
import glob
from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from paddleocr import PaddleOCR
from processor import BeeProcessor
import config
import torch

# --- INITIALIZATION ---
print("--- Loading AI Models ---")
GLOBAL_YOLO = YOLO(config.MODEL_PATH)
if torch.backends.mps.is_available():
    print("✅ Mac GPU (MPS) is available! Moving YOLO to GPU.")
    GLOBAL_YOLO.to('mps')
else:
    print("⚠️ MPS not available. YOLO will run on CPU.")

# Initialize PaddleOCR with settings optimized for bee tag recognition
GLOBAL_PADDLE = PaddleOCR(
    use_gpu=True,
    lang="en",
    text_detection_model_name=None, 
    text_recognition_model_name="en_PP-OCRv5_mobile_rec",
    use_doc_orientation_classify=False,
    use_doc_unwarping=config.USE_DOC_UNWARPING,
    use_textline_orientation=config.USE_TEXTLINE_ORIENTATION,
    show_log=False
)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Adjust this to your frontend URL in production
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared state to track the active processing session
active_session = {
    "id": None,
    "processor": None,
    "is_finished": False,
    "current_file": None
}

def cleanup_temporary_files():
    """
    Finds and deletes any leftover temporary video files 
    from previous sessions to save disk space.
    """
    for temp_file in glob.glob("temp_*"):
        try:
            # We avoid deleting the file that is currently being processed
            if active_session["current_file"] and temp_file in active_session["current_file"]:
                continue
            os.remove(temp_file)
            print(f"Cleanup: Deleted legacy file {temp_file}")
        except Exception as e:
            print(f"Cleanup Error: Could not remove {temp_file} -> {e}")

@app.post("/upload-video")
async def upload(file: UploadFile = File(...)):
    global active_session
    
    # Clean up old files before starting a new upload/detection session
    cleanup_temporary_files()
    
    new_session_id = str(uuid.uuid4())
    filename = f"temp_{int(time.time())}_{file.filename}"
    temp_path = os.path.join(os.getcwd(), filename)

    # Save the uploaded file locally
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Initialize a fresh processor for this specific video
    active_session["processor"] = BeeProcessor(GLOBAL_YOLO, GLOBAL_PADDLE)
    active_session["id"] = new_session_id
    active_session["is_finished"] = False
    active_session["current_file"] = temp_path

    return {"filename": filename, "session_id": new_session_id}

def frame_generator(path, session_id):
    cap = cv2.VideoCapture(path)
    try:
        while cap.isOpened():
            # Stop processing if a new session (new upload) has started
            if session_id != active_session["id"]: 
                break
                
            ret, frame = cap.read()
            if not ret:
                active_session["is_finished"] = True
                break
            
            proc = active_session["processor"]
            annotated_frame, _ = proc.process_and_annotate(frame)
            
            # Encode frame as JPEG for streaming
            _, buffer = cv2.imencode('.jpg', annotated_frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    finally:
        # Release the video file handle
        cap.release()
        
        # Automatically delete the video file once processing is done or interrupted
        if os.path.exists(path):
            try:
                os.remove(path)
                print(f"Auto-Cleanup: Successfully removed {path}")
            except Exception as e:
                print(f"Auto-Cleanup Error: {e}")

@app.get("/video-feed")
async def video_feed(filename: str = Query(...), session_id: str = Query(...)):
    path = os.path.join(os.getcwd(), filename)
    return StreamingResponse(
        frame_generator(path, session_id), 
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.get("/get-result")
async def get_result():
    try:
        proc = active_session["processor"]
        if not proc:
            return {
                "bees": [],
                "video_ended": active_session["is_finished"],
                "status": "Idle"
            }

        active_bees = []
        for bee_id, bee_state in list(proc.bees.items()):
            active_bees.append({
                "track_id": int(bee_state.original_id),
                "number": bee_state.locked_digit if bee_state.locked_digit else bee_state.current_num,
                "is_locked": bee_state.locked_digit is not None,
                "confidence": float(bee_state.current_conf or 0.0)
            })

        # Sort results: locked items first, then by confidence
        active_bees.sort(key=lambda x: (not x["is_locked"], -x["confidence"]))

        return {
            "bees": active_bees,
            "video_ended": active_session["is_finished"],
            "status": "Processing" if not active_session["is_finished"] else "Finished"
        }

    except Exception as e:
        print(f"❌ API Error in /get-result: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "bees": [],
                "video_ended": active_session.get("is_finished", False),
                "status": "Error",
                "message": str(e)
            }
        )

if __name__ == "__main__":
    import uvicorn
    # Start the server
    uvicorn.run(app, host="0.0.0.0", port=8000)