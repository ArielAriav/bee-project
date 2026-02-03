# backend/main.py
import os
import time
import shutil
import cv2
import uuid
from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from paddleocr import PaddleOCR
from processor import BeeProcessor
import config

# --- INITIALIZATION ---
print("--- Loading AI Models (Shared Global Instance) ---")
GLOBAL_YOLO = YOLO(config.MODEL_PATH)
GLOBAL_PADDLE = PaddleOCR(
    lang="en",
    text_detection_model_name=None, 
    text_recognition_model_name="en_PP-OCRv5_mobile_rec",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    show_log=False
)
print("✅ Models loaded successfully!")

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# --- SESSION MANAGEMENT ---
# Added 'is_finished' flag to track video completion
active_session = {
    "id": None,
    "processor": None,
    "is_finished": False
}

@app.post("/upload-video")
async def upload(file: UploadFile = File(...)):
    global active_session

    new_session_id = str(uuid.uuid4())
    print(f"🔄 Starting New Scan Session: {new_session_id}")

    new_processor = BeeProcessor(GLOBAL_YOLO, GLOBAL_PADDLE)
    
    # Reset session state including the finished flag
    active_session["id"] = new_session_id
    active_session["processor"] = new_processor
    active_session["is_finished"] = False 

    filename = f"temp_{int(time.time())}_{file.filename}"
    temp_path = os.path.join(os.getcwd(), filename)
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    return {
        "status": "ready", 
        "filename": filename, 
        "session_id": new_session_id
    }

def frame_generator(path, session_id):
    cap = cv2.VideoCapture(path)
    try:
        while cap.isOpened():
            if session_id != active_session["id"]:
                print(f"🛑 Killing old stream session: {session_id}")
                break 

            ret, frame = cap.read()
            if not ret: 
                # Video ended normally -> Mark as finished
                if session_id == active_session["id"]:
                    active_session["is_finished"] = True
                    print("✅ Video processing completed.")
                break
            
            processor = active_session["processor"]
            annotated_frame, _ = processor.process_and_annotate(frame)
            
            _, buffer = cv2.imencode('.jpg', annotated_frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    finally:
        cap.release()
        time.sleep(0.5)
        if os.path.exists(path) and session_id != active_session["id"]:
            try: os.remove(path)
            except: pass

@app.get("/video-feed")
async def video_feed(filename: str = Query(...), session_id: str = Query(...)):
    path = os.path.join(os.getcwd(), filename)
    return StreamingResponse(
        frame_generator(path, session_id), 
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.get("/get-result")
async def get_result():
    proc = active_session["processor"]
    if not proc:
        return {"status": "Idle"}
        
    # Return the 'is_finished' flag to the frontend
    return {
        "id": proc.locked_digit if proc.locked_digit else proc.current_num,
        "is_locked": proc.locked_digit is not None,
        "confidence": float(proc.current_conf),
        "status": "Stable" if proc.locked_digit else "Analyzing...",
        "video_ended": active_session["is_finished"]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)