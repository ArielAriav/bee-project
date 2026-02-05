# backend/main.py
import os
import time
import shutil
import cv2
import uuid
from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from paddleocr import PaddleOCR
from processor import BeeProcessor
import config

# --- INITIALIZATION ---
print("--- Loading AI Models ---")
GLOBAL_YOLO = YOLO(config.MODEL_PATH)
GLOBAL_PADDLE = PaddleOCR(
    lang="en",
    text_detection_model_name=None, 
    text_recognition_model_name="en_PP-OCRv5_mobile_rec",
    use_doc_orientation_classify=False,
    show_log=False
)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

active_session = {
    "id": None,
    "processor": None,
    "is_finished": False
}

@app.post("/upload-video")
async def upload(file: UploadFile = File(...)):
    global active_session
    new_session_id = str(uuid.uuid4())
    
    active_session["processor"] = BeeProcessor(GLOBAL_YOLO, GLOBAL_PADDLE)
    active_session["id"] = new_session_id
    active_session["is_finished"] = False

    filename = f"temp_{int(time.time())}_{file.filename}"
    temp_path = os.path.join(os.getcwd(), filename)
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    return {"filename": filename, "session_id": new_session_id}

def frame_generator(path, session_id):
    cap = cv2.VideoCapture(path)
    try:
        while cap.isOpened():
            if session_id != active_session["id"]: break
            ret, frame = cap.read()
            if not ret:
                active_session["is_finished"] = True
                break
            
            proc = active_session["processor"]
            annotated_frame, _ = proc.process_and_annotate(frame)
            _, buffer = cv2.imencode('.jpg', annotated_frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    finally:
        cap.release()

@app.get("/video-feed")
async def video_feed(filename: str = Query(...), session_id: str = Query(...)):
    path = os.path.join(os.getcwd(), filename)
    return StreamingResponse(frame_generator(path, session_id), media_type="multipart/x-mixed-replace; boundary=frame")

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

        active_bees.sort(key=lambda x: (not x["is_locked"], -x["confidence"]))

        return {
            "bees": active_bees,
            "video_ended": active_session["is_finished"],
            "status": "Processing" if not active_session["is_finished"] else "Finished"
        }

    except Exception as e:
        print(f"❌ ERROR in /get-result: {e}")
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
    uvicorn.run(app, host="0.0.0.0", port=8000)