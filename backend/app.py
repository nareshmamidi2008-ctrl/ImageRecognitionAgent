from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import shutil
import os
import time

app = FastAPI()

# Allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    
)

UPLOAD_FOLDER = "uploads"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model = YOLO("yolov8n.pt")

@app.get("/")
def home():
    return {
        "message": "Image Recognition API is running!"
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    start = time.time()

    results = model(file_path)

    end = time.time()

    detections = []

    for result in results:

        for box in result.boxes:

            cls = int(box.cls[0])

            conf = float(box.conf[0])

            detections.append({
                "object": model.names[cls],
                "confidence": round(conf*100,2)
            })

    return {
        "detections": detections,
        "processing_time": round(end-start,2)
    }