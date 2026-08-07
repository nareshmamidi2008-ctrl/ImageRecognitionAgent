from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image
import shutil
import os
import time

app = FastAPI()

# Allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load YOLO model
model = YOLO("yolov8n.pt", task="detect")

# Warm up YOLO model
try:
    import numpy as np

    dummy_image = np.zeros((320, 320, 3), dtype=np.uint8)

    model(
        dummy_image,
        imgsz=320,
        conf=0.25,
        verbose=False
    )

    print("✅ YOLO model warmed up successfully!")

except Exception as e:
    print("⚠️ YOLO warm-up failed:", e)


@app.get("/")
def home():
    return {
        "message": "Image Recognition API is running!"
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # Save uploaded image
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Resize large images
    image = Image.open(file_path)

    MAX_SIZE = (640, 640)

    image.thumbnail(MAX_SIZE)

    image.save(file_path, optimize=True, quality=85)

    start = time.time()

    # Run YOLO
    results = model(
        file_path,
        imgsz=320,
        conf=0.25,
        verbose=False
    )

    end = time.time()

    detections = []

    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            detections.append({
                "object": model.names[cls],
                "confidence": round(conf * 100, 2)
            })

    return {
        "detections": detections,
        "processing_time": round(end - start, 2)
    }