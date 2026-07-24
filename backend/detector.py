from ultralytics import YOLO
import cv2

# Load YOLO model (downloads automatically the first time)
model = YOLO("yolov8n.pt")

# Image path
image_path = input("Enter image path: ")

# Run detection
results = model(image_path)

# Display results
for result in results:
    result.show()      # Show image with bounding boxes
    result.save()      # Save detected image

    print("\nDetected Objects:\n")

    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])

        print(f"Object : {model.names[class_id]}")
        print(f"Confidence : {confidence*100:.2f}%")
        print("------------------------")