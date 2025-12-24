from ultralytics import YOLO

# Load YOLOv8 nano model
model = YOLO("yolov8n.pt")

# Train the model
model.train(
    data="data.yaml",
    imgsz=640,
    epochs=50,
    batch=8
)

