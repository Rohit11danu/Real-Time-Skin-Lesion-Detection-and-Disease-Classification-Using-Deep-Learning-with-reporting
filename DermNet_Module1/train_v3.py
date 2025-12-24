from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="data/Object_detection.v3-lesion_v3.yolov8/data.yaml",
    imgsz=640,
    epochs=120,
    batch=8
)

