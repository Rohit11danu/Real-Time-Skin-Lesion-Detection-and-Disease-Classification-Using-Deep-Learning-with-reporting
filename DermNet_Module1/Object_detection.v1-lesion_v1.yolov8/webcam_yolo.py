from ultralytics import YOLO
import cv2

# Load trained model
model = YOLO("runs/detect/train/weights/best.pt")

# Open webcam
cap = cv2.VideoCapture(0)

# Reduce lag
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("✅ Webcam started. Press Q to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run detection
    results = model.predict(
        frame,
        imgsz=640,
        conf=0.25,
        verbose=False
    )[0]

    # Draw boxes
    annotated = results.plot()

    cv2.imshow("Skin Lesion Detection", annotated)

    # Quit on Q or ESC
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        break

cap.release()
cv2.destroyAllWindows()

