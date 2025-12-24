import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from ultralytics import YOLO
from PIL import Image

# =====================================================
# CONFIG
# =====================================================
YOLO_WEIGHTS = "Object_detection.v1-lesion_v1.yolov8/runs/detect/train/weights/best.pt"
CLS_WEIGHTS  = "face_skin_disease_resnet18.pth"

CONF_DET = 0.50   # YOLO confidence threshold
CONF_CLS = 0.40   # Classifier confidence threshold

DEVICE = "cpu"   # Apple Mac CPU

# =====================================================
# LOAD YOLO DETECTOR (MODULE 1)
# =====================================================
print("🔹 Loading YOLO detector...")
detector = YOLO(YOLO_WEIGHTS)

# =====================================================
# LOAD CLASSIFIER (MODULE 2)
# =====================================================
print("🔹 Loading disease classifier...")

checkpoint = torch.load(CLS_WEIGHTS, map_location=DEVICE)

# Load class names
CLASSES = checkpoint.get(
    "classes",
    ["Acne", "Actinic Keratosis", "Basal Cell Carcinoma", "Eczema", "Rosacea"]
)

# Build model architecture
clf = models.resnet18(weights=None)
clf.fc = nn.Linear(clf.fc.in_features, len(CLASSES))

# Load trained weights
clf.load_state_dict(checkpoint["model_state"])
clf.to(DEVICE)
clf.eval()

print("✅ Classifier loaded with classes:", CLASSES)

# =====================================================
# IMAGE TRANSFORMS (MUST MATCH TRAINING)
# =====================================================
cls_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

@torch.no_grad()
def classify_crop(bgr_crop):
    """Classify a cropped lesion image."""
    rgb = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    x = cls_tf(pil).unsqueeze(0).to(DEVICE)

    logits = clf(x)
    probs = torch.softmax(logits, dim=1)[0]
    prob, idx = torch.max(probs, dim=0)

    return CLASSES[int(idx)], float(prob)

# =====================================================
# LIVE CAMERA
# =====================================================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    raise RuntimeError("❌ Could not open webcam")

print("✅ Module 2 Live started")
print("👉 Press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # -----------------------------------------
    # YOLO DETECTION
    # -----------------------------------------
    results = detector.predict(
        frame,
        imgsz=640,
        conf=CONF_DET,
        verbose=False
    )[0]

    if results.boxes is not None:
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            det_conf = float(box.conf[0])

            # Clamp box to image bounds
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)
            if x2 <= x1 or y2 <= y1:
                continue

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            # -----------------------------------------
            # CLASSIFY LESION
            # -----------------------------------------
            label, cls_conf = classify_crop(crop)

            if cls_conf < CONF_CLS:
                text = f"Lesion ({det_conf:.2f})"
            else:
                text = f"{label} ({cls_conf:.2f})"

            # -----------------------------------------
            # DRAW
            # -----------------------------------------
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                text,
                (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    cv2.imshow("Module 2: Lesion Detection + Classification", frame)

    if cv2.waitKey(1) & 0xFF in [ord('q'), 27]:
        break

cap.release()
cv2.destroyAllWindows()
print("🛑 Camera closed")
