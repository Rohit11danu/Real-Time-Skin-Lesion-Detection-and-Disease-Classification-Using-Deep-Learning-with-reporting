import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from ultralytics import YOLO
from PIL import Image
from collections import deque, Counter
import json
import time

# =====================================================
# CONFIG
# =====================================================
YOLO_WEIGHTS = "Object_detection.v1-lesion_v1.yolov8/runs/detect/train/weights/best.pt"
CLS_WEIGHTS  = "face_skin_disease_resnet18.pth"

CONF_DET = 0.5     # YOLO confidence threshold
CONF_CLS = 0.4     # Classifier confidence threshold
DEVICE = "cpu"

# =====================================================
# MODULE 3 SETTINGS (TRACKING + SMOOTHING)
# =====================================================
IOU_MATCH_THR = 0.30     # overlap to keep same lesion ID
MAX_MISSED = 15          # frames before a lesion ID is removed
HISTORY_LEN = 10         # smoothing window length
MIN_AREA = 900           # ignore tiny detections (noise)

next_id = 1
tracks = {}  # lesion_id -> track info

# =====================================================
# LOAD YOLO DETECTOR
# =====================================================
print("🔹 Loading YOLO detector...")
detector = YOLO(YOLO_WEIGHTS)

# =====================================================
# LOAD CLASSIFIER
# =====================================================
print("🔹 Loading classifier...")
checkpoint = torch.load(CLS_WEIGHTS, map_location=DEVICE)

CLASSES = checkpoint.get(
    "classes",
    ["Acne", "Actinic Keratosis", "Basal Cell Carcinoma", "Eczema", "Rosacea"]
)

clf = models.resnet18(weights=None)
clf.fc = nn.Linear(clf.fc.in_features, len(CLASSES))
clf.load_state_dict(checkpoint["model_state"])
clf.to(DEVICE).eval()

print("✅ Classes:", CLASSES)

# =====================================================
# IMAGE TRANSFORM (must match training)
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
    rgb = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    x = cls_tf(pil).unsqueeze(0).to(DEVICE)
    probs = torch.softmax(clf(x), dim=1)[0]
    prob, idx = torch.max(probs, dim=0)
    return CLASSES[int(idx)], float(prob)

# =====================================================
# HELPER FUNCTIONS
# =====================================================
def iou(a, b):
    xA = max(a[0], b[0])
    yA = max(a[1], b[1])
    xB = min(a[2], b[2])
    yB = min(a[3], b[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = (a[2] - a[0]) * (a[3] - a[1])
    areaB = (b[2] - b[0]) * (b[3] - b[1])
    union = areaA + areaB - inter
    return inter / union if union > 0 else 0

def stable_label(hist):
    return Counter(hist).most_common(1)[0][0] if hist else None

def avg_conf(hist):
    return sum(hist) / len(hist) if hist else 0.0

# =====================================================
# LIVE CAMERA + LOGGING
# =====================================================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    raise RuntimeError("❌ Could not open webcam")

print("✅ Module 3 Live Started")
print("👉 Press Q to quit")

log_path = "lesion_log.jsonl"
log_f = open(log_path, "a")
start_time = time.time()
print(f"📝 Logging to {log_path}")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    results = detector.predict(frame, imgsz=640, conf=CONF_DET, verbose=False)[0]

    detections = []

    if results.boxes is not None:
        for b in results.boxes:
            x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w-1, x2), min(h-1, y2)

            area = (x2 - x1) * (y2 - y1)
            if area < MIN_AREA:
                continue

            detections.append((x1, y1, x2, y2))

    used = set()
    assigned = []

    # MATCH detections to existing tracks
    for box in detections:
        best_id, best_iou = None, 0
        for tid, t in tracks.items():
            if tid in used:
                continue
            score = iou(box, t["bbox"])
            if score > best_iou:
                best_iou, best_id = score, tid

        if best_id and best_iou >= IOU_MATCH_THR:
            tracks[best_id]["bbox"] = box
            tracks[best_id]["missed"] = 0
            used.add(best_id)
            assigned.append((best_id, box))
        else:
            tid = next_id
            next_id += 1
            tracks[tid] = {
                "bbox": box,
                "missed": 0,
                "label_hist": deque(maxlen=HISTORY_LEN),
                "conf_hist": deque(maxlen=HISTORY_LEN)
            }
            used.add(tid)
            assigned.append((tid, box))

    # Remove lost tracks
    for tid in list(tracks.keys()):
        if tid not in used:
            tracks[tid]["missed"] += 1
        if tracks[tid]["missed"] > MAX_MISSED:
            del tracks[tid]

    # CLASSIFY + DRAW + LOG
    for tid, (x1, y1, x2, y2) in assigned:
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        label, conf = classify_crop(crop)
        tracks[tid]["label_hist"].append(label)
        tracks[tid]["conf_hist"].append(conf)

        stable = stable_label(tracks[tid]["label_hist"])
        avg_p = avg_conf(tracks[tid]["conf_hist"])

        if avg_p < CONF_CLS:
            text = f"ID {tid}: Lesion"
        else:
            text = f"ID {tid}: {stable} ({avg_p:.2f})"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, text, (x1, max(20, y1-10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        # -------- LOG ENTRY --------
        entry = {
            "t": round(time.time() - start_time, 3),
            "lesion_id": int(tid),
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
            "area": int((x2 - x1) * (y2 - y1)),
            "label": str(stable),
            "label_conf": float(round(avg_p, 4))
        }
        log_f.write(json.dumps(entry) + "\n")

    cv2.imshow("Module 3: Tracking + Logging", frame)
    if cv2.waitKey(1) & 0xFF in [ord('q'), 27]:
        break

cap.release()
log_f.close()
cv2.destroyAllWindows()
print(f"🛑 Camera closed | Log saved to {log_path}")
