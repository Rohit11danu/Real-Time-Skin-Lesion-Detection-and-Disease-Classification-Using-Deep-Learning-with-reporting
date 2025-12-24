import cv2
import torch
from torchvision import transforms, models
from torch import nn
from facenet_pytorch import MTCNN

# =========================
# SETTINGS (tune these)
# =========================
MODEL_PATH = "face_skin_disease_resnet18.pth"
IMG_SIZE = 224

# Speed controls
CAP_W, CAP_H = 640, 480              # webcam resolution (lower = faster)
DETECT_WIDTH = 320                   # downscale width used for face detection (320/480)
DETECT_EVERY_N_FRAMES = 8            # run face detection once every N frames (5–12)
PREDICT_EVERY_N_FRAMES = 3           # run model prediction once every N frames (2–6)

# Display controls
CONF_THRESHOLD = 0.60
SHOW_TOP3 = True

# =========================
# LOAD MODEL
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

ckpt = torch.load(MODEL_PATH, map_location=device)
classes = ckpt["classes"]
print("Classes:", classes)

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(classes))
model.load_state_dict(ckpt["model_state"])
model = model.to(device).eval()

preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# Face detector (MTCNN)
mtcnn = MTCNN(keep_all=False, device=device)

# =========================
# WEBCAM SETUP
# =========================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_H)

if not cap.isOpened():
    raise RuntimeError("❌ Could not open webcam. Try changing VideoCapture(0) to 1.")

print("✅ Webcam started. Press Q to quit.")

# =========================
# CACHES (to reduce lag)
# =========================
frame_count = 0
last_box = None         # last detected face box (x1,y1,x2,y2)
last_label = "Detecting..."
last_top3 = []

# =========================
# MAIN LOOP
# =========================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    h, w = frame.shape[:2]

    # -------------------------
    # FACE DETECTION (sometimes)
    # -------------------------
    run_detect = (frame_count % DETECT_EVERY_N_FRAMES == 0) or (last_box is None)

    if run_detect:
        # Downscale for faster detection
        scale = DETECT_WIDTH / w
        small = cv2.resize(frame, (DETECT_WIDTH, int(h * scale)))
        rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        boxes, _ = mtcnn.detect(rgb_small)

        if boxes is not None:
            # Scale box back up to original frame size
            b = boxes[0] / scale
            x1, y1, x2, y2 = b.astype(int)

            # Clip to bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            last_box = (x1, y1, x2, y2)
        # If detect fails, keep last_box (don’t reset) to avoid flicker

    # -------------------------
    # If we have a face box, crop
    # -------------------------
    if last_box is not None:
        x1, y1, x2, y2 = last_box
        face = frame[y1:y2, x1:x2]

        # -------------------------
        # PREDICT (sometimes)
        # -------------------------
        run_predict = (frame_count % PREDICT_EVERY_N_FRAMES == 0)

        if run_predict and face.size > 0:
            x = preprocess(face).unsqueeze(0).to(device)

            with torch.no_grad():
                logits = model(x)
                probs = torch.softmax(logits, dim=1)[0]

            # Best class
            idx = int(torch.argmax(probs).item())
            conf = float(probs[idx].item())

            if conf < CONF_THRESHOLD:
                last_label = f"Uncertain ({conf*100:.1f}%)"
            else:
                last_label = f"{classes[idx]} ({conf*100:.1f}%)"

            # Top-3 cache
            if SHOW_TOP3:
                vals, inds = torch.topk(probs, 3)
                last_top3 = [
                    f"{classes[int(i)]}: {float(v)*100:.1f}%"
                    for v, i in zip(vals, inds)
                ]

        # -------------------------
        # DRAW (every frame)
        # -------------------------
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, last_label, (x1, max(25, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if SHOW_TOP3 and last_top3:
            for j, t in enumerate(last_top3):
                cv2.putText(frame, t, (x1, y2 + 25 + j * 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

    cv2.imshow("Face Skin Disease (Demo) - Optimized", frame)

    key = cv2.waitKey(1) & 0xFF
    if key in [ord('q'), ord('Q')]:
        break

cap.release()
cv2.destroyAllWindows()
print("✅ Closed.")
