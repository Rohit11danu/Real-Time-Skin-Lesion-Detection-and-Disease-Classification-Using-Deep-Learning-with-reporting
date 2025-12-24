from flask import Flask, render_template, request, jsonify
import os
import cv2
import numpy as np
from openai import OpenAI

app = Flask(__name__)

# -----------------------------
# CONFIG
# -----------------------------
YOLO_WEIGHTS = "./weights/lesion_yolov8n_v3.pt"
CLS_CKPT = "./face_skin_disease_resnet18.pth"

CONF_DET = 0.30

MIN_BOX_AREA_FRAC = 0.00010
MAX_BOX_AREA_FRAC = 0.25
MAX_EDGE_TOUCH_FRAC = 0.02
MIN_BOX_SIZE_PX = 8

CONF_CLS = 0.70
MIN_CROP_SIZE_PX = 40
CROP_PAD_RATIO = 0.15

# -----------------------------
# GLOBAL MODELS
# -----------------------------
yolo_model = None
cls_model = None
cls_classes = None
cls_tfm = None
cls_device = None

# OpenAI client
openai_client = OpenAI()

# -----------------------------
# UTILS
# -----------------------------
def clamp_bbox(b, w, h):
    x1, y1, x2, y2 = b
    x1 = int(max(0, min(w - 1, x1)))
    y1 = int(max(0, min(h - 1, y1)))
    x2 = int(max(0, min(w - 1, x2)))
    y2 = int(max(0, min(h - 1, y2)))
    if x2 <= x1:
        x2 = min(w - 1, x1 + 1)
    if y2 <= y1:
        y2 = min(h - 1, y1 + 1)
    return [x1, y1, x2, y2]


def pad_bbox(b, w, h, pad_ratio=0.15):
    x1, y1, x2, y2 = b
    bw = x2 - x1
    bh = y2 - y1
    px = int(bw * pad_ratio)
    py = int(bh * pad_ratio)
    return clamp_bbox([x1 - px, y1 - py, x2 + px, y2 + py], w, h)


def is_sane_lesion_box(bbox, frame_w, frame_h):
    x1, y1, x2, y2 = bbox
    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)

    if bw < MIN_BOX_SIZE_PX or bh < MIN_BOX_SIZE_PX:
        return False

    area = bw * bh
    frac = area / float(frame_w * frame_h)
    if frac < MIN_BOX_AREA_FRAC or frac > MAX_BOX_AREA_FRAC:
        return False

    mx = frame_w * MAX_EDGE_TOUCH_FRAC
    my = frame_h * MAX_EDGE_TOUCH_FRAC
    if x1 <= mx or y1 <= my or x2 >= (frame_w - mx) or y2 >= (frame_h - my):
        return False

    return True

# -----------------------------
# LOAD MODELS ONCE
# -----------------------------
def load_models():
    global yolo_model, cls_model, cls_classes, cls_tfm, cls_device

    from ultralytics import YOLO
    yolo_model = YOLO(YOLO_WEIGHTS)

    if os.path.exists(CLS_CKPT):
        import torch
        import torchvision
        from torchvision import transforms

        ckpt = torch.load(CLS_CKPT, map_location="cpu")
        cls_classes = ckpt.get("classes")
        state = ckpt.get("model_state")

        if cls_classes and state:
            cls_model = torchvision.models.resnet18(weights=None)
            cls_model.fc = torch.nn.Linear(cls_model.fc.in_features, len(cls_classes))
            cls_model.load_state_dict(state)
            cls_model.eval()

            cls_device = "cuda" if torch.cuda.is_available() else "cpu"
            cls_model.to(cls_device)

            cls_tfm = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])
        else:
            cls_model = None

load_models()

# -----------------------------
# DETECTION
# -----------------------------
def yolo_detect(frame_bgr):
    res = yolo_model.predict(frame_bgr, conf=CONF_DET, verbose=False)
    if not res or res[0].boxes is None:
        return []

    boxes = res[0].boxes.xyxy.cpu().numpy()
    confs = res[0].boxes.conf.cpu().numpy()

    return [{"bbox": b.tolist(), "conf": float(c)} for b, c in zip(boxes, confs)]


def classify_crop(crop_bgr):
    import torch
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    x = cls_tfm(crop_rgb).unsqueeze(0).to(cls_device)

    with torch.no_grad():
        logits = cls_model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0)
        conf, idx = torch.max(probs, dim=0)
        return cls_classes[int(idx)], float(conf)


def analyze_frame(frame_bgr):
    h, w = frame_bgr.shape[:2]
    dets = yolo_detect(frame_bgr)
    dets = [d for d in dets if is_sane_lesion_box(d["bbox"], w, h)]

    results = []
    for d in dets:
        b = pad_bbox(clamp_bbox(d["bbox"], w, h), w, h)
        x1, y1, x2, y2 = b
        crop = frame_bgr[y1:y2, x1:x2]

        label = "Unknown"
        label_conf = 0.0

        if cls_model and crop.shape[0] >= MIN_CROP_SIZE_PX:
            pred_label, pred_conf = classify_crop(crop)
            if pred_conf >= CONF_CLS:
                label = pred_label
                label_conf = pred_conf

        results.append({
            "bbox": [x1, y1, x2, y2],
            "det_conf": float(d["conf"]),
            "label": label,
            "label_conf": float(label_conf)
        })

    return results

# -----------------------------
# ROUTES
# -----------------------------
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/analyze-image", methods=["POST"])
def analyze_image():
    name = request.form.get("name", "")
    age = request.form.get("age", "")

    file = request.files.get("image")
    if not file:
        return jsonify({"error": "No image uploaded"}), 400

    os.makedirs("uploads", exist_ok=True)
    path = os.path.join("uploads", file.filename)
    file.save(path)

    frame = cv2.imread(path)
    detections = analyze_frame(frame)

    return jsonify({
        "name": name,
        "age": age,
        "num_detections": len(detections),
        "detections": detections
    })


@app.route("/generate-report", methods=["POST"])
def generate_report():
    data = request.json
    name = data.get("name", "Patient")
    age = data.get("age", "Unknown")
    detections = data.get("detections", [])

    if not detections:
        summary = "No clear skin lesions were detected."
    else:
        labels = [d["label"] for d in detections if d["label"] != "Unknown"]
        summary = f"Possible skin conditions observed: {', '.join(set(labels)) or 'Uncertain lesions'}."

    prompt = f"""
You are a medical AI assistant.

Patient name: {name}
Age: {age}

Skin analysis:
{summary}

Write a human-friendly report with:
1. Simple explanation
2. Possible causes
3. Prevention tips
4. What NOT to do
5. When to see a dermatologist

Rules:
- No prescriptions
- No certainty claims
- Include a medical disclaimer
"""

    response = openai_client.responses.create(
        model="gpt-4.1-mini",
        input=prompt,
        max_output_tokens=500
    )

    return jsonify({"report": response.output_text})


if __name__ == "__main__":
    app.run(debug=True)
