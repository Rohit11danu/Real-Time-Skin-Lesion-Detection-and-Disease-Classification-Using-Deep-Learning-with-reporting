import os
import cv2
import json
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import numpy as np

# -----------------------------
# CONFIG
# -----------------------------
YOLO_WEIGHTS = "./weights/lesion_yolov8n_v3.pt"

# Detector filters (for far lesions)
CONF_DET = 0.30
IOU_MATCH_THR = 0.30

# Sanity filters (RELAXED for far lesions)
MIN_BOX_AREA_FRAC = 0.00010   # allow smaller boxes
MAX_BOX_AREA_FRAC = 0.25
MAX_EDGE_TOUCH_FRAC = 0.02
MIN_BOX_SIZE_PX = 8

EMA_ALPHA = 0.70

USE_CLASSIFIER = True
CLS_CKPT = "./face_skin_disease_resnet18.pth"
CONF_CLS = 0.70
PREDICT_EVERY_N_FRAMES = 2
MIN_CROP_SIZE_PX = 40
CROP_PAD_RATIO = 0.15

VOTE_WINDOW = 10
VOTE_WIN_COUNT = 6
DEAD_AFTER_MISSED_FRAMES = 30

LOG_PATH = "./lesion_log.jsonl"

SHOW_FPS = True
DRAW_TOPK = False
TOPK = 3

# NEW: face ROI zoom (no mediapipe needed)
USE_FACE_ROI = True
FACE_PAD_RATIO = 0.25
YOLO_IMGSZ = 960

# -----------------------------
# UTILITIES
# -----------------------------
def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()) + f".{int((time.time()%1)*1000):03d}"

def clamp_bbox(b: List[float], w: int, h: int) -> List[int]:
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

def iou_xyxy(a: List[float], b: List[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0

def ema_bbox(old: List[float], new: List[float], alpha: float) -> List[float]:
    return [
        alpha * old[0] + (1 - alpha) * new[0],
        alpha * old[1] + (1 - alpha) * new[1],
        alpha * old[2] + (1 - alpha) * new[2],
        alpha * old[3] + (1 - alpha) * new[3],
    ]

def pad_bbox(b: List[int], w: int, h: int, pad_ratio: float = 0.15) -> List[int]:
    x1, y1, x2, y2 = b
    bw = x2 - x1
    bh = y2 - y1
    px = int(bw * pad_ratio)
    py = int(bh * pad_ratio)
    return clamp_bbox([x1 - px, y1 - py, x2 + px, y2 + py], w, h)

def is_sane_lesion_box(bbox: List[float], frame_w: int, frame_h: int) -> bool:
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
# Face ROI using OpenCV Haarcascade
# -----------------------------
def load_face_detector():
    try:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        if os.path.exists(cascade_path):
            return cv2.CascadeClassifier(cascade_path)
    except Exception:
        pass
    return None

FACE_CASCADE = load_face_detector()

def get_face_roi(frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    if not USE_FACE_ROI or FACE_CASCADE is None:
        return None
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
    if len(faces) == 0:
        return None

    # choose largest face
    x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
    pad = int(FACE_PAD_RATIO * max(fw, fh))
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(w - 1, x + fw + pad)
    y2 = min(h - 1, y + fh + pad)
    if x2 <= x1 + 5 or y2 <= y1 + 5:
        return None
    return (x1, y1, x2, y2)

# -----------------------------
# TRACK STATE
# -----------------------------
@dataclass
class Track:
    bbox: List[float]
    det_conf: float = 0.0
    label: str = "Unknown"
    label_conf: float = 0.0
    label_hist: deque = field(default_factory=lambda: deque(maxlen=VOTE_WINDOW))
    stable_label: str = "Unknown"
    last_seen_frame: int = 0
    last_pred_frame: int = -9999
    topk: List[Tuple[str, float]] = field(default_factory=list)

# -----------------------------
# OPTIONAL CLASSIFIER (ResNet18)
# -----------------------------
def load_classifier():
    import torch
    import torchvision
    from torchvision import transforms

    if not os.path.exists(CLS_CKPT):
        print(f"[WARN] Classifier checkpoint not found: {CLS_CKPT}")
        return None, None, None, None

    ckpt = torch.load(CLS_CKPT, map_location="cpu")
    classes = ckpt.get("classes", None)
    state = ckpt.get("model_state", None)

    if classes is None or state is None:
        print("[WARN] Checkpoint format unexpected. Need keys: 'classes' and 'model_state'.")
        return None, None, None, None

    model = torchvision.models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(state)
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    tfm = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return model, classes, tfm, device

def classify_crop(model, classes, tfm, device, crop_bgr: np.ndarray, topk: int = 3):
    import torch
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    x = tfm(crop_rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0)
        conf, idx = torch.max(probs, dim=0)
        conf = float(conf.item())
        idx = int(idx.item())
        label = classes[idx]
        tk = min(topk, probs.numel())
        top_probs, top_idx = torch.topk(probs, k=tk)
        top_list = [(classes[int(i)], float(p)) for p, i in zip(top_probs.cpu().numpy(), top_idx.cpu().numpy())]
    return label, conf, top_list

# -----------------------------
# YOLO DETECTOR
# -----------------------------
def load_yolo():
    from ultralytics import YOLO
    if not os.path.exists(YOLO_WEIGHTS):
        raise FileNotFoundError(f"YOLO weights not found at: {YOLO_WEIGHTS}")
    return YOLO(YOLO_WEIGHTS)

def yolo_detect(model, frame_bgr: np.ndarray):
    res = model.predict(frame_bgr, imgsz=YOLO_IMGSZ, conf=CONF_DET, verbose=False)
    if not res:
        return []
    r0 = res[0]
    if r0.boxes is None or len(r0.boxes) == 0:
        return []
    boxes = r0.boxes.xyxy.cpu().numpy()
    confs = r0.boxes.conf.cpu().numpy()
    return [{"bbox": [float(b[0]), float(b[1]), float(b[2]), float(b[3])], "conf": float(c)} for b, c in zip(boxes, confs)]

# -----------------------------
# MAIN
# -----------------------------
def main():
    print("[DEBUG] UPDATED CODE LOADED ✅")
    print("[INFO] Starting Module 4 live (stable + sane filters + face ROI)...")
    print(f"[INFO] YOLO_WEIGHTS = {YOLO_WEIGHTS}")
    print(f"[INFO] CONF_DET={CONF_DET} MIN_AREA={MIN_BOX_AREA_FRAC} MIN_PX={MIN_BOX_SIZE_PX} imgsz={YOLO_IMGSZ}")
    print(f"[INFO] Face ROI enabled={USE_FACE_ROI} Haarcascade loaded={FACE_CASCADE is not None}")
    print(f"[INFO] Logging to {LOG_PATH}")

    yolo = load_yolo()

    cls_model = cls_classes = cls_tfm = cls_device = None
    if USE_CLASSIFIER:
        cls_model, cls_classes, cls_tfm, cls_device = load_classifier()
        if cls_model is None:
            print("[WARN] Classifier disabled (could not load).")
        else:
            print(f"[INFO] Loaded classifier with {len(cls_classes)} classes on {cls_device}")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam (index 0). Try VideoCapture(1).")

    tracked: Dict[int, Track] = {}
    next_id = 1
    frame_idx = 0

    log_f = open(LOG_PATH, "a", encoding="utf-8")
    prev_time = time.time()
    fps = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame_idx += 1
            H, W = frame.shape[:2]

            roi = get_face_roi(frame)
            if roi is not None:
                fx1, fy1, fx2, fy2 = roi
                roi_frame = frame[fy1:fy2, fx1:fx2]
                roi_h, roi_w = roi_frame.shape[:2]
                # debug ROI box
                cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), (255, 0, 0), 2)
            else:
                fx1 = fy1 = 0
                roi_frame = frame
                roi_h, roi_w = H, W

            dets = yolo_detect(yolo, roi_frame)
            dets = [d for d in dets if is_sane_lesion_box(d["bbox"], roi_w, roi_h)]

            assigned_track_ids = set()
            det_to_id: Dict[int, int] = {}

            for i, d in enumerate(dets):
                best_iou = 0.0
                best_id = None
                for tid, tr in tracked.items():
                    if tid in assigned_track_ids:
                        continue
                    val = iou_xyxy(d["bbox"], tr.bbox)
                    if val > best_iou:
                        best_iou = val
                        best_id = tid

                if best_id is not None and best_iou > IOU_MATCH_THR:
                    det_to_id[i] = best_id
                    assigned_track_ids.add(best_id)
                else:
                    tid = next_id
                    next_id += 1
                    tracked[tid] = Track(bbox=d["bbox"], last_seen_frame=frame_idx)
                    det_to_id[i] = tid
                    assigned_track_ids.add(tid)

            for i, d in enumerate(dets):
                tid = det_to_id[i]
                tr = tracked[tid]
                tr.bbox = ema_bbox(tr.bbox, d["bbox"], EMA_ALPHA)
                tr.det_conf = float(d["conf"])
                tr.last_seen_frame = frame_idx

            for i, d in enumerate(dets):
                tid = det_to_id[i]
                tr = tracked[tid]

                do_pred = cls_model is not None and (frame_idx - tr.last_pred_frame) >= PREDICT_EVERY_N_FRAMES

                raw_label = tr.label
                raw_conf = tr.label_conf

                if do_pred:
                    b_int = clamp_bbox(tr.bbox, roi_w, roi_h)
                    b_pad = pad_bbox(b_int, roi_w, roi_h, pad_ratio=CROP_PAD_RATIO)
                    x1, y1, x2, y2 = b_pad
                    crop = roi_frame[y1:y2, x1:x2]
                    if crop.shape[0] >= MIN_CROP_SIZE_PX and crop.shape[1] >= MIN_CROP_SIZE_PX:
                        pred_label, pred_conf, _ = classify_crop(cls_model, cls_classes, cls_tfm, cls_device, crop, topk=TOPK)
                        raw_label = pred_label
                        raw_conf = float(pred_conf)
                        tr.last_pred_frame = frame_idx

                display_label = raw_label if raw_conf >= CONF_CLS else "Unknown"
                tr.label = display_label
                tr.label_conf = raw_conf

                tr.label_hist.append(display_label)
                if len(tr.label_hist) > 0:
                    most, cnt = Counter(tr.label_hist).most_common(1)[0]
                    tr.stable_label = most if cnt >= VOTE_WIN_COUNT else "Unknown"
                else:
                    tr.stable_label = "Unknown"

            dead_ids = [tid for tid, tr in tracked.items() if (frame_idx - tr.last_seen_frame) > DEAD_AFTER_MISSED_FRAMES]
            for tid in dead_ids:
                del tracked[tid]

            for i, d in enumerate(dets):
                tid = det_to_id[i]
                tr = tracked[tid]
                b = clamp_bbox(tr.bbox, roi_w, roi_h)
                x1, y1, x2, y2 = b

                x1 += fx1; x2 += fx1
                y1 += fy1; y2 += fy1

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"ID:{tid} {tr.stable_label}", (x1, max(0, y1 - 24)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame, f"det:{tr.det_conf:.2f} cls:{tr.label_conf:.2f}", (x1, max(0, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                rec = {
                    "time": now_iso(),
                    "frame": frame_idx,
                    "lesion_id": tid,
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "det_conf": float(tr.det_conf),
                    "label": tr.label,
                    "label_conf": float(tr.label_conf),
                    "stable_label": tr.stable_label,
                }
                log_f.write(json.dumps(rec) + "\n")

            if SHOW_FPS:
                cur = time.time()
                dt = cur - prev_time
                prev_time = cur
                if dt > 0:
                    fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps > 0 else (1.0 / dt)
                cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            cv2.imshow("Module 4 Live (Stable)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break

    finally:
        log_f.close()
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Exited cleanly.")


if __name__ == "__main__":
    main()
