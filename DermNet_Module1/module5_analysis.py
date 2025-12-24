"""
Module 5 — Lesion History, Trends, Severity, Alerts (Offline Analysis)

Input:
  - lesion_log.jsonl  (produced by Module 4)

Output:
  - module5_summary.json     (per-lesion summary)
  - module5_alerts.json      (alerts list)
  - module5_timeseries.json  (per-lesion time series)
  - module5_report.txt       (human-readable report)

Run:
  python3 module5_analysis.py --log lesion_log.jsonl

Notes:
- This module does NOT change live camera code.
- It reads the stable output and computes:
  * lesion size over time (bbox area)
  * trend (improving/stable/worsening)
  * severity score & bucket (Low/Medium/High)
  * alerts (worsening + high severity or risky labels)
"""

import argparse
import json
import math
from collections import defaultdict, Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional


# -----------------------------
# CONFIG (tune if needed)
# -----------------------------
LABEL_WEIGHTS = {
    "Acne": 1.0,
    "Eczemaa": 2.0,
    "Eczema": 2.0,  # in case your label spelling differs
    "Rosacea": 2.0,
    "Actinic Keratosis": 3.0,
    "Basal Cell Carcinoma": 4.0,
    "Unknown": 0.0
}

# Trend thresholds based on area change
WORSEN_FACTOR = 1.20  # last > first * 1.20 => worsening
IMPROVE_FACTOR = 0.80 # last < first * 0.80 => improving

# Severity bucket thresholds (based on severity_score defined below)
SEVERITY_LOW_MAX = 1.5
SEVERITY_MED_MAX = 3.0

# Alert rules
ALERT_IF_WORSENING_AND_SEVERITY_AT_LEAST = "Medium"  # Medium or High
ALERT_RISKY_LABELS = {"Basal Cell Carcinoma", "Actinic Keratosis"}  # always alert if stable label is one of these

# Optional: ignore very short-lived tracks (in seconds)
MIN_TRACK_DURATION_SEC = 2.0


# -----------------------------
# HELPERS
# -----------------------------
def parse_time_to_seconds(ts: str) -> Optional[float]:
    """
    Parses timestamps like: '2025-12-21T00:35:16.244'
    Returns seconds since epoch-ish? We only need relative durations, so we parse into (h,m,s,ms) if possible.
    If parsing fails, returns None.
    """
    # Minimal parsing without external deps
    try:
        # split date/time
        if "T" not in ts:
            return None
        date_part, time_part = ts.split("T", 1)
        # time_part: HH:MM:SS.mmm
        if "." in time_part:
            hms, ms = time_part.split(".", 1)
            ms = int(ms[:3])
        else:
            hms = time_part
            ms = 0
        hh, mm, ss = hms.split(":")
        hh = int(hh); mm = int(mm); ss = int(ss)
        # convert into "seconds from midnight" (good enough for same-session logs)
        return hh * 3600 + mm * 60 + ss + ms / 1000.0
    except Exception:
        return None


def bbox_area(b: List[int]) -> float:
    x1, y1, x2, y2 = b
    return max(0, x2 - x1) * max(0, y2 - y1)


def median(values: List[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    mid = n // 2
    if n % 2 == 1:
        return float(s[mid])
    return float((s[mid - 1] + s[mid]) / 2.0)


def mean(values: List[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def severity_bucket(score: float) -> str:
    if score <= SEVERITY_LOW_MAX:
        return "Low"
    if score <= SEVERITY_MED_MAX:
        return "Medium"
    return "High"


def bucket_rank(b: str) -> int:
    return {"Low": 1, "Medium": 2, "High": 3}.get(b, 0)


@dataclass
class LesionSummary:
    lesion_id: int
    stable_label: str
    duration_sec: float
    frames: int
    area_first: float
    area_last: float
    area_mean: float
    area_median: float
    trend: str
    label_conf_mean: float
    det_conf_mean: float
    severity_score: float
    severity: str


# -----------------------------
# CORE ANALYSIS
# -----------------------------
def load_logs(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                # skip bad lines
                continue
    return rows


def group_by_lesion(rows: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    grouped = defaultdict(list)
    for r in rows:
        if "lesion_id" not in r:
            continue
        grouped[int(r["lesion_id"])].append(r)
    # keep order by frame if present
    for k in grouped:
        grouped[k].sort(key=lambda x: x.get("frame", 0))
    return grouped


def compute_stable_label(records: List[Dict[str, Any]]) -> str:
    labels = [r.get("stable_label", "Unknown") for r in records]
    if not labels:
        return "Unknown"
    return Counter(labels).most_common(1)[0][0]


def compute_duration(records: List[Dict[str, Any]]) -> float:
    # prefer timestamp parsing; fallback to frame count
    times = [parse_time_to_seconds(r.get("time", "")) for r in records]
    times = [t for t in times if t is not None]
    if len(times) >= 2:
        return float(max(times) - min(times))
    # fallback: assume ~30fps if no valid timestamps
    frames = [r.get("frame") for r in records if isinstance(r.get("frame"), int)]
    if len(frames) >= 2:
        return float(max(frames) - min(frames)) / 30.0
    return float(len(records)) / 30.0


def compute_trend(area_first: float, area_last: float) -> str:
    if area_first <= 0:
        return "stable"
    ratio = area_last / area_first
    if ratio >= WORSEN_FACTOR:
        return "worsening"
    if ratio <= IMPROVE_FACTOR:
        return "improving"
    return "stable"


def compute_severity_score(stable_label: str, area_stat: float, det_conf_mean: float, label_conf_mean: float) -> float:
    """
    Simple, explainable severity score:
    - base weight from label (risk category)
    - scaled by size proxy (log area)
    - boosted slightly by confidence (so low-confidence doesn't over-alert)
    """
    w = LABEL_WEIGHTS.get(stable_label, LABEL_WEIGHTS["Unknown"])
    # log scaling for area so huge outliers don't dominate
    size_term = math.log1p(max(0.0, area_stat)) / 10.0  # ~0..1-ish depending on image scale
    conf_term = 0.5 * det_conf_mean + 0.5 * label_conf_mean  # 0..1
    # Score is weight * (0.5 + size_term) * (0.5 + conf_term/2)
    return float(w * (0.5 + size_term) * (0.5 + conf_term / 2.0))


def make_summary(lesion_id: int, records: List[Dict[str, Any]]) -> LesionSummary:
    stable = compute_stable_label(records)

    areas = [bbox_area(r.get("bbox", [0, 0, 0, 0])) for r in records]
    area_first = float(areas[0]) if areas else 0.0
    area_last = float(areas[-1]) if areas else 0.0
    area_mean = mean(areas)
    area_med = median(areas)

    duration = compute_duration(records)
    trend = compute_trend(area_first, area_last)

    det_confs = [float(r.get("det_conf", 0.0)) for r in records]
    cls_confs = [float(r.get("label_conf", 0.0)) for r in records]
    det_conf_mean = mean(det_confs)
    label_conf_mean = mean(cls_confs)

    # use median area as stable size proxy
    score = compute_severity_score(stable, area_med, det_conf_mean, label_conf_mean)
    sev = severity_bucket(score)

    return LesionSummary(
        lesion_id=lesion_id,
        stable_label=stable,
        duration_sec=float(duration),
        frames=len(records),
        area_first=area_first,
        area_last=area_last,
        area_mean=area_mean,
        area_median=area_med,
        trend=trend,
        label_conf_mean=label_conf_mean,
        det_conf_mean=det_conf_mean,
        severity_score=score,
        severity=sev,
    )


def make_alerts(summaries: List[LesionSummary]) -> List[Dict[str, Any]]:
    alerts = []
    for s in summaries:
        if s.duration_sec < MIN_TRACK_DURATION_SEC:
            continue

        reason = []

        # risky label always alerts
        if s.stable_label in ALERT_RISKY_LABELS:
            reason.append(f"risky_label:{s.stable_label}")

        # worsening + medium/high severity alerts
        if s.trend == "worsening":
            if bucket_rank(s.severity) >= bucket_rank(ALERT_IF_WORSENING_AND_SEVERITY_AT_LEAST):
                reason.append(f"worsening_and_severity:{s.severity}")

        if reason:
            alerts.append({
                "lesion_id": s.lesion_id,
                "stable_label": s.stable_label,
                "severity": s.severity,
                "severity_score": round(s.severity_score, 4),
                "trend": s.trend,
                "duration_sec": round(s.duration_sec, 2),
                "reasons": reason
            })
    return alerts


def export_timeseries(grouped: Dict[int, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """
    Builds a compact time-series structure per lesion for dashboards/plots later.
    """
    out = {}
    for lesion_id, recs in grouped.items():
        series = []
        for r in recs:
            b = r.get("bbox", [0, 0, 0, 0])
            series.append({
                "time": r.get("time"),
                "frame": r.get("frame"),
                "bbox": b,
                "area": bbox_area(b),
                "det_conf": r.get("det_conf"),
                "stable_label": r.get("stable_label"),
                "label_conf": r.get("label_conf"),
            })
        out[str(lesion_id)] = series
    return out


def write_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def write_report(path: str, summaries: List[LesionSummary], alerts: List[Dict[str, Any]]) -> None:
    lines = []
    lines.append("MODULE 5 REPORT")
    lines.append("=" * 60)
    lines.append(f"Total lesions tracked: {len(summaries)}")
    lines.append(f"Alerts: {len(alerts)}")
    lines.append("")

    if alerts:
        lines.append("ALERTS")
        lines.append("-" * 60)
        for a in alerts:
            lines.append(
                f"Lesion {a['lesion_id']}: {a['stable_label']} | severity={a['severity']} "
                f"(score={a['severity_score']}) | trend={a['trend']} | reasons={', '.join(a['reasons'])}"
            )
        lines.append("")

    lines.append("LESION SUMMARIES")
    lines.append("-" * 60)
    # sort by severity score desc
    for s in sorted(summaries, key=lambda x: x.severity_score, reverse=True):
        lines.append(
            f"Lesion {s.lesion_id}: {s.stable_label} | severity={s.severity} (score={s.severity_score:.4f}) "
            f"| trend={s.trend} | duration={s.duration_sec:.2f}s | frames={s.frames} "
            f"| area(first->last)={s.area_first:.0f}->{s.area_last:.0f}"
        )

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# -----------------------------
# CLI
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="lesion_log.jsonl", help="Path to Module 4 JSONL log")
    ap.add_argument("--out_prefix", default="module5", help="Output prefix")
    args = ap.parse_args()

    rows = load_logs(args.log)
    if not rows:
        raise SystemExit(f"No valid JSON lines found in {args.log}")

    grouped = group_by_lesion(rows)

    summaries: List[LesionSummary] = []
    for lesion_id, recs in grouped.items():
        s = make_summary(lesion_id, recs)
        if s.duration_sec < MIN_TRACK_DURATION_SEC:
            # skip very short tracks
            continue
        summaries.append(s)

    alerts = make_alerts(summaries)

    # JSON export
    summary_json = [
        {
            "lesion_id": s.lesion_id,
            "stable_label": s.stable_label,
            "duration_sec": round(s.duration_sec, 3),
            "frames": s.frames,
            "trend": s.trend,
            "severity": s.severity,
            "severity_score": round(s.severity_score, 6),
            "det_conf_mean": round(s.det_conf_mean, 4),
            "label_conf_mean": round(s.label_conf_mean, 4),
            "area_first": round(s.area_first, 2),
            "area_last": round(s.area_last, 2),
            "area_mean": round(s.area_mean, 2),
            "area_median": round(s.area_median, 2),
        }
        for s in summaries
    ]

    timeseries = export_timeseries(grouped)

    write_json(f"{args.out_prefix}_summary.json", summary_json)
    write_json(f"{args.out_prefix}_alerts.json", alerts)
    write_json(f"{args.out_prefix}_timeseries.json", timeseries)

    write_report(f"{args.out_prefix}_report.txt", summaries, alerts)

    print(f"[OK] Wrote: {args.out_prefix}_summary.json")
    print(f"[OK] Wrote: {args.out_prefix}_alerts.json")
    print(f"[OK] Wrote: {args.out_prefix}_timeseries.json")
    print(f"[OK] Wrote: {args.out_prefix}_report.txt")


if __name__ == "__main__":
    main()
