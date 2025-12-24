"""
Module 6 — Patient Session, Persistence & Final Report
(UPDATED: receives name, age, datetime from controller)

Run (normally via run_dermnet.py):
python3 module6_session.py <name> <age> <datetime>
"""

import json
import os
import sys

# -----------------------------
# ARGUMENTS
# -----------------------------
if len(sys.argv) < 4:
    raise SystemExit("Usage: python3 module6_session.py <name> <age> <datetime>")

name = sys.argv[1]
age = sys.argv[2]
session_datetime = sys.argv[3]

patient_id = name.lower().replace(" ", "_")
session_date = session_datetime.split(" ")[0]

# -----------------------------
# PATHS
# -----------------------------
DATA_DIR = "data"
PATIENT_DIR = os.path.join(DATA_DIR, "patients")
SESSION_DIR = os.path.join(DATA_DIR, "sessions")
REPORT_DIR = os.path.join(DATA_DIR, "reports")

SUMMARY_FILE = "module5_summary.json"
ALERT_FILE = "module5_alerts.json"

os.makedirs(PATIENT_DIR, exist_ok=True)
os.makedirs(SESSION_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# -----------------------------
# LOAD MODULE 5 OUTPUTS
# -----------------------------
def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

if not os.path.exists(SUMMARY_FILE):
    raise SystemExit("module5_summary.json not found. Run Module 5 first.")

summary = load_json(SUMMARY_FILE)
alerts = load_json(ALERT_FILE) if os.path.exists(ALERT_FILE) else []

# -----------------------------
# CREATE SESSION OBJECT
# -----------------------------
session = {
    "datetime": session_datetime,
    "total_lesions": len(summary),
    "alerts": alerts,
    "lesions": summary
}

session_path = os.path.join(SESSION_DIR, f"session_{session_date}.json")
with open(session_path, "w") as f:
    json.dump(session, f, indent=2)

# -----------------------------
# LOAD / UPDATE PATIENT HISTORY
# -----------------------------
patient_path = os.path.join(PATIENT_DIR, f"{patient_id}.json")

if os.path.exists(patient_path):
    with open(patient_path, "r") as f:
        patient_data = json.load(f)
else:
    patient_data = {
        "patient": {
            "name": name,
            "age": age
        },
        "sessions": []
    }

patient_data["sessions"].append(session)

with open(patient_path, "w") as f:
    json.dump(patient_data, f, indent=2)

# -----------------------------
# GENERATE FINAL REPORT
# -----------------------------
lines = []
lines.append("DERMNET FINAL REPORT")
lines.append("=" * 50)
lines.append(f"Patient Name : {name}")
lines.append(f"Age          : {age}")
lines.append(f"Session Time : {session_datetime}")
lines.append("")
lines.append(f"Total Lesions Detected: {len(summary)}")
lines.append(f"Alerts Raised: {len(alerts)}")
lines.append("")

if alerts:
    lines.append("ALERTS")
    lines.append("-" * 30)
    for a in alerts:
        lines.append(
            f"Lesion {a['lesion_id']} | {a['stable_label']} | "
            f"Severity: {a['severity']} | Trend: {a['trend']}"
        )
    lines.append("")

lines.append("LESION DETAILS")
lines.append("-" * 30)
for l in summary:
    lines.append(
        f"Lesion {l['lesion_id']} | "
        f"Disease: {l['stable_label']} | "
        f"Severity: {l['severity']} | "
        f"Trend: {l['trend']}"
    )

report_path = os.path.join(REPORT_DIR, f"{patient_id}_report.txt")
with open(report_path, "w") as f:
    f.write("\n".join(lines))

# -----------------------------
# DONE
# -----------------------------
print("\n✅ Module 6 completed")
print(f"📁 Patient file : {patient_path}")
print(f"📁 Session file : {session_path}")
print(f"📄 Report saved : {report_path}")
