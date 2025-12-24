import json
import sys
import os

if len(sys.argv) < 2:
    print("Usage: python3 make_ai_report_prompt.py <session_folder>")
    sys.exit(1)

session = sys.argv[1]

def read(path):
    if not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

report = read(os.path.join(session, "module5_report.txt"))
summary = read(os.path.join(session, "module5_summary.json"))
alerts = read(os.path.join(session, "module5_alerts.json"))

prompt = f"""
You are a medical assistant that summarizes AI skin-lesion detection results.

RULES:
- DO NOT diagnose or prescribe medication
- DO give prevention, care tips, what NOT to do
- DO include red-flag symptoms
- State uncertainty clearly

INPUT DATA:

MODULE 5 REPORT:
{report}

SUMMARY JSON:
{summary}

ALERTS JSON:
{alerts}

OUTPUT FORMAT:
A) Simple summary
B) What was detected
C) What this could mean (possibilities only)
D) Prevention & skin care
E) What NOT to do
F) OTC-safe options only (no prescriptions)
G) Red flags (see doctor urgently)
H) Next steps checklist
"""

out_path = os.path.join(session, "ai_prompt.txt")
with open(out_path, "w", encoding="utf-8") as f:
    f.write(prompt)

print(f"[OK] AI prompt written to {out_path}")

