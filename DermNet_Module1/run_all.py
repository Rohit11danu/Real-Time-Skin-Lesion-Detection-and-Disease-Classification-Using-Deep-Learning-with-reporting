import os
import shutil
import subprocess
import sys
import time

LOG = "lesion_log.jsonl"
ANALYZER = "module5_analysis.py"
LIVE = "module3_live.py"

OUT_FILES = [
    "module5_report.txt",
    "module5_summary.json",
    "module5_alerts.json",
    "module5_timeseries.json",
]

def timestamp():
    return time.strftime("%Y%m%d_%H%M%S")

def file_nonempty(path: str) -> bool:
    return os.path.exists(path) and os.path.getsize(path) > 0

def run(cmd):
    print(f"\n[RUN] {' '.join(cmd)}\n")
    ret = subprocess.call(cmd)
    if ret != 0:
        raise SystemExit(f"[ERROR] Command failed with code {ret}: {' '.join(cmd)}")

def main():
    ts = timestamp()

    # 1) Backup old log if it exists
    if os.path.exists(LOG):
        backup = f"lesion_log_backup_{ts}.jsonl"
        print(f"[INFO] Backing up existing {LOG} -> {backup}")
        os.rename(LOG, backup)

    # 2) Run live module (you quit with 'q')
    print("[INFO] Starting LIVE detection. Press 'q' in the window to stop.")
    run([sys.executable, LIVE])

    # 3) Make sure log was created
    if not file_nonempty(LOG):
        raise SystemExit(
            f"[ERROR] {LOG} not created or empty.\n"
            f"Make sure module3_live.py is writing logs (LOG_PATH = './lesion_log.jsonl')."
        )

    # 4) Run analysis
    print("[INFO] Running analysis...")
    run([sys.executable, ANALYZER, "--log", LOG])

    # 5) Archive this session’s log + outputs
    session_dir = f"session_{ts}"
    os.makedirs(session_dir, exist_ok=True)

    print(f"[INFO] Archiving outputs into {session_dir}/")
    shutil.move(LOG, os.path.join(session_dir, LOG))

    for f in OUT_FILES:
        if os.path.exists(f):
            shutil.move(f, os.path.join(session_dir, f))

    print("\n✅ DONE")
    print(f"[INFO] Session saved in: {session_dir}/")
    print(f"[INFO] Open report: {session_dir}/module5_report.txt")

if __name__ == "__main__":
    main()

