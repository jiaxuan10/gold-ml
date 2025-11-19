import subprocess
import time
import os
import sys

# Paths to your scripts
FETCHER_SCRIPT = os.path.join("src", "live", "realtime_fetcher.py")
INFERENCE_SCRIPT = os.path.join("src", "live", "inference_service.py")
DASHBOARD_SCRIPT = os.path.join("streamlit_app", "app.py")

processes = []

def run_process(command, title):
    print(f"🚀 Starting {title}...")
    # Use 'start' on Windows to open new command windows for each service
    if sys.platform == "win32":
        p = subprocess.Popen(f"start cmd /k {command}", shell=True)
    else:
        p = subprocess.Popen(command, shell=True)
    processes.append(p)

try:
    # 1. Start Data Fetcher
    run_process(f"python {FETCHER_SCRIPT}", "Data Fetcher")
    time.sleep(5) 

    # 2. Start AI Auto-Trader
    run_process(f"python {INFERENCE_SCRIPT}", "AI Trader")
    
    # 3. Start Dashboard
    print("🚀 Starting Dashboard...")
    subprocess.run(f"streamlit run {DASHBOARD_SCRIPT}", shell=True)

except KeyboardInterrupt:
    print("Stopping...")