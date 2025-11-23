import subprocess
import time
import os
import sys

# Paths to your scripts
FETCHER_SCRIPT = os.path.join("src", "live", "realtime_fetcher.py")
INFERENCE_SCRIPT = os.path.join("src", "live", "inference_service.py")
DASHBOARD_SCRIPT = os.path.join("streamlit_app", "app.py")
NEWS_SCRIPT = os.path.join("src", "live", "news_service.py") # 🆕

processes = []

def run_process(command, title):
    print(f"🚀 Starting {title}...")
    if sys.platform == "win32":
        p = subprocess.Popen(f"start cmd /k {command}", shell=True)
    else:
        p = subprocess.Popen(command, shell=True)
    processes.append(p)

try:
    # 1. Start News Service (Background) 🆕
    run_process(f"python {NEWS_SCRIPT}", "News Sentiment")
    
    # 2. Start Data Fetcher
    run_process(f"python {FETCHER_SCRIPT}", "Data Fetcher")
    time.sleep(5) 

    # 3. Start AI Auto-Trader
    run_process(f"python {INFERENCE_SCRIPT}", "AI Trader")
    
    # 4. Start Dashboard
    print("🚀 Starting Dashboard...")
    subprocess.run(f"streamlit run {DASHBOARD_SCRIPT}", shell=True)

except KeyboardInterrupt:
    print("Stopping...")