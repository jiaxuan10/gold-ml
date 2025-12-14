import subprocess
import time
import os
import sys

# Paths to your scripts
ROOT = os.path.dirname(os.path.abspath(__file__)) 
FETCHER_SCRIPT = os.path.join("src", "live", "realtime_fetcher.py")
INFERENCE_SCRIPT = os.path.join("src", "live", "inference_service.py")
DASHBOARD_SCRIPT = os.path.join("streamlit_app", "app.py")
NEWS_SCRIPT = os.path.join("src", "live", "news_service.py")

# Sentiment File Path 
SENTIMENT_FILE = os.path.join("data", "final", "current_sentiment.json")

processes = []

def run_process(command, title):
    print(f" Starting {title}...")
    if sys.platform == "win32":
        # /k keep window open
        p = subprocess.Popen(f"start cmd /k {command}", shell=True)
    else:
        p = subprocess.Popen(command, shell=True)
    processes.append(p)

try:
    if os.path.exists(SENTIMENT_FILE):
        print("🧹 Cleaning up old sentiment data...")
        os.remove(SENTIMENT_FILE)

    # 1. Start News Service (Background)
    run_process(f"python {NEWS_SCRIPT}", "News Sentiment Analysis")
    
    print(" Waiting for FinBERT & News Fetch (This takes ~15s)...")
    start_wait = time.time()
    while not os.path.exists(SENTIMENT_FILE):
        time.sleep(1)
        sys.stdout.write(".") 
        sys.stdout.flush()
        
        if time.time() - start_wait > 60:
            print("\n News Service timed out! Starting Trader anyway...")
            break
            
    print(f"\n✅ News Ready! Found: {SENTIMENT_FILE}")

    # 2. Start Data Fetcher
    run_process(f"python {FETCHER_SCRIPT}", "Data Fetcher")
    time.sleep(3) 

    run_process(f"python {INFERENCE_SCRIPT}", "AI Trader")
    
    # 4. Start Dashboard
    print(" Starting Dashboard...")
    subprocess.run(f"streamlit run {DASHBOARD_SCRIPT}", shell=True)

except KeyboardInterrupt:
    print("\n Stopping System...")