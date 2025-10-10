import time
import subprocess
from datetime import datetime
import sys
sys.stdout.reconfigure(encoding='utf-8')

# =============== CONFIG ===============
INTERVAL_MINUTES = 0.5  # 每隔 60 分钟运行一次
SCRIPT_NAME = "src/news.py"
# ======================================


def run_once():
    """Run the news sentiment script once."""
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]  Running {SCRIPT_NAME} ...")
    try:
        # 使用 subprocess 调用 news.py
        result = subprocess.run(
            ["python", SCRIPT_NAME],
            capture_output=True,
            text=True,
            timeout=1800  # 最多运行 30 分钟
        )
        if result.returncode == 0:
            print(f"Completed successfully at {datetime.now().strftime('%H:%M:%S')}")
        else:
            print(f"Script exited with error code {result.returncode}")
        print(result.stdout)
        if result.stderr:
            print("stderr:", result.stderr)
    except subprocess.TimeoutExpired:
        print("Timeout: news.py took too long and was stopped.")
    except Exception as e:
        print("Unexpected error while running script:", e)


def main():
    print("Starting automated FinBERT news monitoring...\n")
    while True:
        run_once()
        print(f"Sleeping for {INTERVAL_MINUTES} minutes...\n")
        time.sleep(INTERVAL_MINUTES * 60)


if __name__ == "__main__":
    main()