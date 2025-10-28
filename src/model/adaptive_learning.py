"""
adaptive_learning.py
--------------------
Adaptive learning system for gold price prediction models.

This version integrates with Lim Jia Xuan's gold ML pipeline:
- Monitors model performance from backtesting logs
- Automatically retrains if performance drops below threshold
- Updates model registry and feature weights
- Compatible with train_xgb.py and simulate_trading.py

Author: Lim Jia Xuan (optimized version)
"""

import os
import time
import pandas as pd
import numpy as np
from datetime import datetime
import subprocess
import pickle
from sklearn.preprocessing import StandardScaler

# ================= CONFIG =================
MODEL_DIR = "models"
BACKTEST_DIR = "backtest_results"
REGISTRY_PATH = os.path.join(MODEL_DIR, "model_registry.csv")
DATA_PATH = "data/final/final_dataset_daily.csv"

# retrain trigger conditions
PERFORMANCE_THRESHOLD = 0.0        # retrain if Sharpe < 0
DRAWDOWN_LIMIT = -10.0             # retrain if max drawdown < -10%
CHECK_INTERVAL_SEC = 3600 * 6      # check every 6 hours

# =========================================================
def load_latest_backtest():
    files = [f for f in os.listdir(BACKTEST_DIR) if f.endswith(".csv")]
    if not files:
        return None
    files.sort(key=lambda x: os.path.getmtime(os.path.join(BACKTEST_DIR, x)), reverse=True)
    latest = files[0]
    df = pd.read_csv(os.path.join(BACKTEST_DIR, latest))
    print(f"📊 Loaded backtest file: {latest} ({len(df)} rows)")
    return df

def compute_performance(df):
    sharpe = np.sqrt(252) * df["strategy_ret"].mean() / (df["strategy_ret"].std() + 1e-9)
    total_return = (df["equity"].iloc[-1] / df["equity"].iloc[0] - 1) * 100
    max_dd = (df["equity"] / df["equity"].cummax() - 1).min() * 100
    win_rate = (df["strategy_ret"] > 0).mean() * 100
    return {
        "sharpe": sharpe,
        "total_return": total_return,
        "max_drawdown": max_dd,
        "win_rate": win_rate
    }

def update_model_registry(model_name, perf_dict):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row = {
        "timestamp": ts,
        "model": model_name,
        "sharpe": perf_dict["sharpe"],
        "total_return": perf_dict["total_return"],
        "max_drawdown": perf_dict["max_drawdown"],
        "win_rate": perf_dict["win_rate"]
    }

    if os.path.exists(REGISTRY_PATH):
        reg = pd.read_csv(REGISTRY_PATH)
        reg = pd.concat([reg, pd.DataFrame([row])], ignore_index=True)
    else:
        reg = pd.DataFrame([row])

    reg.to_csv(REGISTRY_PATH, index=False)
    print(f"📘 Registry updated: {REGISTRY_PATH}")

def retrain_model():
    print("🔁 Retraining model via train_xgb.py ...")
    try:
        subprocess.run(["python", "src/model/train_xgb.py"], check=True)
        print("✅ Retraining complete.")
    except Exception as e:
        print(f"❌ Retraining failed: {e}")

def evaluate_and_adapt():
    print("\n=== Adaptive Learning Cycle Start ===")
    df = load_latest_backtest()
    if df is None:
        print("⚠️ No backtest results found. Skipping check.")
        return

    perf = compute_performance(df)
    print(f"📈 Current model performance:")
    for k, v in perf.items():
        print(f"  {k:<15}: {v:.3f}")

    model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith(".pkl")]
    if not model_files:
        print("⚠️ No model found. Retraining immediately.")
        retrain_model()
        return

    latest_model = sorted(model_files, key=lambda x: os.path.getmtime(os.path.join(MODEL_DIR, x)), reverse=True)[0]
    print(f"🧠 Latest model: {latest_model}")

    # Log performance
    update_model_registry(latest_model, perf)

    # Check if model underperforms
    if perf["sharpe"] < PERFORMANCE_THRESHOLD or perf["max_drawdown"] < DRAWDOWN_LIMIT:
        print("⚠️ Model underperforming — triggering retraining.")
        retrain_model()
    else:
        print("✅ Model performance acceptable. No retraining needed.")

def adaptive_loop():
    print("🚀 Starting adaptive learning loop ...")
    while True:
        evaluate_and_adapt()
        print(f"⏳ Waiting {CHECK_INTERVAL_SEC/3600:.1f} hours until next check ...")
        time.sleep(CHECK_INTERVAL_SEC)

# ================= ENTRY =================
if __name__ == "__main__":
    evaluate_and_adapt()
    # For continuous loop use:
    # adaptive_loop()
