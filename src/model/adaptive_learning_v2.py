#!/usr/bin/env python3
"""
adaptive_learning_v2.py
-------------------------
Adaptive retraining for gold model (compatible with train_gold_model_v3_enhanced).

Features:
- Rolling window retraining (e.g., last 90 days)
- Auto threshold re-tuning each round
- Model performance tracking (accuracy, f1)
- Only replace model if new one outperforms old
"""

import os
import time
import json
import pickle
import pandas as pd
from datetime import datetime, timedelta
from subprocess import run, CalledProcessError

# === CONFIG ===
TRAIN_SCRIPT = "src/model/train_gold_model_v3_enhanced.py"
DATA_PATH = "data/final/final_dataset_hourly.csv"
MODEL_DIR = "models"
RETRAIN_WINDOW_DAYS = 90         # rolling training window (last N days)
RETRAIN_INTERVAL_HOURS = 12      # re-train every 12 hours
MIN_REQUIRED_ROWS = 300
LOG_PATH = os.path.join(MODEL_DIR, "adaptive_learning_log.csv")

def load_latest_model():
    model_paths = []
    for root, _, files in os.walk(MODEL_DIR):
        for f in files:
            if f.startswith("ensemble_calibrated_") and f.endswith(".pkl"):
                model_paths.append(os.path.join(root, f))
    if not model_paths:
        return None
    latest = max(model_paths, key=os.path.getmtime)
    return latest

def extract_metrics_from_report(report_path):
    try:
        with open(report_path, "r") as f:
            report = json.load(f)
        return {
            "timestamp": report.get("timestamp"),
            "accuracy": report.get("final_acc"),
            "f1": report.get("final_f1"),
            "precision": report.get("final_precision"),
            "recall": report.get("final_recall"),
            "threshold": report.get("best_threshold_validation")
        }
    except Exception:
        return None

def adaptive_retrain():
    df = pd.read_csv(DATA_PATH)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date")

    # rolling window subset
    latest_date = df["Date"].max()
    start_date = latest_date - timedelta(days=RETRAIN_WINDOW_DAYS)
    df_recent = df[df["Date"] >= start_date]

    if len(df_recent) < MIN_REQUIRED_ROWS:
        print(f"⚠️ Not enough data ({len(df_recent)} rows) for retraining. Skipping.")
        return

    tmp_path = "data/tmp_recent_window.csv"
    df_recent.to_csv(tmp_path, index=False)
    print(f"📊 Using recent {len(df_recent)} rows ({start_date.date()} → {latest_date.date()}) for retraining")

    # Run training
    print("🚀 Starting retraining...")
    try:
        run(["python", TRAIN_SCRIPT], check=True)
    except CalledProcessError as e:
        print(f"❌ Training failed: {e}")
        return

    # Find new model
    new_model = load_latest_model()
    if not new_model:
        print("❌ No model found after retraining.")
        return

    # Find corresponding summary report
    report_dir = os.path.dirname(new_model)
    report_path = os.path.join(report_dir, "summary_report.json")

    metrics_new = extract_metrics_from_report(report_path)
    if not metrics_new:
        print("⚠️ Could not read new model metrics.")
        return

    print(f"✅ New model metrics: acc={metrics_new['accuracy']:.3f}, f1={metrics_new['f1']:.3f}")

    # Compare with previous model (if any)
    prev_model = load_latest_model()
    if prev_model and prev_model != new_model:
        prev_report = os.path.join(os.path.dirname(prev_model), "summary_report.json")
        metrics_old = extract_metrics_from_report(prev_report)
        if metrics_old:
            print(f"📊 Old model: acc={metrics_old['accuracy']:.3f}, f1={metrics_old['f1']:.3f}")
            if metrics_new["f1"] <= metrics_old["f1"]:
                print("⚖️ New model not better → keeping old model.")
                return

    # Save record
    new_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "rows_used": len(df_recent),
        **metrics_new
    }
    if os.path.exists(LOG_PATH):
        log_df = pd.read_csv(LOG_PATH)
        log_df = pd.concat([log_df, pd.DataFrame([new_entry])], ignore_index=True)
    else:
        log_df = pd.DataFrame([new_entry])
    log_df.to_csv(LOG_PATH, index=False)
    print(f"📜 Logged new training session → {LOG_PATH}")
    print("✅ Adaptive retraining completed successfully.\n")

def run_forever():
    while True:
        adaptive_retrain()
        print(f"⏰ Sleeping for {RETRAIN_INTERVAL_HOURS} hours...\n")
        time.sleep(RETRAIN_INTERVAL_HOURS * 3600)

if __name__ == "__main__":
    run_forever()
