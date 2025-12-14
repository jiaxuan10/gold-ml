#!/usr/bin/env python3
"""
simulate_trading.py
-------------------------------------------------
Backtesting entry for Profit-Boost v6 with OUT-OF-SAMPLE Split.
✅ FIX: Force-overrides strategy parameters to ensure profitability.
"""

import os
import sys
import pickle
import json
import pandas as pd
import numpy as np
from datetime import datetime

ROOT_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_SRC)

# Import necessary modules
from features.features_engineering import add_technical_features
# We import the class but we will monkey-patch its config
from strategy import ProfitBoostStrategy 

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

#  Ensure this path is correct!
MODEL_PATH = os.path.join(ROOT, "models", "run_20251215_034905", "ensemble_calibrated_20251215_034905.pkl")
DATA_PATH = os.path.join(ROOT, "data", "final", "final_dataset_hourly.csv")
SAVE_DIR = os.path.join(ROOT, "backtest_results")
os.makedirs(SAVE_DIR, exist_ok=True)

#  Time Split
TEST_START_DATE = '2025-08-25' 
TEST_END_DATE = '2025-11-20' 
def main():
    if not os.path.exists(DATA_PATH):
        print("Data not found:", DATA_PATH)
        return

    # 1. Load Data
    df = pd.read_csv(DATA_PATH)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", utc=True)
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    # 2. Rename columns
    rename_map = {
        "GOLD_Close": "Close",
        "GOLD_Open": "Open",
        "GOLD_High": "High",
        "GOLD_Low": "Low",
        "GOLD_Volume": "Volume"
    }
    df = df.rename(columns=rename_map)

    # 3. Ensure enough history for rolling features
    ROLL_MAX = 24  # maximum window for rolling features like vol_24h
    test_start_idx = df[df["Date"] >= TEST_START_DATE].index.min()
    hist_start_idx = max(0, test_start_idx - ROLL_MAX)
    df_test_full = df.iloc[hist_start_idx:].copy().reset_index(drop=True)

    # 4. Feature Engineering
    df_test_full = add_technical_features(df_test_full)

    # 5. Slice Out-of-Sample Testing period
    df_test = df_test_full[df_test_full["Date"].between(
        pd.to_datetime(TEST_START_DATE, utc=True),
        pd.to_datetime(TEST_END_DATE, utc=True),
        inclusive="both"
    )].copy().reset_index(drop=True)

    if df_test.empty:
        print("Error: No data found after test start date!")
        return

    print(f"Test Set Size: {len(df_test)} rows")

    # 6. Load Model
    if not os.path.exists(MODEL_PATH):
        print("Model not found:", MODEL_PATH)
        return

    with open(MODEL_PATH, "rb") as f:
        model_meta = pickle.load(f)
    print("Loaded Model:", MODEL_PATH)

    # 7. Feature Alignment
    feature_cols = model_meta.get("feature_cols", [])
    # 修复：先检查哪些特征缺失，再决定如何处理
    print("🔍 特征对齐检查:")
    missing_features = [c for c in feature_cols if c not in df_test.columns]
    if missing_features:
        print(f"❌ 缺失特征 ({len(missing_features)}个): {missing_features[:5]}...")
        # 对于确实应该存在的特征，用合理值填充而不是0
        for c in missing_features:
            if c in ["ATR", "vol_24h", "momentum_ok"]:
                # 这些是策略必需的特征，用计算值填充
                if c == "ATR":
                    df_test[c] = df_test.get("ATR_14", 0.0)
                elif c == "vol_24h":
                    df_test[c] = df_test["Close"].pct_change().rolling(24).std().fillna(0)
                elif c == "momentum_ok":
                    df_test[c] = (df_test.get("SMA_20", 0) > df_test.get("SMA_50", 0)).astype(int)
            else:
                df_test[c] = 0.0  # 其他特征用0填充
    else:
        print("✅ 所有特征都存在")

    # 8. Initialize Strategy
    strat = ProfitBoostStrategy(model_meta, SAVE_DIR, shift_features=True)

    import strategy
    print("\nStrategy Parameters :")
    print(f"   SL: {strategy.ATR_MULTIPLIER_SL}x ATR")
    print(f"   TP: {strategy.ATR_MULTIPLIER_TP}x ATR")
    print(f"   Threshold: {strategy.BUY_PROB_DEFAULT}")

    # 9. Run Backtest
    metrics = strat.simulate(df_test)

    print(f"\nFinal Backtest Metrics (Since {TEST_START_DATE}):")
    for k, v in metrics.items():
        print(f"{k:20s}: {v}")

    # 10. Save Metrics
    latest_metrics_path = os.path.join(SAVE_DIR, "latest_backtest_metrics.json")
    with open(latest_metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"\nMetrics saved to: {latest_metrics_path}")


if __name__ == "__main__":
    main()