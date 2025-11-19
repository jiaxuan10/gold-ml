#!/usr/bin/env python3
"""
simulate_trading.py
-------------------------------------------------
Backtesting entry for Profit-Boost v6 with correct feature engineering.
"""

import os
import sys
import pickle
import pandas as pd
import numpy as np

# --- make sure src is in path to import features ---
ROOT_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_SRC)

from features.features_engineering import add_technical_features
from strategy import ProfitBoostStrategy

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODEL_PATH = os.path.join(ROOT, "models", "run_20251120_013217", "ensemble_calibrated_20251120_013217.pkl")
DATA_PATH = os.path.join(ROOT, "data", "final", "final_dataset_hourly.csv")
SAVE_DIR = os.path.join(ROOT, "backtest_results")
os.makedirs(SAVE_DIR, exist_ok=True)


def main():
    # --- Load dataset ---
    if not os.path.exists(DATA_PATH):
        print("Data not found:", DATA_PATH)
        return

    df = pd.read_csv(DATA_PATH)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    print(f"Loaded dataset: {len(df)} rows, {df['Date'].min()} → {df['Date'].max()}")

    # --- Load model ---
    if not os.path.exists(MODEL_PATH):
        print("Model not found:", MODEL_PATH)
        return

    with open(MODEL_PATH, "rb") as f:
        model_meta = pickle.load(f)
    print("Loaded model:", MODEL_PATH)

    # --- Rename columns to match training ---
    rename_map = {
        "GOLD_Close": "Close",
        "GOLD_Open": "Open",
        "GOLD_High": "High",
        "GOLD_Low": "Low",
        "GOLD_Volume": "Volume"
    }
    df = df.rename(columns=rename_map)

    # --- Feature engineering ---
    df = add_technical_features(df)

    # --- Feature alignment ---
    feature_cols = model_meta.get("feature_cols", [])
    if not feature_cols:
        print("⚠️ Warning: no feature_cols found in model_meta.")

    # Add missing columns
    for c in feature_cols:
        if c not in df.columns:
            df[c] = np.nan

    # Ensure order + fillna
    X_backtest = df[feature_cols].fillna(df[feature_cols].median())

    # ===== FEATURE DEBUG =====
    print("\n================ FEATURE DEBUG ================")

    print("\n📌 Training feature columns:")
    for f in feature_cols:
        print(" -", f)

    print("\n📌 df columns after add_technical_features():")
    print(df.columns.tolist())

    print("\n📌 X_backtest columns (final used features):")
    print(X_backtest.columns.tolist())

    missing = [c for c in feature_cols if c not in df.columns]
    print("\n📌 Missing features in df:", missing)

    print("\n================================================\n")

    # --- Run backtest ---
    strat = ProfitBoostStrategy(model_meta, SAVE_DIR, shift_features=True)
    metrics = strat.simulate(df)

    print("\nBacktest metrics (v6.6):")
    for k, v in metrics.items():
        print(f"{k:20s}: {v}")


if __name__ == "__main__":
    main()
