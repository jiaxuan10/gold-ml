#!/usr/bin/env python3

import os
import sys
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta, timezone

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))
try:
    from features.features_engineering import add_technical_features
except ImportError:
    print(" Critical: Cannot import 'add_technical_features'. Make sure you are in the project root.")
    sys.exit(1)

DATA_DIR = "data/final"
os.makedirs(DATA_DIR, exist_ok=True)
FINAL_CSV = os.path.join(DATA_DIR, "final_dataset_hourly.csv")

END_DATE = datetime.now(timezone.utc)
START_DATE = END_DATE - timedelta(days=720)

ASSETS = {
    "GOLD": "GC=F",
    "DXY": "DX-Y.NYB",
    "SP500": "^GSPC",
    "NASDAQ": "^IXIC",
    "VIX": "^VIX",
    "BTCUSD": "BTC-USD",
    "ETHUSD": "ETH-USD",
}

def fetch_data():
    print(f" Downloading hourly data from {START_DATE.date()} to {END_DATE.date()}...")
    
    df = yf.download(
        tickers=list(ASSETS.values()),
        start=START_DATE,
        end=END_DATE,
        interval="1h",
        auto_adjust=True,
        progress=False,
        threads=True
    )

    if df.empty:
        print("❌ No data returned.")
        return None

    if isinstance(df.columns, pd.MultiIndex):
        df = df.swaplevel(0, 1, axis=1)
        df.sort_index(axis=1, level=0, inplace=True)

    if "GC=F" not in df.columns.levels[0]:
        print("❌ Gold data missing.")
        return None

    gold_df = df["GC=F"].copy().reset_index()
    merged = gold_df.rename(columns={
        "Datetime": "Date", "Date": "Date",
        "Close": "GOLD_Close", "Open": "GOLD_Open", 
        "High": "GOLD_High", "Low": "GOLD_Low", "Volume": "GOLD_Volume"
    })

    for sym in ASSETS.values():
        if sym == "GC=F": continue
        if sym in df.columns.levels[0]:
            sub = df[sym][["Close"]].reset_index().rename(columns={"Datetime": "Date", "Close": sym})
            merged = pd.merge_asof(merged.sort_values("Date"), sub.sort_values("Date"), on="Date")

    merged = merged.ffill().bfill()
    merged["Date"] = pd.to_datetime(merged["Date"], utc=True)
    
    merged = merged[merged["Date"].dt.dayofweek != 5].reset_index(drop=True)
    
    if "GOLD_Close" in merged.columns:
        merged = merged[merged["GOLD_Close"].diff().fillna(1.0).abs() > 1e-6].reset_index(drop=True)

    return merged

def main():
    df = fetch_data()
    if df is None: return

    print(" Applying unified feature engineering...")
    

    df = add_technical_features(df)
    
    df.to_csv(FINAL_CSV, index=False)
    print(f"\n Training data saved to: {FINAL_CSV}")
    print(f" Shape: {df.shape}")
    print(f" Features: {len(df.columns)}")
    print(f"   (Includes: Gold_DX-Y.NYB_Ratio, Lag_Return_1h, etc.)")

if __name__ == "__main__":
    main()