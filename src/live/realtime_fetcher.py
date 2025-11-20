#!/usr/bin/env python3
# src/live/realtime_fetcher.py

import os
import time
import sys
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta, timezone
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

# Path Setup
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(os.path.join(ROOT, "src"))

# Import Unified Feature Engineering
from features.features_engineering import add_technical_features

OUT_DIR = os.path.join(ROOT, "data", "final")
os.makedirs(OUT_DIR, exist_ok=True)
OUTPUT_CSV = os.path.join(OUT_DIR, "latest_hour_features.csv")

ASSETS = {
    "GOLD": "GC=F",
    "DXY": "DX-Y.NYB",
    "SP500": "^GSPC",
    "NASDAQ": "^IXIC",
    "VIX": "^VIX",
    "BTCUSD": "BTC-USD",
    "ETHUSD": "ETH-USD",
}

HIST_WINDOW = 720  # Need enough history for SMA_60 and Volatility
CHECK_INTERVAL = 300 # Fetch every 5 minutes

def fetch_latest_window():
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(hours=HIST_WINDOW)

    print(f"📡 Connecting to Yahoo Finance... ({datetime.now().time()})")
    df = yf.download(
        tickers=list(ASSETS.values()),
        start=start_date,
        end=end_date,
        interval="1h",
        auto_adjust=True,
        progress=False,
        threads=True,
    )

    if df is None or df.empty:
        print("❌ No data returned.")
        return None

    # Fix MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        df = df.swaplevel(0, 1, axis=1)
        df.sort_index(axis=1, level=0, inplace=True)

    # Extract Gold
    if "GC=F" not in df.columns.levels[0]:
        return None

    gold_df = df["GC=F"].copy().reset_index()
    
    # Rename standard columns
    merged = gold_df.rename(columns={
        "Datetime": "Date", "Date": "Date",
        "Close": "GOLD_Close", "Open": "GOLD_Open", 
        "High": "GOLD_High", "Low": "GOLD_Low", "Volume": "GOLD_Volume"
    })
    
    # Merge other assets (DXY, BTC, etc.)
    for sym in ASSETS.values():
        if sym == "GC=F": continue
        if sym in df.columns.levels[0]:
            sub = df[sym][["Close"]].reset_index().rename(columns={"Datetime": "Date", "Close": sym})
            # Merge logic
            merged = pd.merge_asof(merged.sort_values("Date"), sub.sort_values("Date"), on="Date")

    # Apply Unified Features
    merged = add_technical_features(merged)
    
    # Trim to save space (keep last 100 rows is enough for inference, but keep more for charts)
    return merged.tail(200)

if __name__ == "__main__":
    print(f"🔄 Realtime Fetcher Started. Saving to {OUTPUT_CSV}")
    while True:
        try:
            data = fetch_latest_window()
            if data is not None and not data.empty:
                data.to_csv(OUTPUT_CSV, index=False)
                print(f"✅ Data Updated. Last Candle: {data['Date'].iloc[-1]}")
            else:
                print("⚠️ Fetch failed or empty.")
        except Exception as e:
            print(f"⚠️ Error: {e}")
        
        time.sleep(CHECK_INTERVAL)