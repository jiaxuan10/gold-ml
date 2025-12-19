#!/usr/bin/env python3
# src/live/realtime_fetcher.py - 
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

try:
    from features.features_engineering import add_technical_features
except ImportError:
    print(" Feature engineering import failed, running raw.")
    def add_technical_features(df): return df

OUT_DIR = os.path.join(ROOT, "data", "final")
os.makedirs(OUT_DIR, exist_ok=True)
OUTPUT_CSV = os.path.join(OUT_DIR, "latest_hour_features.csv")

ASSETS = {
    "GOLD": "GC=F", "DXY": "DX-Y.NYB", "SP500": "^GSPC",
    "NASDAQ": "^IXIC", "VIX": "^VIX", "BTCUSD": "BTC-USD", "ETHUSD": "ETH-USD"
}

HIST_WINDOW = 720
CHECK_INTERVAL = 60 

def fetch_latest_window():
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(hours=HIST_WINDOW)

    print(f" Connecting to Yahoo Finance... ({datetime.now().strftime('%H:%M:%S')})")
    
    try:
        df = yf.download(list(ASSETS.values()), start=start_date, end=end_date, interval="1h", progress=False, threads=True)
    except Exception as e:
        print(f"Download Error: {e}")
        return None

    if df is None or df.empty: return None

    if isinstance(df.columns, pd.MultiIndex):
        df = df.swaplevel(0, 1, axis=1)
        df.sort_index(axis=1, level=0, inplace=True)

    if "GC=F" not in df.columns.levels[0]: return None

    gold_df = df["GC=F"].copy().reset_index()
    merged = gold_df.rename(columns={"Datetime": "Date", "Date": "Date", "Close": "GOLD_Close", "Open": "GOLD_Open", "High": "GOLD_High", "Low": "GOLD_Low", "Volume": "GOLD_Volume"})
    
    for sym in ASSETS.values():
        if sym == "GC=F": continue
        if sym in df.columns.levels[0]:
            sub = df[sym][["Close"]].reset_index().rename(columns={"Datetime": "Date", "Close": sym})
            merged = pd.merge_asof(merged.sort_values("Date"), sub.sort_values("Date"), on="Date")

    merged = merged.ffill().bfill()
    merged["Date"] = pd.to_datetime(merged["Date"], utc=True)
    merged["Date"] = merged["Date"].dt.floor("H")

    if not merged.empty:
            current_utc_hour = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
            
            if merged["Date"].iloc[-1] >= current_utc_hour:
                print(f"   Dropping unfinished candle: {merged['Date'].iloc[-1]} (Wait for close)")
                merged = merged.iloc[:-1]

    try:
        merged = add_technical_features(merged)
    except: pass
    
    return merged.tail(200)

if __name__ == "__main__":
    print(f" Realtime Fetcher Started. Interval: {CHECK_INTERVAL}s")
    while True:
        try:
            data = fetch_latest_window()
            if data is not None and not data.empty:
                data.to_csv(OUTPUT_CSV, index=False)
                last_time = data['Date'].iloc[-1]
                
                if 'GOLD_Close' in data.columns:
                    last_price = data['GOLD_Close'].iloc[-1]
                elif 'Close' in data.columns:
                    last_price = data['Close'].iloc[-1]
                else:
                    last_price = 0.0
                    
                print(f" Data Updated. Last Candle: {last_time} | Price: {last_price:.2f}")
            else:
                print(" No data.")
        except Exception as e:
            print(f" Loop Error: {e}")
        
        time.sleep(CHECK_INTERVAL)