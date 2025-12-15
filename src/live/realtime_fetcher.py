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
try:
    from features.features_engineering import add_technical_features
except ImportError:
    print("❌ Error: Cannot import features_engineering. Check path.")
    sys.exit(1)

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

    print(f"Connecting to Yahoo Finance... ({datetime.now().strftime('%H:%M:%S')})")
    
    try:
        # 1. Download Data
        # Using group_by='column' to handle MultiIndex better
        df = yf.download(
            tickers=list(ASSETS.values()),
            start=start_date,
            end=end_date,
            interval="1h",
            auto_adjust=True,
            progress=False,
            threads=True,
            group_by='column' 
        )
    except Exception as e:
        print(f"❌ Yahoo Download Failed: {e}")
        return None

    if df is None or df.empty:
        print("❌ No data returned.")
        return None

    # 2. Handle MultiIndex (Price, Ticker) vs (Ticker, Price)
    if isinstance(df.columns, pd.MultiIndex):
        first_level = df.columns.get_level_values(0)[0]
        if first_level in ['Close', 'Open', 'High', 'Low', 'Volume', 'Adj Close']:
            df = df.swaplevel(0, 1, axis=1)
            df.sort_index(axis=1, level=0, inplace=True)

    # 3. Extract Gold Data
    # Check if GC=F exists
    if "GC=F" not in df.columns.levels[0] if isinstance(df.columns, pd.MultiIndex) else "GC=F" not in df.columns:
        print("❌ Critical: Gold (GC=F) not found in data.")
        return None

    try:
        gold_df = df["GC=F"].copy()
        
        # Handle 'Adj Close' fallback
        if "Close" not in gold_df.columns and "Adj Close" in gold_df.columns:
            gold_df.rename(columns={"Adj Close": "Close"}, inplace=True)
            
        if "Close" not in gold_df.columns:
            print("❌ Critical: No 'Close' price found for Gold.")
            return None
            
        gold_df = gold_df.reset_index()
    except Exception as e:
        print(f"❌ Error extracting gold data: {e}")
        return None
    
    # Standardize Names
    merged = gold_df.rename(columns={
        "Datetime": "Date", "Date": "Date",
        "Close": "GOLD_Close", "Open": "GOLD_Open", 
        "High": "GOLD_High", "Low": "GOLD_Low", "Volume": "GOLD_Volume"
    })
    
    # 4. Merge Other Assets
    for sym in ASSETS.values():
        if sym == "GC=F": continue
        
        # Check existence
        exists = sym in df.columns.levels[0] if isinstance(df.columns, pd.MultiIndex) else sym in df.columns
        if exists:
            try:
                sub = df[sym].copy()
                col_name = "Close" if "Close" in sub.columns else "Adj Close" if "Adj Close" in sub.columns else None
                
                if col_name:
                    sub = sub[[col_name]].reset_index().rename(columns={"Datetime": "Date", col_name: sym})
                    merged = pd.merge_asof(merged.sort_values("Date"), sub.sort_values("Date"), on="Date")
            except: pass # Skip failed merges silently

    merged = merged.ffill().bfill()
    merged["Date"] = pd.to_datetime(merged["Date"], utc=True)

    # 5. Filtering Logic
    # Keep only XX:00 candles
    merged = merged[merged["Date"].dt.minute == 0].reset_index(drop=True)
    # Remove Saturdays (UTC)
    merged = merged[merged["Date"].dt.dayofweek != 5].reset_index(drop=True)
    # Remove flat candles (no volume/movement)
    if "GOLD_Close" in merged.columns:
        merged = merged[merged["GOLD_Close"].diff().fillna(1.0).abs() > 1e-6].reset_index(drop=True)

    # Remove unfinished candle (if current hour hasn't closed)
    if not merged.empty:
        current_utc_hour = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
        if merged["Date"].iloc[-1] >= current_utc_hour:
            print(f"   Dropping unfinished candle: {merged['Date'].iloc[-1]}")
            merged = merged.iloc[:-1]

    # 6. Feature Engineering
    try:
        merged = add_technical_features(merged)
    except Exception as e:
        print(f"❌ Feature Engineering Failed: {e}")
        return None
    
    return merged.tail(200)

if __name__ == "__main__":
    print(f"🚀 Realtime Fetcher Started.")
    print(f"   Saving to: {OUTPUT_CSV}")
    print(f"   Interval : {CHECK_INTERVAL}s")
    
    while True:
        try:
            data = fetch_latest_window()
            if data is not None and not data.empty:
                data.to_csv(OUTPUT_CSV, index=False)
                print(f"✅ Data Updated. Last Candle: {data['Date'].iloc[-1]} (UTC)")
            else:
                print("⚠️ Update Skipped (No valid data).")
        except Exception as e:
            print(f"❌ Loop Error: {e}")
        
        time.sleep(CHECK_INTERVAL)