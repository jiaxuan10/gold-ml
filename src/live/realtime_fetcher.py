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

    print(f"📡 Connecting to Yahoo Finance... ({datetime.now().strftime('%H:%M:%S')})")
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
    
    # Merge other assets
    for sym in ASSETS.values():
        if sym == "GC=F": continue
        if sym in df.columns.levels[0]:
            sub = df[sym][["Close"]].reset_index().rename(columns={"Datetime": "Date", "Close": sym})
            merged = pd.merge_asof(merged.sort_values("Date"), sub.sort_values("Date"), on="Date")

    # 1. 先填充 (ffill) 
    # 这一步是为了防止因为某些资产偶尔缺数据导致 NaN
    merged = merged.ffill().bfill()

    # ✅ 关键逻辑一：确保时间格式并剔除周六 (Closed Market)
    # 0=Mon, 4=Fri, 5=Sat(休市), 6=Sun(晚上开市)
    merged["Date"] = pd.to_datetime(merged["Date"], utc=True)
    merged = merged[merged["Date"].dt.dayofweek != 5].reset_index(drop=True)

    # ✅ 关键逻辑二：剔除“僵尸数据” (Flat Line Cleaner)
    # 如果 Close 价格跟上一行完全一样，说明这是 ffill 制造的假数据（例如假期休市）
    # diff() 计算差值，abs() 取绝对值，> 1e-6 意味着“只要有极微小的变动就保留”
    # fillna(1.0) 是为了保护第一行不被删掉
    if "GOLD_Close" in merged.columns:
        merged = merged[merged["GOLD_Close"].diff().fillna(1.0).abs() > 1e-6].reset_index(drop=True)

    # ✅ 关键逻辑三：剔除“未完成”的最新 K 线 (Unfinished Candle Logic)
    # 只有当 K 线时间 < 当前时间 (即已经过去的完整小时) 才保留
    if not merged.empty:
        current_utc_hour = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
        last_row_time = merged["Date"].iloc[-1]
        
        if last_row_time >= current_utc_hour:
            print(f"✂️ Dropping unfinished candle: {last_row_time} (Current UTC: {current_utc_hour})")
            merged = merged.iloc[:-1]

    # 2. 然后再计算指标 (此时数据在时间上是“无缝拼接”且“完整”的)
    merged = add_technical_features(merged)
    
    # Trim to save space (keep last 200 rows is enough for inference)
    return merged.tail(200)

if __name__ == "__main__":
    print(f"🔄 Realtime Fetcher Started. Saving to {OUTPUT_CSV}")
    while True:
        try:
            data = fetch_latest_window()
            if data is not None and not data.empty:
                data.to_csv(OUTPUT_CSV, index=False)
                print(f"✅ Data Updated. Last Candle Used: {data['Date'].iloc[-1]}")
            else:
                print("⚠️ Fetch failed or empty.")
        except Exception as e:
            print(f"⚠️ Error: {e}")
        
        time.sleep(CHECK_INTERVAL)