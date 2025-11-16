#!/usr/bin/env python3
"""
realtime_hourly_update.py
-----------------------------------------------------
⚡ Automatically fetch latest hourly data + generate all features
⚡ Ensures features match your original 350h window full-feature pipeline
⚡ Saves latest hour features to CSV whenever new data appears
-----------------------------------------------------
"""

import os
import time
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta, timezone

# ---------- PATH ----------
os.makedirs("data/final", exist_ok=True)
OUTPUT_CSV = "data/final/latest_hour_features.csv"

# ---------- ASSETS ----------
ASSETS = {
    "GOLD": "GC=F",
    "DXY": "DX-Y.NYB",
    "SP500": "^GSPC",
    "NASDAQ": "^IXIC",
    "VIX": "^VIX",
    "BTCUSD": "BTC-USD",
    "ETHUSD": "ETH-USD",
}

# ---------- HISTORICAL WINDOW ----------
HIST_WINDOW = 350  # 保证 rolling 指标完整

CHECK_INTERVAL = 300  # 秒，每5分钟检查一次

# ---------- FEATURE FUNCTION ----------
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Return"] = df["GOLD_Close"].pct_change()
    df["LogReturn"] = np.log(df["GOLD_Close"] / df["GOLD_Close"].shift(1))

    df["SMA_20"] = df["GOLD_Close"].rolling(20, min_periods=5).mean()
    df["SMA_50"] = df["GOLD_Close"].rolling(50, min_periods=10).mean()
    df["EMA_12"] = df["GOLD_Close"].ewm(span=12, adjust=False).mean()
    df["EMA_26"] = df["GOLD_Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = df["EMA_12"] - df["EMA_26"]
    df["Signal_Line"] = df["MACD"].ewm(span=9, adjust=False).mean()

    delta = df["GOLD_Close"].diff()
    gain = delta.clip(lower=0).rolling(14, min_periods=3).mean()
    loss = (-delta.clip(upper=0)).rolling(14, min_periods=3).mean()
    rs = gain / (loss + 1e-9)
    df["RSI_14"] = 100 - (100 / (1 + rs))

    hl = df["GOLD_High"] - df["GOLD_Low"]
    hc = (df["GOLD_High"] - df["GOLD_Close"].shift()).abs()
    lc = (df["GOLD_Low"] - df["GOLD_Close"].shift()).abs()
    tr = np.maximum.reduce([hl, hc, lc])
    df["ATR_14"] = pd.Series(tr).rolling(14, min_periods=3).mean()

    df["Volatility_20"] = df["Return"].rolling(20, min_periods=5).std()
    df["Momentum_10"] = df["GOLD_Close"].pct_change(10)

    for asset in ["DX-Y.NYB", "^GSPC", "BTC-USD"]:
        if asset in df.columns:
            df[f"Gold_{asset}_Ratio"] = df["GOLD_Close"] / (df[asset] + 1e-6)
            df[f"Corr_{asset}_24h"] = df["GOLD_Close"].rolling(24).corr(df[asset])
            df[f"Corr_{asset}_24h"] = df[f"Corr_{asset}_24h"].replace([np.inf, -np.inf], np.nan).fillna(0)

    df["Lag_Return_1h"] = df["Return"].shift(1).fillna(0)
    df["Lag_Return_6h"] = df["Return"].shift(6).fillna(0)
    df["MA_diff"] = df["SMA_20"] - df["SMA_50"]

    rolling_mean = df["GOLD_Close"].rolling(24).mean()
    rolling_std = df["GOLD_Close"].rolling(24).std()
    df["Rolling_Zscore_24h"] = (df["GOLD_Close"] - rolling_mean) / (rolling_std + 1e-9)

    df = df.dropna(how="all").reset_index(drop=True)
    return df

# ---------- FETCH + MERGE FUNCTION ----------
def fetch_latest_window():
    END_DATE = datetime.now(timezone.utc)
    START_DATE = END_DATE - timedelta(hours=HIST_WINDOW)

    df = yf.download(
        tickers=list(ASSETS.values()),
        start=START_DATE,
        end=END_DATE,
        interval="1h",
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    if df.empty:
        return None

    if isinstance(df.columns, pd.MultiIndex):
        df = df.swaplevel(0, 1, axis=1)
        df.sort_index(axis=1, level=0, inplace=True)
    else:
        df.columns = pd.MultiIndex.from_product([list(ASSETS.values()), df.columns])

    frames = {}
    for sym in df.columns.levels[0]:
        sub = df[sym].copy().reset_index()
        sub.rename(columns={"Datetime": "Date"}, inplace=True)
        frames[sym] = sub

    gold_df = frames["GC=F"].copy()
    merged = gold_df.rename(
        columns={
            "Close": "GOLD_Close",
            "Open": "GOLD_Open",
            "High": "GOLD_High",
            "Low": "GOLD_Low",
            "Volume": "GOLD_Volume",
        }
    )[["Date","GOLD_Close","GOLD_Open","GOLD_High","GOLD_Low","GOLD_Volume"]]

    for name, df2 in frames.items():
        if name == "GC=F":
            continue
        if "Close" not in df2.columns:
            continue
        merged = pd.merge_asof(
            merged.sort_values("Date"),
            df2[["Date","Close"]].sort_values("Date").rename(columns={"Close": name}),
            on="Date",
            direction="backward"
        )

    # Fill missing hours
    full_idx = pd.date_range(start=merged["Date"].min(), end=merged["Date"].max(), freq="H", tz=timezone.utc)
    merged = merged.set_index("Date").reindex(full_idx).reset_index().rename(columns={"index":"Date"})
    merged = merged.interpolate(limit_direction="both").ffill().bfill()

    return add_features(merged)

# ---------- REALTIME LOOP ----------
last_timestamp = None

while True:
    try:
        data = fetch_latest_window()
        if data is None or data.empty:
            print(f"{datetime.utcnow()} - No data fetched.")
            time.sleep(CHECK_INTERVAL)
            continue

        current_ts = data["Date"].iloc[-1]
        if last_timestamp is None or current_ts > last_timestamp:
            data.tail(1).to_csv(OUTPUT_CSV, index=False)
            print(f"{datetime.utcnow()} - New hour saved: {current_ts}")
            last_timestamp = current_ts
        else:
            print(f"{datetime.utcnow()} - No new hour yet.")

    except Exception as e:
        print(f"❌ Error: {e}")

    time.sleep(CHECK_INTERVAL)
