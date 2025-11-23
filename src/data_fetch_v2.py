#!/usr/bin/env python3
"""
fetch_data_highfreq_final_v3.py
-----------------------------------------------------
✅ Gold Scalper–style hourly data pipeline (stable)
✅ Auto fallback if 1h data unavailable (→ 1d)
✅ Smart NaN handling (no interpolation fabrication)
✅ Adds key technical & cross-asset features
✅ Auto-clean: drop leading/trailing static rows
-----------------------------------------------------
"""

import os
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta, timezone

# ---------- PATH CONFIG ----------
os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/final", exist_ok=True)
FINAL_CSV = "data/final/final_dataset_hourly.csv"
FINAL_PARQUET = "data/final/final_dataset_hourly.parquet"

# ---------- DATE RANGE ----------
END_DATE = datetime.now(timezone.utc)
START_DATE = END_DATE - timedelta(days=720)
print(f"📥 Fetching hourly data from {START_DATE.date()} → {END_DATE.date()}")

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

# ---------- FETCH FUNCTION ----------
def fetch_with_fallback(tickers, interval="1h"):
    """Attempt to fetch hourly data; fallback to daily if Yahoo blocks it."""
    print(f"📡 Fetching {len(tickers)} tickers ({interval}) via Yahoo Finance...")
    df = yf.download(
        tickers=tickers,
        start=START_DATE,
        end=END_DATE,
        interval=interval,
        auto_adjust=True,
        progress=False,
        threads=True,
    )

    if df.empty:
        raise RuntimeError(f"❌ No {interval} data returned from Yahoo Finance.")

    if isinstance(df.columns, pd.MultiIndex):
        df = df.swaplevel(0, 1, axis=1)
        df.sort_index(axis=1, level=0, inplace=True)
        print(f"✅ MultiIndex fixed with symbols: {df.columns.levels[0].tolist()}")
    else:
        raise RuntimeError("❌ Unexpected format — expected MultiIndex from Yahoo Finance.")

    if "GC=F" in df.columns.levels[0]:
        gold_close = df["GC=F"]["Close"].dropna()
        if len(gold_close.unique()) <= 3:
            print(f"⚠️ GC=F 1h data too flat, retrying with daily interval...")
            return fetch_with_fallback(tickers, interval="1d")

    return df


# ---------- MAIN FETCH ----------
raw_df = fetch_with_fallback(list(ASSETS.values()), interval="1h")

frames = {}
for sym in raw_df.columns.levels[0]:
    sub = raw_df[sym].copy().reset_index()
    sub.rename(columns={"Datetime": "Date"}, inplace=True)
    sub["Symbol"] = sym
    frames[sym] = sub

if "GC=F" not in frames:
    raise RuntimeError("❌ Missing gold data (GC=F).")

gold_df = frames["GC=F"].copy()
merged = gold_df.rename(
    columns={
        "Close": "GOLD_Close",
        "Open": "GOLD_Open",
        "High": "GOLD_High",
        "Low": "GOLD_Low",
        "Volume": "GOLD_Volume",
    }
)[["Date", "GOLD_Close", "GOLD_Open", "GOLD_High", "GOLD_Low", "GOLD_Volume"]]

for name, df in frames.items():
    if name == "GC=F":
        continue
    if "Close" not in df.columns:
        print(f"⚠️ Skipping {name} (no Close column)")
        continue
    merged = pd.merge_asof(
        merged.sort_values("Date"),
        df[["Date", "Close"]].sort_values("Date").rename(columns={"Close": name}),
        on="Date",
    )

merged = merged.drop_duplicates(subset="Date").reset_index(drop=True)
print(f"🔗 Merged shape: {merged.shape}")

# ---------- FEATURE ENGINEERING ----------
def add_ta(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # 🔥 CRITICAL FIX: REMOVED INTERPOLATION
    # df = df.interpolate(limit_direction="both")  <-- This was creating fake weekend data
    
    # Use forward fill to propagate the last known Friday price through the weekend
    df = df.ffill().bfill()

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
    
    # --- Advanced Gold Scalper+ Features ---
    df["Lag_Return_1h"] = df["Return"].shift(1)
    df["Lag_Return_6h"] = df["Return"].shift(6)

    df["MA_diff"] = df["SMA_20"] - df["SMA_50"]

    rolling_mean = df["GOLD_Close"].rolling(24).mean()
    rolling_std = df["GOLD_Close"].rolling(24).std()
    df["Rolling_Zscore_24h"] = (df["GOLD_Close"] - rolling_mean) / (rolling_std + 1e-9)

    df = df.dropna(how="all").reset_index(drop=True)
    print(f"✅ Cleaned dataset: {len(df)} rows remain after feature generation.")
    return df


print("⚙️ Generating technical indicators...")
merged = add_ta(merged)

# ---------- AUTO CLEAN HEAD/TAIL ----------
# Remove long flat segments (where price doesn’t change)
# This effectively deletes the static weekend rows created by ffill()
merged = merged.loc[merged["GOLD_Close"].diff().abs() > 1e-6].reset_index(drop=True)

# ---------- SAVE ----------
merged.to_csv(FINAL_CSV, index=False)
merged.to_parquet(FINAL_PARQUET, index=False)

print("\n✅ Final dataset generated successfully:")
print(f"📄 CSV: {FINAL_CSV}")
print(f"📦 Parquet: {FINAL_PARQUET}")
print(f"📊 Shape: {merged.shape}")
print(f"📅 Date Range: {merged['Date'].min()} → {merged['Date'].max()}")
print(f"🧠 Columns: {list(merged.columns)}")