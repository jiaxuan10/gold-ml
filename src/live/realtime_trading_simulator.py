#!/usr/bin/env python3
"""
real_time_trader_v1.py
---------------------------------------------------------
💹 Real-time Gold Price Prediction & Simulated Trading
- Fetches hourly gold data (last 500 hours)
- Generates same features as training
- Loads trained ensemble model
- Predicts Buy/Sell based on probability threshold
- Simulates portfolio equity (auto trade)
---------------------------------------------------------
Author: Lim Jia Xuan
Date: 2025-10-29
"""

import os
import time
import pickle
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta, timezone
import schedule

# =========================================================
# --- CONFIG ---
# =========================================================
MODEL_PATH = "models/run_20251029_123456/ensemble_calibrated_20251029_123456.pkl"  # ⚠️ 修改成你自己的路径
INITIAL_CAPITAL = 10000
HISTORY_HOURS = 500
TRADE_LOG_PATH = "backtest_results/real_time_trades.csv"

os.makedirs("backtest_results", exist_ok=True)

# =========================================================
# --- FEATURE CREATION (copy from training version) ---
# =========================================================
def add_technical_features(df):
    df = df.copy()
    close = df["GOLD_Close"]

    df["SMA_20"] = close.rolling(20).mean()
    df["SMA_50"] = close.rolling(50).mean()
    df["EMA_12"] = close.ewm(span=12, adjust=False).mean()
    df["EMA_26"] = close.ewm(span=26, adjust=False).mean()
    df["MACD"] = df["EMA_12"] - df["EMA_26"]
    df["Signal_Line"] = df["MACD"].ewm(span=9, adjust=False).mean()

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df["RSI_14"] = 100 - (100 / (1 + rs))

    df["Volatility_20"] = close.pct_change().rolling(20).std()
    df["Momentum_10"] = close.pct_change(10)
    df["Lag_Return_1h"] = close.pct_change(1)
    df["Lag_Return_6h"] = close.pct_change(6)

    df["MA_diff"] = df["SMA_20"] - df["SMA_50"]
    rolling_mean = close.rolling(24).mean()
    rolling_std = close.rolling(24).std()
    df["Rolling_Zscore_24h"] = (close - rolling_mean) / (rolling_std + 1e-9)

    df = df.dropna().reset_index(drop=True)
    return df


# =========================================================
# --- FETCH LATEST DATA ---
# =========================================================
def fetch_latest_data():
    end = datetime.now(timezone.utc)
    start = end - timedelta(hours=HISTORY_HOURS)
    tickers = ["GC=F", "DX-Y.NYB", "^GSPC", "BTC-USD"]

    df = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        interval="1h",
        auto_adjust=True,
        progress=False,
        threads=True,
    )

    if isinstance(df.columns, pd.MultiIndex):
        df = df.swaplevel(0, 1, axis=1)
        df.sort_index(axis=1, level=0, inplace=True)
        df = df.reset_index().rename(columns={"Datetime": "Date"})

    gold_df = df["GC=F"].copy()
    merged = gold_df.rename(
        columns={
            "Close": "GOLD_Close",
            "Open": "GOLD_Open",
            "High": "GOLD_High",
            "Low": "GOLD_Low",
            "Volume": "GOLD_Volume",
        }
    )

    # Merge others
    for sym in ["DX-Y.NYB", "^GSPC", "BTC-USD"]:
        if sym in df.columns.levels[0]:
            merged = pd.merge_asof(
                merged.sort_values("Date"),
                df[sym]["Close"].reset_index().rename(columns={"Close": sym}),
                on="Date"
            )
    return merged


# =========================================================
# --- LOAD MODEL ---
# =========================================================
def load_model():
    with open(MODEL_PATH, "rb") as f:
        data = pickle.load(f)
    model = data.get("calibrated_model", data.get("raw_ensemble"))
    threshold = data.get("threshold", 0.5)
    feature_cols = data.get("feature_cols", [])
    print(f"✅ Model loaded ({len(feature_cols)} features, thr={threshold:.3f})")
    return model, threshold, feature_cols


# =========================================================
# --- TRADING SIMULATION ENGINE ---
# =========================================================
class SimulatedTrader:
    def __init__(self, capital=10000):
        self.cash = capital
        self.position = 0
        self.last_price = None
        self.equity_curve = []
        self.trade_log = []

    def update(self, price, signal, timestamp):
        if self.last_price is None:
            self.last_price = price

        # Execute based on signal
        if signal == "BUY" and self.position <= 0:
            # close short if any
            if self.position < 0:
                self.cash += self.position * (self.last_price - price)
            self.position = 1
            self.entry_price = price

        elif signal == "SELL" and self.position >= 0:
            # close long if any
            if self.position > 0:
                self.cash += (price - self.entry_price)
            self.position = -1
            self.entry_price = price

        equity = self.cash + (self.position * (price - self.entry_price if self.position != 0 else 0))
        self.equity_curve.append({"time": timestamp, "equity": equity})
        self.trade_log.append({
            "time": timestamp,
            "signal": signal,
            "price": price,
            "equity": equity
        })
        self.last_price = price

    def save_logs(self):
        df = pd.DataFrame(self.trade_log)
        df.to_csv(TRADE_LOG_PATH, index=False)
        print(f"📊 Trades saved to {TRADE_LOG_PATH}")


# =========================================================
# --- MAIN LOOP ---
# =========================================================
def run_cycle(model, threshold, feature_cols, trader):
    print("\n⏰ Running new prediction cycle...")
    df = fetch_latest_data()
    df = add_technical_features(df)

    if len(df) < 50:
        print("⚠️ Not enough data to compute features.")
        return

    X_live = df[feature_cols].iloc[-1:].values
    current_price = df["GOLD_Close"].iloc[-1]
    current_time = df["Date"].iloc[-1]

    prob = model.predict_proba(X_live)[0, 1]
    signal = "BUY" if prob > threshold else "SELL"

    print(f"🕒 {current_time} | GOLD: {current_price:.2f} | Signal: {signal} | Prob: {prob:.3f}")

    trader.update(current_price, signal, current_time)
    trader.save_logs()


# =========================================================
# --- ENTRY POINT ---
# =========================================================
if __name__ == "__main__":
    model, threshold, feature_cols = load_model()
    trader = SimulatedTrader(capital=INITIAL_CAPITAL)

    # Run immediately
    run_cycle(model, threshold, feature_cols, trader)

    # Schedule every hour
    schedule.every().hour.at(":00").do(run_cycle, model, threshold, feature_cols, trader)

    print("\n⏳ Real-time trading simulation started (updates every hour)...")
    while True:
        schedule.run_pending()
        time.sleep(60)
