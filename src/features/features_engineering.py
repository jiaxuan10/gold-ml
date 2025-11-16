import pandas as pd
import numpy as np

from utils.market_regime_detector import MarketRegimeDetector



# ---------- FEATURE CREATION ----------
def add_technical_features(df):
    df = df.copy()
    for col in ["Close","Open","High","Low"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    close = df["Close"]

    # moving averages
    df["sma_5"] = close.rolling(5).mean()
    df["sma_10"] = close.rolling(10).mean()
    df["sma_20"] = close.rolling(20).mean()
    df["ema_12"] = close.ewm(span=12, adjust=False).mean()
    df["ema_26"] = close.ewm(span=26, adjust=False).mean()

    # volatility & momentum
    df["vol_5"] = close.pct_change().rolling(5).std()
    df["mom_3"] = close.pct_change(3)
    df["mom_5"] = close.pct_change(5)

    # Bollinger band position (normalized)
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    # avoid division by zero
    df["bb_pos"] = (close - (bb_mid - 2*bb_std)) / (4*bb_std.replace(0, np.nan))

    # RSI
    delta = close.diff()
    gain = delta.where(delta>0, 0).rolling(14).mean()
    loss = (-delta.where(delta<0, 0)).rolling(14).mean()
    rs = gain / (loss.replace(0, np.nan))
    df["rsi"] = 100 - (100 / (1 + rs))

    # MACD
    macd = df["ema_12"] - df["ema_26"]
    signal = macd.ewm(span=9, adjust=False).mean()
    df["macd"] = macd
    df["macd_signal"] = signal
    df["macd_hist"] = macd - signal

    # Price ratios (scalper style) and 24h correlation if available
    for col in [c for c in df.columns if ("DX" in c or "SP" in c or "BTC" in c)]:
        # guard division
        df[f"Gold_{col}_Ratio"] = df["Close"] / df[col].replace(0, np.nan)
        df[f"Corr_{col}_24h"] = df["Close"].rolling(24).corr(df[col])

    # lag features (common in time series)
    for lag in [1,2,3,6,12]:
        df[f"ret_lag_{lag}"] = df["Close"].pct_change(lag)

    # ma differences
    df["ma5_ma20"] = df["sma_5"] - df["sma_20"]
    # --- Structural / volatility ratio features ---
    df["high_break"] = (df["Close"] > df["High"].rolling(24).max()).astype(int)
    df["low_break"] = (df["Close"] < df["Low"].rolling(24).min()).astype(int)
    df["vol_ratio"] = df["vol_5"] / df["vol_5"].rolling(20).mean()
    df["ret_accel"] = df["Close"].pct_change() - df["Close"].pct_change(2)
    df["rsi_macd_interact"] = df["rsi"] * df["macd"]

    return df


def prepare_target(df, horizon=5, threshold= 0.0005):
    df = df.copy()
    df["target_ret"] = (df["Close"].shift(-horizon) / df["Close"] - 1)
    # thresholded label: ignore small moves -> NaN
    df["target_bin"] = np.where(df["target_ret"] > threshold, 1,
                                np.where(df["target_ret"] < -threshold, 0, np.nan))
    return df


def detect_market_regime(df):
    detector = MarketRegimeDetector(ma_fast=20, ma_slow=50)
    regime_df = detector.detect_regime(df)
    # merge on Date (assumes detector returns Date)
    df = pd.merge(df, regime_df[["Date", "regime"]], on="Date", how="left")
    return df



