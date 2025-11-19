import pandas as pd
import numpy as np
from utils.market_regime_detector import MarketRegimeDetector

def add_technical_features(df):
    """
    Unified Feature Engineering.
    Expects: df with 'GOLD_Close' (or 'Close').
    Outputs: df with Capitalized Feature names (SMA_20, RSI_14) to match your CSV.
    """
    df = df.copy()
    
    # 1. Standardize Input Columns (Handle GOLD_ prefix)
    # We map GOLD_Close to a temporary 'close' variable for calculation
    if "GOLD_Close" in df.columns:
        close = df["GOLD_Close"]
        high = df["GOLD_High"]
        low = df["GOLD_Low"]
    elif "Close" in df.columns:
        close = df["Close"]
        high = df["High"]
        low = df["Low"]
    else:
        raise ValueError("❌ dataframe missing 'GOLD_Close' or 'Close' column")

    # Ensure numeric
    close = pd.to_numeric(close, errors='coerce')
    high = pd.to_numeric(high, errors='coerce')
    low = pd.to_numeric(low, errors='coerce')

    # --- 2. Overwrite/Create Features (Use CAPITAL names to match CSV) ---
    
    # Moving Averages
    df["SMA_20"] = close.rolling(20, min_periods=5).mean()
    df["SMA_50"] = close.rolling(50, min_periods=10).mean()
    df["EMA_12"] = close.ewm(span=12, adjust=False).mean()
    df["EMA_26"] = close.ewm(span=26, adjust=False).mean()

    # MACD
    df["MACD"] = df["EMA_12"] - df["EMA_26"]
    df["Signal_Line"] = df["MACD"].ewm(span=9, adjust=False).mean()

    # RSI (14)
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14, min_periods=3).mean()
    loss = (-delta.clip(upper=0)).rolling(14, min_periods=3).mean()
    rs = gain / (loss.replace(0, np.nan))
    df["RSI_14"] = 100 - (100 / (1 + rs))

    # ATR (14)
    hl = high - low
    hc = (high - close.shift()).abs()
    lc = (low - close.shift()).abs()
    tr = np.maximum.reduce([hl, hc, lc])
    df["ATR_14"] = pd.Series(tr).rolling(14, min_periods=3).mean()

    # Volatility & Momentum
    df["Return"] = close.pct_change()
    df["Volatility_20"] = df["Return"].rolling(20, min_periods=5).std()
    df["Momentum_10"] = close.pct_change(10)

    # Cross-Asset Ratios (Safe Logic)
    # Check for external assets in columns (e.g. DX-Y.NYB)
    for col in df.columns:
        if col in ["DX-Y.NYB", "BTC-USD", "^GSPC", "^IXIC"]:
            # Create Ratio and Correlation
            df[f"Gold_{col}_Ratio"] = close / (pd.to_numeric(df[col], errors='coerce') + 1e-9)
            df[f"Corr_{col}_24h"] = close.rolling(24).corr(pd.to_numeric(df[col], errors='coerce'))

    # Advanced Features
    df["Lag_Return_1h"] = df["Return"].shift(1)
    df["Lag_Return_6h"] = df["Return"].shift(6)
    df["MA_diff"] = df["SMA_20"] - df["SMA_50"]

    # Z-Score
    rolling_mean = close.rolling(24).mean()
    rolling_std = close.rolling(24).std()
    df["Rolling_Zscore_24h"] = (close - rolling_mean) / (rolling_std + 1e-9)
    
    # Fill NaNs for training stability
    df = df.replace([np.inf, -np.inf], np.nan)
    
    return df

def prepare_target(df, horizon=5, threshold=0.0005):
    """
    Creates target labels. 
    Note: We check for GOLD_Close or Close again to be safe.
    """
    df = df.copy()
    if "GOLD_Close" in df.columns:
        close = pd.to_numeric(df["GOLD_Close"], errors='coerce')
    else:
        close = pd.to_numeric(df["Close"], errors='coerce')
        
    # Create future return target
    df["target_ret"] = (close.shift(-horizon) / close - 1)
    
    # Label: 1 = Up, 0 = Down, NaN = Flat
    df["target_bin"] = np.where(df["target_ret"] > threshold, 1,
                                np.where(df["target_ret"] < -threshold, 0, np.nan))
    return df

def detect_market_regime(df):
    detector = MarketRegimeDetector(ma_fast=20, ma_slow=50)
    # Ensure detector uses correct column names if it relies on them
    # If MarketRegimeDetector expects "Close", we might need to rename temporarily
    temp_df = df.copy()
    if "GOLD_Close" in temp_df.columns and "Close" not in temp_df.columns:
        temp_df = temp_df.rename(columns={"GOLD_Close": "Close"})
        
    regime_df = detector.detect_regime(temp_df)
    
    if "regime" not in df.columns:
        df = pd.merge(df, regime_df[["Date", "regime"]], on="Date", how="left")
    return df