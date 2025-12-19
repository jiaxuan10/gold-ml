import os
import sys
import pandas as pd
import numpy as np

current_file = os.path.abspath(__file__) 
features_dir = os.path.dirname(current_file) 
src_dir = os.path.dirname(features_dir) 
root_dir = os.path.dirname(src_dir) 

if src_dir not in sys.path:
    sys.path.append(src_dir)

try:
    from utils.market_regime_detector import MarketRegimeDetector
except ImportError:
    print("Error importing MarketRegimeDetector. Regime detection will be disabled.")
    MarketRegimeDetector = None

def add_technical_features(df):
    df = df.copy()
    
    if "GOLD_Close" in df.columns:
        close = pd.to_numeric(df["GOLD_Close"], errors='coerce')
        high = pd.to_numeric(df["GOLD_High"], errors='coerce')
        low = pd.to_numeric(df["GOLD_Low"], errors='coerce')
    elif "Close" in df.columns:
        close = pd.to_numeric(df["Close"], errors='coerce')
        high = pd.to_numeric(df["High"], errors='coerce')
        low = pd.to_numeric(df["Low"], errors='coerce')
    else:
        return df

    df["SMA_20"] = close.rolling(20).mean()
    df["SMA_50"] = close.rolling(50).mean()
    df["EMA_12"] = close.ewm(span=12).mean()
    df["EMA_26"] = close.ewm(span=26).mean()
    df["MACD"] = df["EMA_12"] - df["EMA_26"]
    df["Signal_Line"] = df["MACD"].ewm(span=9).mean()
    
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df["RSI_14"] = 100 - (100 / (1 + rs))
    
    macro_assets = ["DX-Y.NYB", "BTC-USD", "^GSPC", "^IXIC", "^VIX", "ETH-USD"]
    for col in macro_assets:
        if col in df.columns:
            df[f"Gold_{col}_Ratio"] = close / (pd.to_numeric(df[col], errors='coerce') + 1e-9)
            df[f"Corr_{col}_24h"] = close.rolling(24).corr(pd.to_numeric(df[col], errors='coerce'))
        else:
            df[f"Gold_{col}_Ratio"] = 0.0
            df[f"Corr_{col}_24h"] = 0.0

    df["Return"] = close.pct_change()

    df["Lag_Return_1h"] = df["Return"].shift(1) 
    df["Lag_Return_6h"] = df["Return"].shift(6)

    df["MA_diff"] = df["SMA_20"] - df["SMA_50"]
    df["vol_24h"] = df["Return"].rolling(24).std()
    
    s20 = df["SMA_20"].fillna(0); s50 = df["SMA_50"].fillna(0)
    df["momentum_ok"] = (s20 > s50).astype(int)
    
    hc = (high - close.shift(1)).abs()
    lc = (low - close.shift(1)).abs()
    tr = np.maximum(high - low, np.maximum(hc, lc))
    df["ATR_14"] = pd.Series(tr).rolling(14).mean()
    df["ATR"] = df["ATR_14"]

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], utc=True)
        df["Hour"] = df["Date"].dt.hour
        df["DayOfWeek"] = df["Date"].dt.dayofweek
        df["Hour_Sin"] = np.sin(2 * np.pi * df["Hour"] / 24)
        df["Hour_Cos"] = np.cos(2 * np.pi * df["Hour"] / 24)

    df = df.replace([np.inf, -np.inf], np.nan)
    return df
if __name__ == "__main__":
    print("✅ features_engineering.py compiled successfully.")

def prepare_target(df, horizon=5, threshold=0.0005):
    df = df.copy()
    
    if "GOLD_Close" in df.columns:
        close = pd.to_numeric(df["GOLD_Close"], errors='coerce')
    elif "Close" in df.columns:
        close = pd.to_numeric(df["Close"], errors='coerce')
    else:
        raise ValueError("prepare_target: lack of price column (GOLD_Close or Close)")
    
    df["target_ret"] = (close.shift(-horizon) / close - 1)
    df["target_bin"] = np.where(df["target_ret"] > threshold, 1, 
                                 np.where(df["target_ret"] < -threshold, 0, np.nan))
    return df

def detect_market_regime(df):

    if MarketRegimeDetector is None:
        df["regime"] = "neutral"
        return df
        
    try:
        detector = MarketRegimeDetector(ma_fast=20, ma_slow=50)
        temp_df = df.copy()
        if "GOLD_Close" in temp_df.columns:
            temp_df = temp_df.rename(columns={"GOLD_Close": "Close"})
        
        regime_df = detector.detect_regime(temp_df)
        
        if "regime" in regime_df.columns:
            df["regime"] = regime_df["regime"]
        else:
            df["regime"] = "neutral"
    except Exception as e:
        print(f"⚠️ Regime Detection failed: {e}")
        df["regime"] = "neutral"
        
    return df

def add_technical_features_backtest(df):

    df = df.copy()
    
    if "GOLD_Close" in df.columns:
        close = pd.to_numeric(df["GOLD_Close"], errors='coerce')
        high = pd.to_numeric(df["GOLD_High"], errors='coerce')
        low = pd.to_numeric(df["GOLD_Low"], errors='coerce')
    elif "Close" in df.columns:
        close = pd.to_numeric(df["Close"], errors='coerce')
        high = pd.to_numeric(df["High"], errors='coerce')
        low = pd.to_numeric(df["Low"], errors='coerce')
    else:
        raise ValueError("❌ Lack of price column (GOLD_Close or Close)")
    
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], utc=True)
    

    
    sma_20 = close.rolling(20, min_periods=5).mean()
    sma_50 = close.rolling(50, min_periods=10).mean()
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    macd = ema_12 - ema_26
    
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14, min_periods=3).mean()
    loss = (-delta.clip(upper=0)).rolling(14, min_periods=3).mean()
    rs = gain / (loss.replace(0, np.nan))
    rsi_14 = 100 - (100 / (1 + rs))
    
    hl = high - low
    hc = (high - close.shift(1)).abs()
    lc = (low - close.shift(1)).abs()
    tr = np.maximum.reduce([hl, hc, lc])
    atr_14 = pd.Series(tr).rolling(14, min_periods=3).mean()
    
    ret = close.pct_change()
    vol_20 = ret.rolling(20, min_periods=5).std()
    mom_10 = close.pct_change(10)
    
    vol_24h = close.pct_change().rolling(24).std()
    

    df["SMA_20"] = sma_20.shift(1)
    df["SMA_50"] = sma_50.shift(1)
    df["EMA_12"] = ema_12.shift(1)
    df["EMA_26"] = ema_26.shift(1)
    df["MACD"] = macd.shift(1)
    df["Signal_Line"] = macd.ewm(span=9, adjust=False).mean().shift(1)
    
    df["RSI_14"] = rsi_14.shift(1)
    df["ATR_14"] = atr_14.shift(1)
    df["ATR"] = atr_14.shift(1)  
    
    df["Return"] = ret.shift(1)
    df["Volatility_20"] = vol_20.shift(1)
    df["Momentum_10"] = mom_10.shift(1)
    
    macro_assets = ["DX-Y.NYB", "BTC-USD", "^GSPC", "^IXIC", "^VIX", "ETH-USD"]
    for col in macro_assets:
        if col in df.columns:
            ratio = close / (pd.to_numeric(df[col], errors='coerce') + 1e-9)
            corr = close.rolling(24).corr(pd.to_numeric(df[col], errors='coerce'))
            df[f"Gold_{col}_Ratio"] = ratio.shift(1)
            df[f"Corr_{col}_24h"] = corr.shift(1)
        else:
            df[f"Gold_{col}_Ratio"] = 0.0
            df[f"Corr_{col}_24h"] = 0.0
    
    df["Lag_Return_1h"] = df["Return"].shift(1) 
    df["Lag_Return_6h"] = df["Return"].shift(6)
    
    df["MA_diff"] = df["SMA_20"] - df["SMA_50"]
    df["vol_24h"] = vol_24h.shift(1)
    
    df["momentum_ok"] = (df["SMA_20"] > df["SMA_50"]).astype(int)
    
    if "Date" in df.columns:
        df["Hour"] = df["Date"].dt.hour
        df["DayOfWeek"] = df["Date"].dt.dayofweek
        df["Hour_Sin"] = np.sin(2 * np.pi * df["Hour"] / 24)
        df["Hour_Cos"] = np.cos(2 * np.pi * df["Hour"] / 24)
    
    df = df.replace([np.inf, -np.inf], np.nan)
    
    return df

if __name__ == "__main__":
    print("✅ features_engineering.py compiled successfully.")