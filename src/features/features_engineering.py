import os
import sys
import pandas as pd
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
if src_dir not in sys.path:
    sys.path.append(src_dir)

from utils.market_regime_detector import MarketRegimeDetector
def add_technical_features(df):
    """
    正确的特征工程函数 - 无额外shift，在回测时统一做shift(1)
    """
    df = df.copy()
    
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

    close = pd.to_numeric(close, errors='coerce')
    high = pd.to_numeric(high, errors='coerce')
    low = pd.to_numeric(low, errors='coerce')

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], utc=True)

    # ============================================
    # ✅ 正确版本：移除所有.shift(1)，只在回测时统一做shift
    # 这样特征在时间t包含t及之前的信息
    # ============================================
    
    # 移动平均线 - 无shift
    df["SMA_20"] = close.rolling(20, min_periods=5).mean()
    df["SMA_50"] = close.rolling(50, min_periods=10).mean()
    df["EMA_12"] = close.ewm(span=12, adjust=False).mean()
    df["EMA_26"] = close.ewm(span=26, adjust=False).mean()
    df["MACD"] = df["EMA_12"] - df["EMA_26"]
    
    # RSI - 无shift（RSI在时间t需要close[t]）
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14, min_periods=3).mean()
    loss = (-delta.clip(upper=0)).rolling(14, min_periods=3).mean()
    rs = gain / (loss.replace(0, np.nan))
    df["RSI_14"] = 100 - (100 / (1 + rs))
    
    # ATR - 无shift（ATR在时间t需要high[t], low[t], close[t-1]）
    hl = high - low
    hc = (high - close.shift()).abs()  # 这里的.shift()是正确的，使用前一天的close
    lc = (low - close.shift()).abs()
    tr = np.maximum.reduce([hl, hc, lc])
    df["ATR_14"] = pd.Series(tr).rolling(14, min_periods=3).mean()
    
    # 其他指标 - 无shift
    df["Return"] = close.pct_change()
    df["Volatility_20"] = df["Return"].rolling(20, min_periods=5).std()
    df["Momentum_10"] = close.pct_change(10)
    
    # 相关性特征 - 无shift
    for col in df.columns:
        if col in ["DX-Y.NYB", "BTC-USD", "^GSPC", "^IXIC"]:
            df[f"Gold_{col}_Ratio"] = close / (pd.to_numeric(df[col], errors='coerce') + 1e-9)
            df[f"Corr_{col}_24h"] = close.rolling(24).corr(pd.to_numeric(df[col], errors='coerce'))
    
    # 滞后特征 - 这些需要shift，表示过去的信息
    df["Lag_Return_1h"] = df["Return"].shift(1)  # 1小时前的收益率
    df["Lag_Return_6h"] = df["Return"].shift(6)  # 6小时前的收益率
    df["MA_diff"] = df["SMA_20"] - df["SMA_50"]  # 无shift，差值本身
    
    # Z-score - 无shift
    rolling_mean = close.rolling(24).mean()
    rolling_std = close.rolling(24).std()
    df["Rolling_Zscore_24h"] = (close - rolling_mean) / (rolling_std + 1e-9)
    
    # 时间特征（这些是确定性的，不需要shift）
    if "Date" in df.columns:
        df["Hour"] = df["Date"].dt.hour
        df["DayOfWeek"] = df["Date"].dt.dayofweek
        df["Hour_Sin"] = np.sin(2 * np.pi * df["Hour"] / 24)
        df["Hour_Cos"] = np.cos(2 * np.pi * df["Hour"] / 24)
    
    # ============================================
    # 🔥 为策略类添加必需的特征
    # ============================================
    df["ATR"] = df["ATR_14"]  # 为策略创建ATR别名
    df["vol_24h"] = df["Close"].pct_change().rolling(24).std().fillna(0)
    df["momentum_ok"] = (df["SMA_20"] > df["SMA_50"]).astype(int)
    
    df = df.replace([np.inf, -np.inf], np.nan)
    
    return df

def prepare_target(df, horizon=5, threshold=0.0005):
    df = df.copy()
    if "GOLD_Close" in df.columns:
        close = pd.to_numeric(df["GOLD_Close"], errors='coerce')
    else:
        close = pd.to_numeric(df["Close"], errors='coerce')
    df["target_ret"] = (close.shift(-horizon) / close - 1)
    df["target_bin"] = np.where(df["target_ret"] > threshold, 1, np.where(df["target_ret"] < -threshold, 0, np.nan))
    return df

def detect_market_regime(df):
    detector = MarketRegimeDetector(ma_fast=20, ma_slow=50)
    temp_df = df.copy()
    if "GOLD_Close" in temp_df.columns and "Close" not in temp_df.columns:
        temp_df = temp_df.rename(columns={"GOLD_Close": "Close"})
    regime_df = detector.detect_regime(temp_df)
    if "regime" not in df.columns:
        df = pd.merge(df, regime_df[["Date", "regime"]], on="Date", how="left")
    return df

if __name__ == "__main__":
    print("✅ features_engineering.py compiled successfully.")