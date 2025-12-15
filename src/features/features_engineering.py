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
    Unified Feature Engineering (Final Corrected).
    Logic: Calculate ALL indicators on RAW data first, then apply ONE global shift.
    Benefit: No Double-Lag, No Leakage, No KeyError.
    """
    df = df.copy()
    
    # 1. 智能识别 Close 列 (Handle GOLD_Close vs Close)
    if "GOLD_Close" in df.columns:
        close = df["GOLD_Close"]
        high = df["GOLD_High"]
        low = df["GOLD_Low"]
    elif "Close" in df.columns:
        close = df["Close"]
        high = df["High"]
        low = df["Low"]
    else:
        # Fallback
        if "Adj Close" in df.columns:
            close = df["Adj Close"]
            high = df["High"]
            low = df["Low"]
        else:
            raise ValueError(f"❌ dataframe missing 'GOLD_Close' or 'Close'. Cols: {df.columns.tolist()}")

    close = pd.to_numeric(close, errors='coerce')
    high = pd.to_numeric(high, errors='coerce')
    low = pd.to_numeric(low, errors='coerce')

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], utc=True)

    # ============================================
    # 第一步：计算原始指标 (Raw Indicators) - 不加 Shift
    # ============================================
    
    # 趋势
    sma_20 = close.rolling(20, min_periods=5).mean()
    sma_50 = close.rolling(50, min_periods=10).mean()
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    macd = ema_12 - ema_26
    
    # 震荡
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14, min_periods=3).mean()
    loss = (-delta.clip(upper=0)).rolling(14, min_periods=3).mean()
    rs = gain / (loss.replace(0, np.nan))
    rsi_14 = 100 - (100 / (1 + rs))
    
    # ATR (需要昨收，所以这里内部有一个 shift(1) 是公式需要，不算泄露)
    hl = high - low
    hc = (high - close.shift(1)).abs()
    lc = (low - close.shift(1)).abs()
    tr = np.maximum.reduce([hl, hc, lc])
    atr_14 = pd.Series(tr).rolling(14, min_periods=3).mean()
    
    # 波动与动量
    ret = close.pct_change()
    vol_20 = ret.rolling(20, min_periods=5).std()
    mom_10 = close.pct_change(10)
    
    # 策略专用特征
    vol_24h = close.pct_change().rolling(24).std().fillna(0) # 修复：用 close 变量
    
    # Z-Score
    rolling_mean = close.rolling(24).mean()
    rolling_std = close.rolling(24).std()
    zscore = (close - rolling_mean) / (rolling_std + 1e-9)

    # ============================================
    # 第二步：统一赋值并 Shift(1) (防止泄露)
    # ============================================
    
    # 核心特征
    df["SMA_20"] = sma_20.shift(1)
    df["SMA_50"] = sma_50.shift(1)
    df["EMA_12"] = ema_12.shift(1)
    df["EMA_26"] = ema_26.shift(1)
    df["MACD"] = macd.shift(1)      # 修正：现在是 T-1
    
    df["RSI_14"] = rsi_14.shift(1)
    df["ATR_14"] = atr_14.shift(1)
    df["ATR"] = df["ATR_14"]        # 别名
    
    df["Return"] = ret.shift(1)
    df["Volatility_20"] = vol_20.shift(1)
    df["Momentum_10"] = mom_10.shift(1)
    
    # 外部宏观特征 (Raw -> Shift)
    for col in df.columns:
        if col in ["DX-Y.NYB", "BTC-USD", "^GSPC", "^IXIC"]:
            ratio = close / (pd.to_numeric(df[col], errors='coerce') + 1e-9)
            corr = close.rolling(24).corr(pd.to_numeric(df[col], errors='coerce'))
            df[f"Gold_{col}_Ratio"] = ratio.shift(1)
            df[f"Corr_{col}_24h"] = corr.shift(1)
            
    # 滞后特征 (基于已经 Shift 过的 Return 再 Shift)
    df["Lag_Return_1h"] = df["Return"].shift(1) # T-2
    df["Lag_Return_6h"] = df["Return"].shift(6)
    
    # 衍生特征
    df["MA_diff"] = df["SMA_20"] - df["SMA_50"] # 两个 T-1 相减，还是 T-1 (逻辑修正)
    df["Rolling_Zscore_24h"] = zscore.shift(1)
    
    # 策略特征 (修正：统一 Shift，防止泄露)
    df["vol_24h"] = vol_24h.shift(1) 
    
    # 动量确认 (SMA T-1 比较)
    if "SMA_20" in df.columns and "SMA_50" in df.columns:
        df["momentum_ok"] = (df["SMA_20"] > df["SMA_50"]).astype(int)

    # 时间特征 (无需 Shift，确定性数据)
    if "Date" in df.columns:
        df["Hour"] = df["Date"].dt.hour
        df["DayOfWeek"] = df["Date"].dt.dayofweek
        df["Hour_Sin"] = np.sin(2 * np.pi * df["Hour"] / 24)
        df["Hour_Cos"] = np.cos(2 * np.pi * df["Hour"] / 24)
    
    df = df.replace([np.inf, -np.inf], np.nan)
    
    return df

def prepare_target(df, horizon=5, threshold=0.0005):
    df = df.copy()
    
    # 🔥 确保使用与特征计算相同的价格列
    if "GOLD_Close" in df.columns:
        close = pd.to_numeric(df["GOLD_Close"], errors='coerce')
    elif "Close" in df.columns:
        close = pd.to_numeric(df["Close"], errors='coerce')
    else:
        raise ValueError("prepare_target: 缺少价格列")
    
    df["target_ret"] = (close.shift(-horizon) / close - 1)
    df["target_bin"] = np.where(df["target_ret"] > threshold, 1, 
                                 np.where(df["target_ret"] < -threshold, 0, np.nan))
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