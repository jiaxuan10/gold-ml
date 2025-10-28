import pandas as pd
import numpy as np

def add_indicators(df: pd.DataFrame):
    df = df.copy()
    
    # 移动平均
    for window in [5, 10, 20]:
        df[f"MA{window}"] = df["Close"].rolling(window).mean()
    
    # 波动率（标准差）
    for window in [5, 10, 20]:
        df[f"Volatility{window}"] = df["Return"].rolling(window).std()
    
    # RSI
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df["RSI14"] = 100 - (100 / (1 + rs))
    
    # MACD
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    
    # 布林带
    df["BB_Mid"] = df["Close"].rolling(20).mean()
    df["BB_Upper"] = df["BB_Mid"] + 2 * df["Close"].rolling(20).std()
    df["BB_Lower"] = df["BB_Mid"] - 2 * df["Close"].rolling(20).std()
    
    # 滞后特征
    for lag in [1, 2, 3]:
        df[f"Return_lag{lag}"] = df["Return"].shift(lag)
        df[f"Close_lag{lag}"] = df["Close"].shift(lag)
    
    # 未来一天回报（预测目标）
    df["Return_next"] = df["Close"].pct_change().shift(-1)
    
    # 清理
    df = df.dropna().reset_index(drop=True)
    return df
