# src/utils/market_regime_detector_custom.py
"""
Enhanced Market Regime Detector (compatible with train_gold_model.py)
Inspired by ai-gold-scalper's market_regime_detector.py
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

class MarketRegimeDetector:
    def __init__(self, ma_fast=20, ma_slow=50):
        self.ma_fast = ma_fast
        self.ma_slow = ma_slow

    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        close = df["Close"]

        # Moving Averages
        df["MA_fast"] = close.rolling(self.ma_fast).mean()
        df["MA_slow"] = close.srolling(self.ma_slow).mean()

        # Momentum: slope of MA_fast
        df["slope"] = df["MA_fast"].diff(self.ma_fast) / df["MA_fast"]

        # Volatility (standard deviation of returns)
        df["volatility"] = close.pct_change().rolling(20).std()

        # Trend strength ratio
        df["trend_ratio"] = df["MA_fast"] / df["MA_slow"]

        df = df.dropna().reset_index(drop=True)
        return df

    # --- 修正版 detect_regime() ---
    def detect_regime(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        close = df["Close"]

        # 移动平均
        df["MA_fast"] = close.rolling(self.ma_fast).mean()
        df["MA_slow"] = close.rolling(self.ma_slow).mean()

        # 价格斜率（趋势强度）
        df["slope"] = df["MA_fast"].diff(self.ma_fast) / df["MA_fast"]

        # 波动率
        df["volatility"] = close.pct_change().rolling(20).std()

        # 市场趋势比率
        df["trend_ratio"] = df["MA_fast"] / df["MA_slow"]

        # 定义简单阈值逻辑
        bull_cond = (df["MA_fast"] > df["MA_slow"]) & (df["slope"] > 0)
        bear_cond = (df["MA_fast"] < df["MA_slow"]) & (df["slope"] < 0)
        neutral_cond = ~(bull_cond | bear_cond)

        # ✅ 关键修复：让默认值为字符串类型而非数字
        df["regime"] = np.select(
            [bull_cond, bear_cond, neutral_cond],
            ["bull", "bear", "neutral"],
            default="neutral"  # <---- 修复重点
        )

        return df[["Date", "Close", "MA_fast", "MA_slow", "regime"]]


