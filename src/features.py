# src/features.py
import pandas as pd
import ta
from src.utils import ensure_dir

def generate_features(input_path="data/raw/gold_ohlcv.csv", output_path="data/processed/gold_features.csv"):
    """
    根据黄金历史数据生成技术指标（Technical Indicators）。
    """
    ensure_dir("data/processed")

    df = pd.read_csv(input_path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    print("⚙️ Generating technical indicators...")

    # ---- 常见技术指标 ----
    # 移动平均线
    df["SMA_10"] = df["Close"].rolling(window=10).mean()
    df["SMA_50"] = df["Close"].rolling(window=50).mean()
    df["EMA_20"] = ta.trend.ema_indicator(df["Close"], window=20)

    # 动量指标
    df["RSI_14"] = ta.momentum.rsi(df["Close"], window=14)
    df["MACD"] = ta.trend.macd(df["Close"])
    df["MACD_Signal"] = ta.trend.macd_signal(df["Close"])
    df["Stoch_RSI"] = ta.momentum.stochrsi(df["Close"])

    # 波动率指标
    df["ATR_14"] = ta.volatility.average_true_range(df["High"], df["Low"], df["Close"], window=14)
    df["Bollinger_High"] = ta.volatility.bollinger_hband(df["Close"])
    df["Bollinger_Low"] = ta.volatility.bollinger_lband(df["Close"])

    # 收益率
    df["Return"] = df["Close"].pct_change()
    df["LogReturn"] = (df["Close"] / df["Close"].shift(1)).apply(lambda x: np.log(x))

    # 删除缺失值（由滚动窗口产生）
    df.dropna(inplace=True)

    df.to_csv(output_path, index=False)
    print(f"✅ Features saved to {output_path} ({len(df)} rows)")

    return df

if __name__ == "__main__":
    generate_features()
