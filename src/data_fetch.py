# src/data_fetch.py
import os
import pandas as pd
import yfinance as yf

def fetch_gold_history_csv(ticker="GC=F", period="5y", interval="1d", save_path="data/raw/gold_ohlcv.csv"):
    """拉取黄金期货历史数据并保存为 CSV"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    print(f"📡 Fetching {ticker} historical data from Yahoo Finance...")
    
    df = yf.download(tickers=ticker, period=period, interval=interval, progress=False)
    
    if df.empty:
        raise ValueError("无法获取历史数据，请检查 ticker/网络")
    
    df = df.reset_index()
    if "Date" in df.columns:
        df.rename(columns={"Date":"date"}, inplace=True)
    
    for col in ["Open","High","Low","Close","Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    df.to_csv(save_path, index=False)
    print(f"✅ Historical data saved to {save_path} ({len(df)} rows)")
    return df

if __name__ == "__main__":
    fetch_gold_history_csv()
