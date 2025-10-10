# ======================================
# GOLD PRICE DATA COLLECTION (XAU/USD)
# Short-Mid Term Prediction (1~15 days)
# ======================================

import os
import pandas as pd
import yfinance as yf
from pandas_datareader import data as pdr
from alpha_vantage.timeseries import TimeSeries
from datetime import datetime, timedelta

# ---------- PATH CONFIG ----------
os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/final", exist_ok=True)

FINAL_CSV = "data/final/final_dataset_daily.csv"
FINAL_PARQUET = "data/final/final_dataset_daily.parquet"

# ---------- DATA RANGE ----------
END_DATE = datetime.today()
START_DATE = END_DATE - timedelta(days=5*365)  # 最近5年

# ---------- HELPER: Incremental Update ----------
def incremental_update(df_old, df_new, date_col="Date"):
    df_new = df_new.reset_index(drop=True)
    df_old = df_old.reset_index(drop=True)
    df_merged = pd.concat([df_old, df_new]).drop_duplicates(subset=[date_col])
    df_merged = df_merged.sort_values(by=date_col).reset_index(drop=True)
    return df_merged

# ---------- 1. XAU/USD Daily OHLCV ----------
print("Fetching XAU/USD from Alpha Vantage...")
ts = TimeSeries(key="Y0QJSPYANK6SFHKC", output_format="pandas")
xau_df, _ = ts.get_daily(symbol="XAUUSD", outputsize="full")
xau_df = xau_df.reset_index()
xau_df.rename(columns={
    "date": "Date",
    "1. open": "Open",
    "2. high": "High",
    "3. low": "Low",
    "4. close": "Close",
    "5. volume": "Volume"
}, inplace=True)
xau_df["Date"] = pd.to_datetime(xau_df["Date"])
xau_df = xau_df[(xau_df["Date"] >= START_DATE) & (xau_df["Date"] <= END_DATE)]
xau_df = xau_df.sort_values("Date").reset_index(drop=True)
xau_df["Ticker"] = "XAUUSD"

# ---------- 1a. Feature Engineering ----------
# 日收益率
xau_df["Return"] = xau_df["Close"].pct_change()

# 移动平均线
xau_df["MA5"] = xau_df["Close"].rolling(5).mean()
xau_df["MA10"] = xau_df["Close"].rolling(10).mean()
xau_df["MA15"] = xau_df["Close"].rolling(15).mean()

# 波动率 (rolling std)
xau_df["Volatility5"] = xau_df["Return"].rolling(5).std()
xau_df["Volatility10"] = xau_df["Return"].rolling(10).std()

xau_df.to_parquet("data/raw/gold_spot_daily.parquet", index=False)

# ---------- 2. Macro Data (FRED) ----------
fred_symbols = {
    "DXY": "DTWEXBGS",
    "CPI": "CPIAUCSL",
    "FEDFUNDS": "FEDFUNDS",
    "CRUDE_OIL": "DCOILWTICO"
}

macro_df = pd.DataFrame()
for label, symbol in fred_symbols.items():
    try:
        temp = pdr.DataReader(symbol, "fred", START_DATE, END_DATE)
        temp.rename(columns={symbol: label}, inplace=True)
        macro_df = pd.concat([macro_df, temp], axis=1)
    except Exception as e:
        print(f"⚠️ Failed to fetch {label}: {e}")

macro_df = macro_df.reset_index()
macro_df.rename(columns={macro_df.columns[0]: "Date"}, inplace=True)
macro_df["Date"] = pd.to_datetime(macro_df["Date"])
macro_df.to_parquet("data/raw/macro_fred.parquet", index=False)

# ---------- 3. Stock Indices ----------
indices = {"S&P500": "^GSPC", "NASDAQ": "^IXIC"}
index_data = pd.DataFrame()

for name, ticker in indices.items():
    df = yf.download(ticker, start=START_DATE, end=END_DATE)[["Close"]].copy()
    
    # 避免 MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(-1)
    
    df = df.reset_index()  # Date 从 index 转为列
    df.rename(columns={"Close": name}, inplace=True)
    df["Date"] = pd.to_datetime(df["Date"])
    
    index_data = pd.merge(index_data, df, on="Date", how="outer") if not index_data.empty else df

# 按日期排序
index_data = index_data.sort_values("Date").reset_index(drop=True)
index_data.to_parquet("data/raw/stock_indices.parquet", index=False)


# ---------- 4. Merge All ----------
merged = xau_df.merge(macro_df, on="Date", how="left")
merged = pd.merge_asof(
    merged.sort_values("Date"),
    index_data.sort_values("Date"),
    on="Date"
)

# ---------- 5. Save ----------
merged.to_csv(FINAL_CSV, index=False)
merged.to_parquet(FINAL_PARQUET, index=False)

print(f"✅ Final dataset saved: {FINAL_PARQUET} & {FINAL_CSV}")
print(f"Shape: {merged.shape}")
print(f"Date range: {merged['Date'].min()} → {merged['Date'].max()}")
print("Most common gap (days):", merged['Date'].diff().dropna().mode()[0].days)
print("Sample rows:")
print(merged.head())
