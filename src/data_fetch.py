# ======================================
# fetch_data.py
# GOLD PRICE + MACRO + INDEX DATA PIPELINE (Optimized + Enhanced)
# ======================================

import os
import pandas as pd
import numpy as np
import yfinance as yf
from pandas_datareader import data as pdr
from datetime import datetime

# ---------- PATH CONFIG ----------
os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/final", exist_ok=True)

LOCAL_CSV = "data/XAU_USD_DATASET.csv"
FINAL_CSV = "data/final/final_dataset_daily.csv"
FINAL_PARQUET = "data/final/final_dataset_daily.parquet"

# ---------- DATE RANGE ----------
START_DATE = datetime(2004, 1, 1)
END_DATE = datetime.today()

# ======================================
# 1. Load Local Gold Data
# ======================================
print("📂 Loading local gold dataset...")

xau_df = pd.read_csv(LOCAL_CSV, thousands=",", quotechar='"', skipinitialspace=True)
xau_df.columns = xau_df.columns.str.strip()
xau_df.rename(columns={"Price": "Close", "Change %": "Change_pct"}, inplace=True)

xau_df["Date"] = pd.to_datetime(xau_df["Date"], errors="coerce")
xau_df.dropna(subset=["Date"], inplace=True)

if "Change_pct" in xau_df.columns:
    xau_df["Change_pct"] = (
        xau_df["Change_pct"].astype(str).str.replace("%", "", regex=False).astype(float) / 100.0
    )

for col in ["Open", "High", "Low", "Close"]:
    xau_df[col] = pd.to_numeric(xau_df[col], errors="coerce")

xau_df = xau_df[(xau_df["Date"] >= START_DATE) & (xau_df["Date"] <= END_DATE)]
xau_df = xau_df.sort_values("Date").reset_index(drop=True)
print(f"✅ Gold data loaded: {len(xau_df)} rows ({xau_df['Date'].min().date()} → {xau_df['Date'].max().date()})")

# ======================================
# 2. Feature Engineering
# ======================================
print("⚙️ Generating gold features...")

xau_df["Return"] = xau_df["Close"].pct_change()
xau_df["LogReturn"] = np.log(xau_df["Close"] / xau_df["Close"].shift(1))
xau_df["MA5"] = xau_df["Close"].rolling(5).mean()
xau_df["MA10"] = xau_df["Close"].rolling(10).mean()
xau_df["MA20"] = xau_df["Close"].rolling(20).mean()
xau_df["Volatility5"] = xau_df["Return"].rolling(5).std()
xau_df["Volatility10"] = xau_df["Return"].rolling(10).std()
xau_df["Volatility20"] = xau_df["Return"].rolling(20).std()

xau_df.to_parquet("data/raw/gold_spot_daily.parquet", index=False)

# ======================================
# 3. Fetch Macroeconomic Data (FRED)
# ======================================
print("🌍 Fetching macroeconomic data (FRED)...")

fred_symbols = {
    "DXY": "DTWEXBGS",
    "CPI": "CPIAUCSL",
    "FEDFUNDS": "FEDFUNDS",
    "CRUDE_OIL": "DCOILWTICO",
    "VIX": "VIXCLS",
    "US10Y": "DGS10",        # 新增：10年期美债收益率
    "M2": "M2SL"             # 新增：M2货币供应量
}

macro_frames = []
for label, symbol in fred_symbols.items():
    try:
        df = pdr.DataReader(symbol, "fred", START_DATE, END_DATE)
        df = df.rename(columns={symbol: label}).reset_index()
        df["Date"] = pd.to_datetime(df["DATE"], errors="coerce")
        df.drop(columns=["DATE"], inplace=True)
        macro_frames.append(df)
        print(f"✅ {label} fetched")
    except Exception as e:
        print(f"⚠️ Failed {label}: {e}")

if macro_frames:
    macro_df = macro_frames[0]
    for df in macro_frames[1:]:
        macro_df = pd.merge(macro_df, df, on="Date", how="outer")
else:
    macro_df = pd.DataFrame(columns=["Date"])

macro_df = macro_df.sort_values("Date").reset_index(drop=True)
macro_df.to_parquet("data/raw/macro_fred.parquet", index=False)

# ======================================
# 4. Fetch Stock Indices (Yahoo Finance)
# ======================================
print("📈 Fetching stock indices & commodity prices...")

indices = {
    "SP500": "^GSPC",
    "NASDAQ": "^IXIC",
    "DJIA": "^DJI",           # 新增：道琼斯
    "GDX": "GDX",             # 黄金矿业ETF
    "USO": "USO",             # 原油ETF
    "SI": "SI=F",             # 白银
    "PL": "PL=F",             # 铂金
    "PA": "PA=F",             # 钯金
    "HG": "HG=F"              # 铜
}
idx_list = []

for name, ticker in indices.items():
    try:
        df = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
        if "Close" not in df.columns:
            print(f"⚠️ {name} missing 'Close' column, skipped.")
            continue
        df = df[["Close"]].reset_index()
        df.rename(columns={"Close": name}, inplace=True)
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        idx_list.append(df)
        print(f"✅ {name} fetched")
    except Exception as e:
        print(f"⚠️ Failed {name}: {e}")

if idx_list:
    idx_df = idx_list[0]
    for df in idx_list[1:]:
        idx_df = pd.merge(idx_df, df, on="Date", how="outer")
else:
    idx_df = pd.DataFrame(columns=["Date"])

idx_df = idx_df.sort_values("Date").reset_index(drop=True)
idx_df.to_parquet("data/raw/stock_indices.parquet", index=False)

# ======================================
# 5. Merge All Data
# ======================================
print("🔗 Merging all datasets...")

for df in [xau_df, macro_df, idx_df]:
    df.columns = df.columns.map(str)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

merged = pd.merge_asof(xau_df.sort_values("Date"), macro_df.sort_values("Date"), on="Date")
merged = pd.merge_asof(merged.sort_values("Date"), idx_df.sort_values("Date"), on="Date")
merged = merged.drop_duplicates(subset="Date").reset_index(drop=True)

# ======================================
# 6. Feature Cleanup & Derived Ratios
# ======================================
print("🧹 Cleaning up & generating derived ratios...")

# 删除冗余列
to_drop = ["Change_pct"]
merged = merged.drop(columns=[c for c in to_drop if c in merged.columns], errors="ignore")

# 前向填充 + 后向填充（确保月度宏观指标对齐到每日）
merged = merged.ffill().bfill()

# 去掉因rolling而产生的前期NaN
merged = merged.dropna(subset=["MA20", "Volatility20"]).reset_index(drop=True)

# 比率特征
if "DXY" in merged.columns:
    merged["Gold_vs_DXY"] = merged["Close"] / merged["DXY"]
if "SP500" in merged.columns:
    merged["Gold_vs_SP500"] = merged["Close"] / merged["SP500"]
if "SI" in merged.columns:
    merged["Gold_vs_Silver"] = merged["Close"] / merged["SI"]

# ======================================
# 7. Save Final Dataset
# ======================================
merged.to_csv(FINAL_CSV, index=False)
merged.to_parquet(FINAL_PARQUET, index=False)

print("\n✅ Final dataset generated successfully:")
print(f"📄 CSV: {FINAL_CSV}")
print(f"📦 Parquet: {FINAL_PARQUET}")
print(f"📊 Shape: {merged.shape}")
print(f"📅 Date Range: {merged['Date'].min().date()} → {merged['Date'].max().date()}")
print(f"🧠 Columns: {list(merged.columns)}")
