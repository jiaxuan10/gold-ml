import os
import pandas as pd
import yfinance as yf
from indicators import add_indicators
import yaml

def load_config(path="config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

cfg = load_config("config.yaml")

RAW_DIR = cfg["paths"]["raw_dir"]
FINAL_DIR = cfg["paths"]["final_dir"]
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(FINAL_DIR, exist_ok=True)

start_date = cfg["date_range"]["start"]
end_date = cfg["date_range"]["end"]

# =============== Step 1. Fetch main assets ===============
symbols = cfg["symbols"]

def fetch(symbol, name):
    print(f"📈 Fetching {name} ({symbol}) ...")
    df = yf.download(symbol, start=start_date, end=end_date)
    df = df[["Open", "High", "Low", "Close", "Adj Close", "Volume"]]
    df["Return"] = df["Close"].pct_change()
    df["LogReturn"] = (df["Close"] / df["Close"].shift(1)).apply(lambda x: np.log(x))
    df = add_indicators(df)
    df.reset_index(inplace=True)
    df.rename(columns={"Date": "Date"}, inplace=True)
    df.to_csv(f"{RAW_DIR}/{name}.csv", index=False)
    return df

gold_df = fetch(symbols["gold"], "gold")
sp500_df = fetch(symbols["sp500"], "sp500")
dxy_df = fetch(symbols["dxy"], "dxy")
silver_df = fetch(symbols["silver"], "silver")

# =============== Step 2. Merge datasets ===============
merged = gold_df[["Date", "Close", "Return", "LogReturn"]].rename(
    columns={"Close": "Gold_Close", "Return": "Gold_Return", "LogReturn": "Gold_LogReturn"}
)

def merge_asset(df, name):
    df = df[["Date", "Close"]].rename(columns={"Close": f"{name}_Close"})
    return pd.merge_asof(merged.sort_values("Date"), df.sort_values("Date"), on="Date")

for name, df in [("SP500", sp500_df), ("DXY", dxy_df), ("Silver", silver_df)]:
    merged = merge_asset(df, name)

# =============== Step 3. Relative features ===============
merged["Gold_vs_SP500"] = merged["Gold_Close"] / merged["SP500_Close"]
merged["Gold_vs_DXY"] = merged["Gold_Close"] / merged["DXY_Close"]
merged["Gold_vs_Silver"] = merged["Gold_Close"] / merged["Silver_Close"]

# =============== Step 4. Export final dataset ===============
final_path = os.path.join(FINAL_DIR, "final_dataset_v3.csv")
merged.to_csv(final_path, index=False)
print(f"✅ Final dataset saved to {final_path}")
