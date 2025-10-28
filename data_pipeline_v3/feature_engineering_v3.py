
import os
import yaml
import pandas as pd
import numpy as np
import logging
import ta
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

with open('config.yaml') as f:
    cfg = yaml.safe_load(f)

RAW_DIR = cfg['paths']['raw_dir']
FINAL_DIR = cfg['paths']['final_dir']
OUTPUT = cfg['paths']['output_parquet']

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(FINAL_DIR, exist_ok=True)

# Load gold price
gold_path = os.path.join(RAW_DIR, 'gold_price.parquet')
if not os.path.exists(gold_path):
    raise FileNotFoundError('gold_price.parquet not found; run fetch_data_v3 first')

gold = pd.read_parquet(gold_path)
if 'Date' not in gold.columns:
    gold = gold.reset_index()

gold['Date'] = pd.to_datetime(gold['Date'], errors='coerce')

gold = gold.sort_values('Date').reset_index(drop=True)

# Load macro and indices if exist
macro_path = os.path.join(RAW_DIR, 'macro_fred.parquet')
idx_path = os.path.join(RAW_DIR, 'stock_indices.parquet')
extra_path = os.path.join(RAW_DIR, 'extra_assets.parquet')

macro = pd.read_parquet(macro_path) if os.path.exists(macro_path) else pd.DataFrame(columns=['Date'])
idx = pd.read_parquet(idx_path) if os.path.exists(idx_path) else pd.DataFrame(columns=['Date'])
extra = pd.read_parquet(extra_path) if os.path.exists(extra_path) else pd.DataFrame(columns=['Date'])

# Normalize column names and Date
for df in [macro, idx, extra]:
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

# Merge datasets using asof merge (assumes sorted by Date)
logging.info('Merging datasets...')
merged = pd.merge_asof(gold.sort_values('Date'), macro.sort_values('Date'), on='Date')
merged = pd.merge_asof(merged.sort_values('Date'), idx.sort_values('Date'), on='Date')
if not extra.empty:
    merged = pd.merge_asof(merged.sort_values('Date'), extra.sort_values('Date'), on='Date')

# Basic features already in gold: Close, Open, High, Low
merged['Return'] = merged['Close'].pct_change()
merged['LogReturn'] = np.log(merged['Close'] / merged['Close'].shift(1))

# Technical indicators (ta)
merged['MA5'] = merged['Close'].rolling(5).mean()
merged['MA10'] = merged['Close'].rolling(10).mean()
merged['MA20'] = merged['Close'].rolling(20).mean()
merged['Volatility5'] = merged['Return'].rolling(5).std()
merged['Volatility10'] = merged['Return'].rolling(10).std()
merged['Volatility20'] = merged['Return'].rolling(20).std()

# Using ta library for RSI and MACD
merged['RSI'] = ta.momentum.rsi(merged['Close'], window=14)
macd = ta.trend.MACD(merged['Close'])
merged['MACD'] = macd.macd()
merged['BB_high'] = ta.volatility.BollingerBands(merged['Close']).bollinger_hband()
merged['BB_low'] = ta.volatility.BollingerBands(merged['Close']).bollinger_lband()

# Lag features
lags = [1,3,5]
for l in lags:
    merged[f'Close_lag_{l}'] = merged['Close'].shift(l)
    merged[f'Return_lag_{l}'] = merged['Return'].shift(l)

# Rolling stats for macro/indices
for col in ['DXY','VIX','SP500']:
    if col in merged.columns:
        merged[f'{col}_MA5'] = merged[col].rolling(5).mean()
        merged[f'{col}_Vol5'] = merged[col].rolling(5).std()

# Ratio features
if 'DXY' in merged.columns:
    merged['Gold_vs_DXY'] = merged['Close'] / merged['DXY']
if 'SP500' in merged.columns:
    merged['Gold_vs_SP500'] = merged['Close'] / merged['SP500']
if 'SI' in merged.columns:
    merged['Gold_vs_Silver'] = merged['Close'] / merged['SI']

# Time features
merged['Year'] = merged['Date'].dt.year
merged['Month'] = merged['Date'].dt.month
merged['DayOfYear'] = merged['Date'].dt.dayofyear
merged['Weekday'] = merged['Date'].dt.weekday
merged['Month_sin'] = np.sin(2 * np.pi * merged['Month'] / 12)
merged['Month_cos'] = np.cos(2 * np.pi * merged['Month'] / 12)

# Drop rows with NaN created by rolling calculations
merged = merged.dropna(subset=[cfg['pipeline']['drop_initial_nan_from']])

# Target creation (next-day return)
h = cfg['pipeline'].get('target_horizon_days', 1)
merged['Target'] = merged['Close'].shift(-h) / merged['Close'] - 1
merged['Target_bin'] = (merged['Target'] > 0).astype(int)

# Keep columns of interest (example)
keep_cols = ['Date','Close','Open','High','Low','Return','LogReturn','MA5','MA10','MA20',
             'Volatility5','Volatility10','Volatility20','DXY','CPI','FEDFUNDS','CRUDE_OIL',
             'VIX','US10Y','M2','SP500','NASDAQ','DJIA','GDX','USO','SI','PL','PA','HG',
             'Gold_vs_DXY','Gold_vs_SP500','Gold_vs_Silver','RSI','MACD','BB_high','BB_low']

keep_cols = [c for c in keep_cols if c in merged.columns]
final = merged[keep_cols + ['Target','Target_bin']]

# Scaling numeric columns for downstream models (save scaler separately in practice)
numeric_cols = final.select_dtypes(include=[np.number]).columns.tolist()
scaler = StandardScaler()
final[numeric_cols] = scaler.fit_transform(final[numeric_cols])

final.to_parquet(OUTPUT, index=False)
logging.info(f'Final enhanced dataset saved to {OUTPUT} with shape {final.shape}')