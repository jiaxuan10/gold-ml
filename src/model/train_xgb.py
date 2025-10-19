# train_xgb_detrend_enhanced_v2.py
"""
Enhanced regression + dynamic quantile classification pipeline for gold next-day / 5-day signal.

Improvements:
- Regression target (detrended_ret_next) instead of strict Q30-Q70 binary
- Shorter detrend window (63)
- Extended lag features
- Optional horizon (1-day / 5-day)
- Macro features lagged
- Dynamic threshold based on train quantiles
"""
import os
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, balanced_accuracy_score
)
import joblib
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from xgboost.sklearn import XGBRegressor
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score

RSEED = 42
np.random.seed(RSEED)

# ---------- CONFIG ----------
DATA_PATH = "data/final/final_dataset_daily.csv"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

TEST_DAYS = 365
LONG_W = 63           # short-term detrend
LAG_DAYS = [1,2,3,5,10,15,20,30]
HORIZON = 5           # 可改为1/5天
USE_REGRESSION = True
Q_LOW = 0.4
Q_HIGH = 0.6
TS_SPLITS = 5

MACRO_COLS = ['DXY','CPI','CRUDE_OIL','VIX','SP500','US10Y','M2']

# ---------- LOAD ----------
df = pd.read_csv(DATA_PATH)
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)
print(f"Loaded {len(df)} rows: {df['Date'].min().date()} -> {df['Date'].max().date()}")

df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
df = df.dropna(subset=['Close']).reset_index(drop=True)

# ---------- FEATURES ----------
df['log_price'] = np.log(df['Close'])
df['LogReturn'] = df['log_price'].diff()
df['Return'] = df['Close'].pct_change()
df['Diff_Close'] = df['Close'].diff()

# Macro features
if 'CPI' in df.columns:
    df['CPI'] = df['CPI'].fillna(method='ffill')
for col in MACRO_COLS:
    if col in df.columns:
        if col == 'CPI':
            df[f'{col}_chg'] = df[col].diff()
        else:
            df[f'{col}_chg'] = df[col].pct_change()
macro_derived = [c for c in df.columns if any(c.startswith(m) and c.endswith('_chg') for m in MACRO_COLS)]

# ---------- Detrend ----------
df['trend_long'] = df['log_price'].rolling(LONG_W, min_periods=int(LONG_W/4)).mean()
df['detrended'] = df['log_price'] - df['trend_long']
df['detrended_ret_next'] = df['detrended'].shift(-HORIZON) - df['detrended']

df = df.dropna(subset=['log_price','trend_long','detrended','detrended_ret_next']).reset_index(drop=True)
print(f"After detrend processing: {len(df)} rows remain")

# ---------- FEATURE ENGINEERING ----------
candidate = [
 "Open","High","Low","Close","log_price","LogReturn","Return","Diff_Close",
 "MA5","MA10","MA20",
 "Volatility5","Volatility10","Volatility20"
] + macro_derived + [
 "GDX","USO","SI","PL","PA","HG",
 "Gold_vs_DXY","Gold_vs_SP500","Gold_vs_Silver"
]

features = [c for c in candidate if c in df.columns]
df.loc[:, 'DayOfWeek'] = df['Date'].dt.weekday
features.append('DayOfWeek')

# Lagged log returns
for lag in LAG_DAYS:
    col = f'logret_lag_{lag}'
    df.loc[:, col] = df['LogReturn'].shift(lag)
    features.append(col)

# MA diff
if 'MA5' in df.columns and 'MA10' in df.columns:
    df.loc[:, 'MA5_MA10_diff'] = df['MA5'] - df['MA10']
    features.append('MA5_MA10_diff')

# detrended z-score
df.loc[:, 'detrended_z'] = (df['detrended'] - df['detrended'].rolling(LONG_W, min_periods=30).mean()) / (
    df['LogReturn'].rolling(LONG_W, min_periods=30).std().replace(0, np.nan)
)
features += ['detrended','detrended_z']

features = [f for f in pd.unique(features) if f in df.columns]
df = df.dropna(subset=features + ['detrended_ret_next']).reset_index(drop=True)
print(f"After feature construction: {len(df)} rows (features count: {len(features)})")

# ---------- SPLIT ----------
X = df[features].copy()
y_cont = df['detrended_ret_next'].copy()
dates = df['Date'].copy()

X_train = X.iloc[:-TEST_DAYS].reset_index(drop=True)
y_train = y_cont.iloc[:-TEST_DAYS].reset_index(drop=True)
X_test = X.iloc[-TEST_DAYS:].reset_index(drop=True)
y_test = y_cont.iloc[-TEST_DAYS:].reset_index(drop=True)
dates_test = dates.iloc[-TEST_DAYS:].reset_index(drop=True)

# ---------- SCALE ----------
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler_detrend_enhanced_v2.pkl"))

# ---------- XGB Regressor ----------
xgb_reg = xgb.XGBRegressor(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.03,
    subsample=0.6,
    colsample_bytree=1.0,
    random_state=RSEED,
    n_jobs=-1
)
xgb_reg.fit(X_train_s, y_train)
joblib.dump(xgb_reg, os.path.join(MODEL_DIR,"xgb_reg_detrend_enhanced_v2.pkl"))
print("Saved XGB regressor")

# ---------- Random Forest Regressor ----------
rf_reg = RandomForestRegressor(
    n_estimators=300,
    random_state=RSEED,
    n_jobs=-1
)
rf_reg.fit(X_train_s, y_train)
joblib.dump(rf_reg, os.path.join(MODEL_DIR,"rf_reg_detrend_enhanced_v2.pkl"))
print("Saved RF regressor")

# ---------- Optional Stacking ----------
do_stacking = True
if do_stacking:
    stack = StackingRegressor(
        estimators=[('xgb', xgb_reg), ('rf', rf_reg)],
        final_estimator=LinearRegression(),
        n_jobs=-1,
        passthrough=False
    )
    stack.fit(X_train_s, y_train)
    joblib.dump(stack, os.path.join(MODEL_DIR,"stack_reg_detrend_enhanced_v2.pkl"))
    print("Saved stacking regressor")

# ---------- Predictions & dynamic thresholds ----------
xgb_pred_test = xgb_reg.predict(X_test_s)
rf_pred_test = rf_reg.predict(X_test_s)
if do_stacking:
    stack_pred_test = stack.predict(X_test_s)

# Train quantiles as thresholds
q_low_val = y_train.quantile(Q_LOW)
q_high_val = y_train.quantile(Q_HIGH)

def reg2signal(pred, ql, qh):
    return np.where(pred <= ql, 0, np.where(pred >= qh, 1, 2))

pred_df = pd.DataFrame({
    'Date': dates_test,
    'True': y_test,
    'XGB_pred': xgb_pred_test,
    'XGB_signal': reg2signal(xgb_pred_test,q_low_val,q_high_val),
    'RF_pred': rf_pred_test,
    'RF_signal': reg2signal(rf_pred_test,q_low_val,q_high_val)
})
if do_stacking:
    pred_df['STACK_pred'] = stack_pred_test
    pred_df['STACK_signal'] = reg2signal(stack_pred_test,q_low_val,q_high_val)

pred_csv_path = os.path.join(MODEL_DIR,"predictions_detrend_enhanced_v2.csv")
pred_df.to_csv(pred_csv_path,index=False)
print("Saved predictions to", pred_csv_path)

print("\nDone.")

# ---------- Directional accuracy & confusion matrix ----------
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score

# 根据训练集 quantile 生成 True_signal
df_bin = pred_df[pred_df['True'].apply(lambda x: 0 if x<=q_low_val else (1 if x>=q_high_val else 2)) != 2].copy()
df_bin['True_signal'] = df_bin['True'].apply(lambda x: 0 if x<=q_low_val else 1)

# 过滤模型 signal 中非 0/1 的情况
for col in ['XGB_signal','RF_signal','STACK_signal']:
    if col in df_bin.columns:
        df_bin = df_bin[df_bin[col].isin([0,1])]

def print_metrics(name, true_labels, pred_labels):
    print(f"\n--- {name} ---")
    print("Accuracy:", accuracy_score(true_labels, pred_labels))
    print("Balanced Acc:", balanced_accuracy_score(true_labels, pred_labels))
    print("Precision:", precision_score(true_labels, pred_labels, average='binary', zero_division=0))
    print("Recall:", recall_score(true_labels, pred_labels, average='binary', zero_division=0))
    print("F1:", f1_score(true_labels, pred_labels, average='binary', zero_division=0))
    print("Confusion:\n", confusion_matrix(true_labels, pred_labels))

# XGB
print_metrics("XGB directional", df_bin['True_signal'], df_bin['XGB_signal'])
# RF
print_metrics("RF directional", df_bin['True_signal'], df_bin['RF_signal'])
# STACK
if do_stacking and 'STACK_signal' in df_bin.columns:
    print_metrics("STACK directional", df_bin['True_signal'], df_bin['STACK_signal'])