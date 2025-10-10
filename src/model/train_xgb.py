"""
train_xgb_v3.py
--------------------------------------------
Optimized XGBoost regression model for predicting
next-day gold price return (%), with advanced features
and time-series validation.

Author: Lim Jia Xuan
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor, plot_importance
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, accuracy_score

# ===============================
# 1️⃣ CONFIGURATION
# ===============================
DATA_PATH = "./cleaned_gold_dataset.csv"
OUTPUT_DIR = "outputs/figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===============================
# 2️⃣ LOAD DATA
# ===============================
print("📂 Loading dataset...")
df = pd.read_csv(DATA_PATH)

if "Date" in df.columns:
    df["Date"] = pd.to_datetime(df["Date"])

print(f"✅ Loaded dataset with shape: {df.shape}")
print(f"Date range: {df['Date'].min().date()} → {df['Date'].max().date()}")

# ===============================
# 3️⃣ BASIC CLEANING
# ===============================
macro_cols = ["CPI", "FEDFUNDS", "DXY", "CRUDE_OIL"]
for col in macro_cols:
    if col in df.columns:
        df[col] = df[col].ffill()  # 推荐写法 (取代 fillna(method='ffill'))

df = df.drop_duplicates(subset=["Date"]).reset_index(drop=True)

# ===============================
# 4️⃣ FEATURE ENGINEERING
# ===============================
# Target: next-day return
df["Target"] = df["Close"].shift(-1) / df["Close"] - 1

# Lag features
for lag in range(1, 6):
    df[f"Close_lag{lag}"] = df["Close"].shift(lag)
    df[f"Return_lag{lag}"] = df["Target"].shift(lag)

# Rolling stats (technical indicators)
df["MA5"] = df["Close"].rolling(5).mean()
df["MA10"] = df["Close"].rolling(10).mean()
df["Volatility5"] = df["Target"].rolling(5).std()
df["Momentum10"] = df["Close"] / df["Close"].shift(10) - 1
df["RSI14"] = 100 - (100 / (1 + (df["Target"].rolling(14).apply(lambda x: (x[x > 0].mean() / abs(x[x < 0].mean())) if len(x[x < 0]) > 0 else 0.0))))

# Drop missing
df = df.dropna().reset_index(drop=True)
print(f"✅ After feature engineering: {df.shape}")

# ===============================
# 5️⃣ DEFINE FEATURES
# ===============================
features = [
    "Open", "High", "Low", "Volume",
    "MA5", "MA10", "Volatility5", "Momentum10", "RSI14",
    "DXY", "CPI", "FEDFUNDS", "CRUDE_OIL", "^GSPC", "^IXIC",
    "Close_lag1", "Close_lag2", "Close_lag3", "Close_lag4", "Close_lag5",
    "Return_lag1", "Return_lag2", "Return_lag3", "Return_lag4", "Return_lag5"
]

X = df[features].copy()
y = df["Target"]

low_var_cols = [c for c in X.columns if X[c].std() < 1e-8]
if low_var_cols:
    X = X.drop(columns=low_var_cols)
    print(f"⚠️ Dropped low-variance columns: {low_var_cols}")

# ===============================
# 6️⃣ TIME SERIES CV + TUNING
# ===============================
print("\n🔍 Performing time-series CV tuning...")
tscv = TimeSeriesSplit(n_splits=5)

param_grid = {
    "max_depth": [6, 8, 10],
    "learning_rate": [0.01, 0.03, 0.05],
    "n_estimators": [500, 800],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
    "reg_lambda": [1, 2],
}

grid = GridSearchCV(
    XGBRegressor(random_state=42, objective="reg:squarederror"),
    param_grid,
    cv=tscv,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1,
    verbose=1
)

grid.fit(X, y)
best_model = grid.best_estimator_
print(f"\n✅ Best params: {grid.best_params_}")

# ===============================
# 7️⃣ FINAL TRAIN/TEST SPLIT
# ===============================
split_idx = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)

# ===============================
# 8️⃣ EVALUATION
# ===============================
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Direction accuracy (up/down)
actual_dir = np.sign(y_test)
pred_dir = np.sign(y_pred)
direction_acc = accuracy_score(actual_dir, pred_dir)

print("\n🎯 Final Evaluation:")
print(f"   RMSE: {rmse:.6f}")
print(f"   MAE:  {mae:.6f}")
print(f"   R²:   {r2:.4f}")
print(f"   Direction Accuracy: {direction_acc * 100:.2f}%")

# ===============================
# 9️⃣ VISUALIZATION
# ===============================
comparison = pd.DataFrame({
    "Date": df["Date"].iloc[split_idx:].values,
    "Actual_Return": y_test.values,
    "Predicted_Return": y_pred
})

plt.figure(figsize=(12, 6))
plt.plot(comparison["Date"], comparison["Actual_Return"], label="Actual", color="black", alpha=0.8)
plt.plot(comparison["Date"], comparison["Predicted_Return"], label="Predicted", color="orange", alpha=0.8)
plt.title("Gold Next-Day Return Prediction (Optimized XGBoost)")
plt.xlabel("Date")
plt.ylabel("Return")
plt.legend()
plt.tight_layout()
save_path = os.path.join(OUTPUT_DIR, "gold_return_pred_vs_actual_v3.png")
plt.savefig(save_path, dpi=300)
plt.show()
print(f"📈 Prediction plot saved to: {save_path}")

# ===============================
# 🔟 FEATURE IMPORTANCE
# ===============================
plt.figure(figsize=(10, 6))
plot_importance(best_model, max_num_features=15)
plt.title("Top Feature Importances")
plt.tight_layout()
plt.show()

print("\n🚀 Training & evaluation complete!")
