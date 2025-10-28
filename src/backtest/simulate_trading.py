"""
simulate_trading_v4_regime_aware.py
----------------------------------
Advanced simulated trading with market regime adaptation.

✓ Loads latest ensemble model (stacking/voting)
✓ Applies same feature logic from training
✓ Detects market regimes dynamically (bull / bear / neutral)
✓ Regime-aware signal weighting and stop-loss control
✓ Signal filtering with probability thresholds
✓ Full backtesting metrics and equity curve visualization

Author: Lim Jia Xuan (v4)
"""

import os
import sys
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

# === Path fix for imports ===
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../model")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../utils")))

# === Try import feature function from training ===
try:
    from train_xgb import add_technical_features
except Exception:
    try:
        from train_gold_model import add_technical_features
    except Exception:
        add_technical_features = None
        print("⚠️ Could not import add_technical_features; using raw columns.")

# === Try import market regime detector ===
try:
    from market_regime_detector_custom import MarketRegimeDetector
except Exception:
    MarketRegimeDetector = None
    print("⚠️ Could not import MarketRegimeDetector; continuing without it.")

# === CONFIG ===
MODEL_DIR = "models"
DATA_PATH = "data/final/final_dataset_daily.csv"
INITIAL_CAPITAL = 10000.0
SAVE_DIR = "backtest_results"
os.makedirs(SAVE_DIR, exist_ok=True)
PLOT = True


# === HELPERS ===
def load_latest_model(model_dir: str):
    model_files = sorted(
        [f for f in os.listdir(model_dir) if f.endswith(".pkl")],
        key=lambda x: os.path.getmtime(os.path.join(model_dir, x)),
        reverse=True,
    )
    for f in model_files:
        if "stacking" in f or "voting" in f:
            print(f"✅ Loaded model: {f}")
            with open(os.path.join(model_dir, f), "rb") as fp:
                return pickle.load(fp)
    raise FileNotFoundError("No ensemble model (.pkl) found in /models")


def compute_metrics(df):
    total_profit = df["equity"].iloc[-1] - INITIAL_CAPITAL
    total_return = (df["equity"].iloc[-1] / INITIAL_CAPITAL - 1) * 100
    sharpe = (
        np.sqrt(252) * df["strategy_ret"].mean() / df["strategy_ret"].std()
        if df["strategy_ret"].std() != 0
        else 0.0
    )
    win_rate = (df["strategy_ret"] > 0).mean() * 100
    max_dd = (df["equity"] / df["equity"].cummax() - 1).min() * 100
    return {
        "final_equity": df["equity"].iloc[-1],
        "total_profit": total_profit,
        "total_return_%": total_return,
        "sharpe_ratio": sharpe,
        "win_rate_%": win_rate,
        "max_drawdown_%": max_dd,
    }


def visualize_equity(df, save_path):
    plt.figure(figsize=(10, 5))
    plt.plot(df["Date"], df["equity"], color="gold", label="Equity Curve")
    plt.title("Simulated Equity Curve (Regime-aware)", fontsize=14)
    plt.xlabel("Date")
    plt.ylabel("Equity ($)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"📈 Saved equity chart → {save_path}")


# === MAIN ===
def simulate_trading():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    if "Date" not in df.columns:
        df.reset_index(inplace=True)
        df.rename(columns={"index": "Date"}, inplace=True)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date", "Close"]).reset_index(drop=True)
    df["Return"] = df["Close"].pct_change()
    df = df.dropna().reset_index(drop=True)

    # === Add Features ===
    if add_technical_features is not None:
        df = add_technical_features(df)
    else:
        print("⚠️ No feature generator found — using raw columns only.")

    # === Add Market Regime ===
    if MarketRegimeDetector is not None:
        try:
            detector = MarketRegimeDetector(ma_fast=20, ma_slow=50)
            regime_df = detector.detect_regime(df)
            df = pd.merge(df, regime_df[["Date", "regime"]], on="Date", how="left")
            print("✅ Market regime added.")
        except Exception as e:
            print(f"⚠️ Regime detection failed: {e}")
            df["regime"] = "neutral"
    else:
        df["regime"] = "neutral"

    # === Macro Features (defensive) ===
    macro_cols = ["DXY", "CPI", "FEDFUNDS", "CRUDE_OIL", "VIX", "US10Y", "M2", "SP500", "NASDAQ", "DJIA"]
    for col in macro_cols:
        if col in df.columns:
            df[f"{col}_change"] = df[col].pct_change()
            df[f"{col}_lag_3"] = df[col].shift(3)
            df[f"{col}_lag_5"] = df[col].shift(5)

    # === Load model ===
    model = load_latest_model(MODEL_DIR)

    # === Feature alignment ===
    exclude = {"Date", "target_bin", "target_multi", "target_ret", "future_ret", "regime"}
    feature_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    X = df[feature_cols].copy().fillna(df[feature_cols].median())

    model_feats = getattr(model, "feature_names_in_", None)
    if model_feats is not None:
        missing = [f for f in model_feats if f not in X.columns]
        if missing:
            print(f"⚠️ Adding missing columns (set=0): {missing}")
            for f in missing:
                X[f] = 0
        X = X[model_feats]

    # === Predict signals ===
    preds = model.predict(X)

    df["buy_prob"] = np.nan
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)
        buy_prob = probs[:, -1] if probs.shape[1] > 1 else probs[:, 0]
        df["buy_prob"] = buy_prob
        THRESHOLD_BUY, THRESHOLD_SELL = 0.6, 0.4
        filtered_signal = np.zeros(len(df))
        filtered_signal = np.where(buy_prob > THRESHOLD_BUY, 1, filtered_signal)
        filtered_signal = np.where(buy_prob < THRESHOLD_SELL, -1, filtered_signal)
        df["signal_filtered"] = filtered_signal
        print(f"⚙️ Signal filtering applied (buy>{THRESHOLD_BUY}, sell<{THRESHOLD_SELL})")
    else:
        df["signal_filtered"] = np.nan
        print("ℹ️ Model has no predict_proba; skipping filtering.")

    # === Final Signal ===
    if np.unique(preds).max() > 1:
        base_signal = np.where(preds == 2, 1, np.where(preds == 0, -1, 0))
    else:
        base_signal = np.where(preds == 1, 1, -1)

    combined = base_signal.copy()
    if "signal_filtered" in df.columns:
        filt = df["signal_filtered"].fillna(0).astype(int).values
        mask = filt != 0
        combined[mask] = filt[mask]
    df["signal"] = combined

    # === Compute basic returns ===
    df["strategy_ret"] = df["signal"].shift(1).fillna(0) * df["Return"]

    # === Regime-aware weighting ===
    weights = {"bull": 1.0, "neutral": 0.4, "bear": -0.5}
    df["position_weight"] = df["regime"].map(weights).fillna(0.4)
    df["weighted_ret"] = df["strategy_ret"] * df["position_weight"]

    # === Regime-aware Stop-loss ===
    STOP_LOSS_LEVELS = {"bull": -0.05, "neutral": -0.02, "bear": -0.015}
    DRAWDOWN_LIMIT = -0.15

    equity = [INITIAL_CAPITAL]
    current_equity = INITIAL_CAPITAL
    triggered_stop = False
    triggered_dd = False

    for i in range(1, len(df)):
        ret = df.loc[i, "weighted_ret"]
        current_equity *= (1 + ret)
        equity.append(current_equity)

        regime = df.loc[i, "regime"]
        stop_loss = STOP_LOSS_LEVELS.get(regime, -0.03)

        if ret < stop_loss and not triggered_stop:
            print(f"🛑 Stop loss ({stop_loss:.1%}) triggered at {df.loc[i, 'Date'].date()} ({regime})")
            df.loc[i:, "signal"] = 0
            df.loc[i:, "weighted_ret"] = 0
            triggered_stop = True
            equity.extend([current_equity] * (len(df) - len(equity)))
            break

        max_eq = max(equity)
        drawdown = (current_equity / max_eq) - 1
        if drawdown < DRAWDOWN_LIMIT and not triggered_dd:
            print(f"⚠️ Drawdown limit reached ({drawdown:.1%}) at {df.loc[i, 'Date'].date()} → Exit.")
            df.loc[i:, "signal"] = 0
            df.loc[i:, "weighted_ret"] = 0
            triggered_dd = True
            equity.extend([current_equity] * (len(df) - len(equity)))
            break

    if len(equity) < len(df):
        equity.extend([equity[-1]] * (len(df) - len(equity)))

    df["equity"] = equity[:len(df)]
    df["strategy_ret"] = df["weighted_ret"]

    # === Save results ===
    metrics = compute_metrics(df)
    print("\n===== Trading Simulation Summary =====")
    for k, v in metrics.items():
        print(f"{k:<20s}: {v:.3f}" if isinstance(v, float) else f"{k:<20s}: {v}")
    print("======================================")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(SAVE_DIR, f"backtest_regime_{timestamp}.csv")
    df.to_csv(csv_path, index=False)
    print(f"💾 Saved trade log: {csv_path}")

    if PLOT:
        chart_path = os.path.join(SAVE_DIR, f"equity_curve_regime_{timestamp}.png")
        visualize_equity(df, chart_path)


if __name__ == "__main__":
    simulate_trading()
