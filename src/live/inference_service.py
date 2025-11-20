#!/usr/bin/env python3
# src/live/inference_service.py

import os
import sys
import time
import pickle
import json
import pandas as pd
import numpy as np
import warnings
from datetime import datetime, timezone

# Suppress warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ====== CONFIG PATHS ======
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(ROOT, "data", "final")
MODELS_DIR = os.path.join(ROOT, "models")
LATEST_CSV = os.path.join(DATA_DIR, "latest_hour_features.csv")

# ====== STRATEGY CONFIG (MATCHING v6.6 STABLE) ======
INITIAL_CAPITAL = 10000.0
BASE_RISK_PCT = 0.02        # 2% Risk
ATR_SL_MULT = 1.5           # SL = 1.5x ATR
ATR_TP_MULT = 4.0           # TP = 4.0x ATR
MAX_POS_FRAC = 0.5          # Max 50% Equity
VOL_WINDOW = 24
VOL_THRESHOLD_PCTL = 0.995  # 99.5% Volatility Cutoff
REGIME_WEIGHTS = {"bull": 1.2, "neutral": 1.0, "bear": 0.9}
ENABLE_TRAILING = False     # Default Off
TRAILING_ATR_MULT = 1.2

# ====== IMPORTS ======
sys.path.append(os.path.join(ROOT, "src"))
from utils.market_regime_detector import MarketRegimeDetector

# ====== STATE MANAGEMENT ======
PRED_JSON = os.path.join(DATA_DIR, "latest_prediction.json")
PRED_LOG = os.path.join(DATA_DIR, "prediction_log.csv")
TRADE_LOG = os.path.join(DATA_DIR, "trade_log.csv")
PORTFOLIO_STATE = os.path.join(DATA_DIR, "portfolio_state.json")

def load_portfolio():
    default = {
        "balance": INITIAL_CAPITAL, "position": 0.0, "entry_price": 0.0, 
        "sl": 0.0, "tp": 0.0, "trail_stop": 0.0, "equity": INITIAL_CAPITAL
    }
    if not os.path.exists(PORTFOLIO_STATE): return default
    try:
        with open(PORTFOLIO_STATE, "r") as f:
            data = f.read().strip()
            if not data: return default
            loaded = json.loads(data)
            # Robustness: Ensure all keys exist
            for k, v in default.items():
                if k not in loaded: loaded[k] = v
            return loaded
    except:
        return default

def save_portfolio(p):
    with open(PORTFOLIO_STATE, "w") as f: json.dump(p, f)

def log_trade(date, side, price, size, pnl=0, reason=""):
    exists = os.path.exists(TRADE_LOG)
    with open(TRADE_LOG, "a") as f:
        if not exists: f.write("Date,Side,Price,Size,PnL,Reason,Balance\n")
        f.write(f"{date},{side},{price},{size:.4f},{pnl:.2f},{reason},{portfolio['balance']:.2f}\n")

# ====== MODEL LOADING ======
try:
    LATEST_RUN = sorted([d for d in os.listdir(MODELS_DIR) if d.startswith("run_")])[-1]
    pkl_name = f"ensemble_calibrated_{LATEST_RUN.split('_')[1]}_{LATEST_RUN.split('_')[2]}.pkl"
    MODEL_PATH = os.path.join(MODELS_DIR, LATEST_RUN, pkl_name)
    print(f"📂 Model: {MODEL_PATH}")
    with open(MODEL_PATH, "rb") as f: model_meta = pickle.load(f)
except Exception as e:
    print(f"❌ Model Error: {e}")
    sys.exit(1)

calib_model = model_meta["calibrated_model"]
feature_cols = model_meta["feature_cols"]
optimal_threshold = model_meta.get("threshold", 0.5)

portfolio = load_portfolio()
last_processed_time = None

print(f"🚀 ProfitBoost v6.6 Live Engine Started.")
print(f"⚙️  Risk: {BASE_RISK_PCT*100}% | SL: {ATR_SL_MULT}x ATR | VolFilter: {VOL_THRESHOLD_PCTL*100}%")

# ====== MAIN LOOP ======
while True:
    try:
        if not os.path.exists(LATEST_CSV):
            time.sleep(5)
            continue

        # 1. Load Data
        try:
            df = pd.read_csv(LATEST_CSV)
        except:
            time.sleep(1)
            continue

        # Standardize Columns
        rename_map = {"GOLD_Close": "Close", "GOLD_Open": "Open", "GOLD_High": "High", "GOLD_Low": "Low"}
        for gold_col, std_col in rename_map.items():
            if gold_col in df.columns: df[std_col] = df[gold_col]

        # Recalculate Live Indicators for Strategy
        price_col = "GOLD_Close" if "GOLD_Close" in df.columns else "Close"
        df["SMA_20_Live"] = df[price_col].rolling(20).mean()
        df["SMA_60_Live"] = df[price_col].rolling(60).mean() 
        
        # Regime Detection
        detector = MarketRegimeDetector(ma_fast=20, ma_slow=50)
        regime_df = detector.detect_regime(df)
        
        latest_row = df.iloc[-1]
        latest_time = latest_row["Date"]
        
        if latest_time == last_processed_time:
            time.sleep(5)
            continue
            
        last_processed_time = latest_time
        print(f"\n🔎 Analyzing Candle: {latest_time}")

        # 2. Prepare Model Input
        input_df = pd.DataFrame(0.0, index=[0], columns=feature_cols)
        for col in feature_cols:
            if col in latest_row:
                 val = latest_row[col]
                 if pd.notna(val) and not np.isinf(val):
                     input_df.loc[0, col] = float(val)
                 else:
                     if len(df) > 1:
                         prev_val = df.iloc[-2].get(col, 0.0)
                         if pd.notna(prev_val): input_df.loc[0, col] = float(prev_val)
        input_df = input_df.fillna(0.0)
        
        # 3. Predict
        prob = calib_model.predict_proba(input_df.values)[0, 1]
        model_signal = 1 if prob > optimal_threshold else 0

        # 4. Strategy Conditions
        price = float(latest_row[price_col])
        atr = float(latest_row.get("ATR_14", 2.0))
        vol_now = float(latest_row.get("Volatility_20", 0.0))
        
        # Momentum Check
        sma20 = df["SMA_20_Live"].iloc[-1]
        sma60 = df["SMA_60_Live"].iloc[-1]
        if pd.isna(sma20) or pd.isna(sma60): momentum_ok = True
        else: momentum_ok = sma20 > sma60
        
        # Volatility Check
        vol_cutoff = df["Volatility_20"].quantile(VOL_THRESHOLD_PCTL)
        if pd.isna(vol_cutoff): vol_cutoff = 100.0
        vol_ok = vol_now <= vol_cutoff
        
        # Regime Check
        current_regime = regime_df["regime"].iloc[-1] if "regime" in regime_df.columns else "neutral"
        regime_weight = REGIME_WEIGHTS.get(current_regime, 1.0)

        # Final Signal
        want_long = (model_signal == 1) and momentum_ok and vol_ok
        
        # Update Status (UTC)
        status = {
            "Date": str(latest_time), 
            "probability": round(prob, 4), 
            "threshold": round(optimal_threshold, 4),
            "signal": 1 if want_long else 0, 
            "price": price, 
            "regime": current_regime,
            "vol_ok": bool(vol_ok), 
            "mom_ok": bool(momentum_ok),
            "last_updated": datetime.now(timezone.utc).isoformat()
        }
        with open(PRED_JSON, "w") as f: json.dump(status, f)
        pd.DataFrame([status]).to_csv(PRED_LOG, mode='a', header=not os.path.exists(PRED_LOG), index=False)

        # ====== 5. EXECUTION ENGINE ======
        pos = portfolio["position"]
        bal = portfolio["balance"]

        # --- Exit Logic ---
        if pos > 0:
            if price <= portfolio["sl"]:
                pnl = (portfolio["sl"] - portfolio["entry_price"]) * pos
                portfolio["balance"] += (portfolio["sl"] * pos)
                portfolio["position"] = 0.0
                log_trade(latest_time, "SELL", portfolio["sl"], pos, pnl, "Stop Loss")
                print(f"🛑 SL Hit @ {portfolio['sl']:.2f}")
            
            elif ENABLE_TRAILING and price <= portfolio["trail_stop"]:
                pnl = (portfolio["trail_stop"] - portfolio["entry_price"]) * pos
                portfolio["balance"] += (portfolio["trail_stop"] * pos)
                portfolio["position"] = 0.0
                log_trade(latest_time, "SELL", portfolio["trail_stop"], pos, pnl, "Trailing Stop")
                print(f"🛑 Trail Hit @ {portfolio['trail_stop']:.2f}")

            elif not want_long: 
                revenue = price * pos
                pnl = (price - portfolio["entry_price"]) * pos
                portfolio["balance"] += revenue
                portfolio["position"] = 0.0
                log_trade(latest_time, "SELL", price, pos, pnl, "Strategy Exit")
                print(f"🔴 Strategy Exit @ {price:.2f} | PnL: {pnl:.2f}")
                
            elif ENABLE_TRAILING:
                new_trail = max(portfolio["trail_stop"], price - (atr * TRAILING_ATR_MULT))
                portfolio["trail_stop"] = new_trail

        # --- Entry Logic ---
        if want_long and pos == 0:
            median_vol = df["Volatility_20"].median()
            if pd.isna(median_vol): median_vol = vol_now
            
            vol_scale = np.clip((median_vol + 1e-9) / (vol_now + 1e-9), 0.5, 1.5)
            stop_dist = max(atr * ATR_SL_MULT, price * 0.0005)
            risk_amt = bal * BASE_RISK_PCT * regime_weight * vol_scale
            calc_units = risk_amt / stop_dist
            max_units = (bal * MAX_POS_FRAC) / price
            final_units = min(calc_units, max_units)
            
            if final_units > 0.0001:
                cost = final_units * price
                portfolio["balance"] -= cost
                portfolio["position"] = final_units
                portfolio["entry_price"] = price
                portfolio["sl"] = price - stop_dist
                portfolio["tp"] = price + (atr * ATR_TP_MULT)
                portfolio["trail_stop"] = price - (atr * TRAILING_ATR_MULT)
                
                reason = f"Conf:{prob:.2f}|Reg:{current_regime}|Vol:{vol_scale:.2f}"
                log_trade(latest_time, "BUY", price, final_units, 0, reason)
                print(f"🟢 BUY @ {price:.2f} | Size: {final_units:.4f} | {reason}")

        # --- Update Equity (FIXED) ---
        # Equity = Cash + Market Value of Holdings
        market_value = portfolio["position"] * price
        portfolio["equity"] = portfolio["balance"] + market_value
        
        save_portfolio(portfolio)
        
        print(f"📊 Equity: ${portfolio['equity']:.2f} | Prob: {prob:.1%} | Regime: {current_regime}")

    except Exception as e:
        print(f"⚠️ Error: {e}")
        time.sleep(5)
    
    time.sleep(5)