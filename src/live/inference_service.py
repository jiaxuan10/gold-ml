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

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ====== CONFIG ======
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(ROOT, "data", "final")
MODELS_DIR = os.path.join(ROOT, "models")
LATEST_CSV = os.path.join(DATA_DIR, "latest_hour_features.csv")
SENTIMENT_FILE = os.path.join(DATA_DIR, "current_sentiment.json")

# Strategy Params
INITIAL_CAPITAL = 10000.0
BASE_RISK_PCT = 0.02
ATR_SL_MULT = 1.5
ATR_TP_MULT = 4.0
MAX_POS_FRAC = 0.5
VOL_THRESHOLD_PCTL = 0.995
REGIME_WEIGHTS = {"bull": 1.2, "neutral": 1.0, "bear": 0.9}
ENABLE_TRAILING = False
TRAILING_ATR_MULT = 1.2

# State Files
PRED_JSON = os.path.join(DATA_DIR, "latest_prediction.json")
PRED_LOG = os.path.join(DATA_DIR, "prediction_log.csv")
TRADE_LOG = os.path.join(DATA_DIR, "trade_log.csv")
PORTFOLIO_STATE = os.path.join(DATA_DIR, "portfolio_state.json")

sys.path.append(os.path.join(ROOT, "src"))
from utils.market_regime_detector import MarketRegimeDetector

def load_portfolio():
    default = {
        "balance": INITIAL_CAPITAL, "position": 0.0, "entry_price": 0.0, 
        "sl": 0.0, "tp": 0.0, "trail_stop": 0.0, "equity": INITIAL_CAPITAL, "last_candle": None
    }
    if not os.path.exists(PORTFOLIO_STATE): return default
    try:
        with open(PORTFOLIO_STATE, "r") as f:
            return {**default, **json.load(f)}
    except: return default

def save_portfolio(p):
    with open(PORTFOLIO_STATE, "w") as f: json.dump(p, f)

def log_trade(date, side, price, size, pnl=0, reason=""):
    exists = os.path.exists(TRADE_LOG)
    with open(TRADE_LOG, "a") as f:
        if not exists: f.write("Date,Side,Price,Size,PnL,Reason,Balance\n")
        f.write(f"{date},{side},{price},{size:.4f},{pnl:.2f},{reason},{portfolio['balance']:.2f}\n")

def get_sentiment_modifier():
    if not os.path.exists(SENTIMENT_FILE): return 0.0, "NoNews"
    try:
        with open(SENTIMENT_FILE, "r") as f:
            data = json.load(f)
            score = data.get("sentiment_score", 0.0)
            
            # === ✅ 修正：数值改回 0.01，文字改回 BearNews ===
            
            # 1. 极好新闻 (> 0.2) -> 门槛降低 0.01
            if score > 0.2: 
                return -0.01, f"BullNews({score:.2f})"
            
            # 2. 稍好新闻 (> 0.05) -> 门槛降低 0.005
            if score > 0.05: 
                return -0.005, f"GoodNews({score:.2f})"
            
            # 3. 极差新闻 (< -0.2) -> 门槛升高 0.01
            if score < -0.2: 
                return 0.01, f"BearNews({score:.2f})"
            
            # 4. 稍差新闻 (< -0.05) -> 门槛升高 0.005
            if score < -0.05: 
                return 0.005, f"BadNews({score:.2f})"
            
            return 0.0, "NeutralNews"
    except: return 0.0, "Error"
# Load Model
try:
    LATEST_RUN = sorted([d for d in os.listdir(MODELS_DIR) if d.startswith("run_")])[-1]
    pkl_name = f"ensemble_calibrated_{LATEST_RUN.split('_')[1]}_{LATEST_RUN.split('_')[2]}.pkl"
    MODEL_PATH = os.path.join(MODELS_DIR, LATEST_RUN, pkl_name)
    print(f"📂 Loading Model: {MODEL_PATH}")
    with open(MODEL_PATH, "rb") as f: model_meta = pickle.load(f)
except Exception as e:
    print(f"❌ Model Error: {e}")
    sys.exit(1)

calib_model = model_meta["calibrated_model"]
# 🔥 重要：这会自动获取模型训练时用的特征列表 (不含 Open/Close)
feature_cols = model_meta["feature_cols"] 
# base_threshold = model_meta.get("threshold", 0.5)
# base_threshold = model_meta.get("threshold", 0.5) # Comment out the original
base_threshold = 0.50 # <--- FORCE IT TO 50% (Coin flip odds)
print(f"⚠️ FYP MODE: Overriding threshold to {base_threshold} to force trades.")

portfolio = load_portfolio()
print(f"🚀 Live Engine Started. Risk: {BASE_RISK_PCT*100}% | Threshold: {base_threshold}")

while True:
    try:
        portfolio = load_portfolio() 
        
        if not os.path.exists(LATEST_CSV): time.sleep(5); continue
        try: df = pd.read_csv(LATEST_CSV)
        except: time.sleep(1); continue

        rename_map = {"GOLD_Close": "Close", "GOLD_Open": "Open", "GOLD_High": "High", "GOLD_Low": "Low"}
        for k, v in rename_map.items():
            if k in df.columns: df[v] = df[k]
        
        latest_row = df.iloc[-1]
        latest_time = latest_row["Date"]

        # 🔥 周末休市检查 (UTC时间 周六/周日)
        # 这能防止系统在周末对着周五的旧数据空转
        # fetcher 已经切掉了周末数据，所以如果读到了周五的数据，这里会一直等到周一新数据来才动
        if portfolio.get("last_candle") == str(latest_time):
            time.sleep(10); continue
            
        print(f"\n🔎 Analyzing: {latest_time}")
        price = float(latest_row.get("Close", 0))
        if price <= 0: continue

        # Indicators
        df["SMA_20_Live"] = df["Close"].rolling(20).mean()
        df["SMA_60_Live"] = df["Close"].rolling(60).mean()
        detector = MarketRegimeDetector(ma_fast=20, ma_slow=50)
        regime_df = detector.detect_regime(df)
        
        # 🔥 Prepare Input for Model (Strict Feature Matching)
        # 即使 CSV 里有 Close，我们只提取 feature_cols 里的列
        input_df = pd.DataFrame(0.0, index=[0], columns=feature_cols)
        for col in feature_cols:
            if col in latest_row:
                input_df.loc[0, col] = float(latest_row[col])
            # 如果某些特征缺失 (例如 Hour_Sin)，用 0 填充防止报错
        
        input_df = input_df.fillna(0.0)
        
        # Predict
        raw_prob = calib_model.predict_proba(input_df.values)[0, 1]

        # Sentiment Adj
        sent_adj, sent_reason = get_sentiment_modifier()
        final_threshold = max(0.3, min(0.9, base_threshold + sent_adj))
        
        model_signal = 1 if raw_prob > final_threshold else 0

        # Strategy Filters
        atr = float(latest_row.get("ATR_14", 2.0))
        vol_now = float(latest_row.get("Volatility_20", 0.0))
        sma20 = df["SMA_20_Live"].iloc[-1]
        sma60 = df["SMA_60_Live"].iloc[-1]
        momentum_ok = sma20 > sma60 if (pd.notna(sma20) and pd.notna(sma60)) else True
        vol_cutoff = df["Volatility_20"].quantile(VOL_THRESHOLD_PCTL)
        vol_ok = vol_now <= (vol_cutoff if pd.notna(vol_cutoff) else 100)
        curr_regime = regime_df["regime"].iloc[-1] if "regime" in regime_df.columns else "neutral"
        
        # want_long = (model_signal == 1) and momentum_ok and vol_ok
        want_long = (model_signal == 1)
        
        # Save Status
        status = {
            "Date": str(latest_time), 
            "probability": round(raw_prob, 4), 
            "base_threshold": round(base_threshold, 4),
            "final_threshold": round(final_threshold, 4),
            "sentiment_impact": sent_reason,
            "signal": 1 if want_long else 0, 
            "price": price, 
            "regime": curr_regime,
            "last_updated": datetime.now(timezone.utc).isoformat()
        }
        with open(PRED_JSON, "w") as f: json.dump(status, f)
        pd.DataFrame([status]).to_csv(PRED_LOG, mode='a', header=not os.path.exists(PRED_LOG), index=False)

        # Execution
        pos = portfolio["position"]
        bal = portfolio["balance"]
        action_taken = False 

        # EXIT
        if pos > 0:
            # 1. Check Stop Loss
            if price <= portfolio["sl"]:
                reason = "Stop Loss"
                pnl = (portfolio["sl"] - portfolio["entry_price"]) * pos
                portfolio["balance"] += portfolio["sl"] * pos
                portfolio["position"] = 0.0
                log_trade(latest_time, "SELL", portfolio["sl"], pos, pnl, reason)
                print(f"🛑 {reason}")
                action_taken = True
            
            # 2. 🆕 ADDED: Check Take Profit
            elif price >= portfolio["tp"]:
                reason = "Take Profit"
                pnl = (price - portfolio["entry_price"]) * pos
                portfolio["balance"] += price * pos
                portfolio["position"] = 0.0
                log_trade(latest_time, "SELL", price, pos, pnl, reason)
                print(f"💰 {reason} Hit! PnL: ${pnl:.2f}")
                action_taken = True

            # 3. Check AI Signal (Exit if confidence drops below 50%)
            elif not want_long:
                reason = f"Exit(Prob {raw_prob:.2f} < {final_threshold:.2f})"
                pnl = (price - portfolio["entry_price"]) * pos
                portfolio["balance"] += price * pos
                portfolio["position"] = 0.0
                log_trade(latest_time, "SELL", price, pos, pnl, reason)
                print(f"🔴 {reason}")
                action_taken = True

        # ENTRY
        if not action_taken and want_long and pos == 0:
            regime_weight = REGIME_WEIGHTS.get(curr_regime, 1.0)
            stop_dist = max(atr * ATR_SL_MULT, price * 0.0005)
            risk_amt = bal * BASE_RISK_PCT * regime_weight
            units = min(risk_amt / stop_dist, (bal * MAX_POS_FRAC) / price)
            
            if units > 0.0001:
                portfolio["balance"] -= units * price
                portfolio["position"] = units
                portfolio["entry_price"] = price
                portfolio["sl"] = price - stop_dist
                portfolio["tp"] = price + (atr * ATR_TP_MULT)
                reason = f"Buy(Prob {raw_prob:.2f} > {final_threshold:.2f}) | {sent_reason}"
                log_trade(latest_time, "BUY", price, units, 0, reason)
                print(f"🟢 {reason}")
                action_taken = True

        if not action_taken:
            status_side = "HOLD" if pos > 0 else "WAIT"
            reason = f"{sent_reason} | Prob:{raw_prob:.2f} vs Thr:{final_threshold:.2f}"
            log_trade(latest_time, status_side, price, 0.0, 0.0, reason)
            print(f"⏳ {status_side} | {reason}")

        portfolio["equity"] = portfolio["balance"] + (portfolio["position"] * price)
        portfolio["last_candle"] = str(latest_time)
        save_portfolio(portfolio)
        print(f"📊 Equity: ${portfolio['equity']:.2f}")

    except Exception as e:
        print(f"⚠️ Error: {e}")
        time.sleep(5)
    
    time.sleep(5)