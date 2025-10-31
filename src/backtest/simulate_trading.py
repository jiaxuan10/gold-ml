#!/usr/bin/env python3
"""
simulate_trading_v6_2.py
------------------------
Profit-Boost v6.2 - Dual-direction (Long+Short) backtester with
drawdown-based risk adaptation.

Features:
 - Long + Short entries (symmetric logic)
 - Confidence-weighted position sizing
 - ATR-based adaptive stop-loss & take-profit
 - Volatility filter
 - Regime-weighted sizing
 - Short-specific risk multiplier (safer shorts)
 - Drawdown-based dynamic risk scaling (reduces RISK_PER_TRADE when in large drawdown)
 - Saves trades, equity CSV, equity PNG into backtest_results/
Author: adapted for Lim Jia Xuan (Profit-Boost v6.2)
"""

import os
import sys
import pickle
from datetime import datetime
from typing import Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# ---------------- CONFIG ----------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODEL_PATH = os.path.join(ROOT, "models", "run_20251029_231844", "ensemble_calibrated_20251029_231844.pkl")
DATA_PATH = os.path.join(ROOT, "data", "final", "final_dataset_hourly.csv")
SAVE_DIR = os.path.join(ROOT, "backtest_results")
os.makedirs(SAVE_DIR, exist_ok=True)

SEED = 42
np.random.seed(SEED)

INITIAL_CAPITAL = 10000.0

# === 更激进的风险/头寸设置（v6.3 aggressive-style） ===
# Risk sizing (base) — 提升每笔风险敞口
BASE_RISK_PER_TRADE_FRAC = 0.25   # 从 0.05 提升到 0.08（8% 的资金风险）
MAX_POSITION_FRAC = 1.0          # 单笔头寸最大占比从 0.50 提升至 0.75

# Short-specific multiplier -> 与多头同等或略保守
SHORT_RISK_MULT = 1.0             # 不再降低空头仓位（与多头等同）

# ATR / stop / tp 调整：放宽止损、拉远止盈以提升RR
ATR_WINDOW = 14
ATR_MULTIPLIER_SL = 1.2    # 从 1.8 下调到 1.6（略宽止损，减少被震出）
ATR_MULTIPLIER_TP = 6.0     # 从 3.2 提高到 4.0（放大获利空间）
MIN_STOP_PCT = 0.001        # 最低止损（保留）

# Volatility filter：放宽高波动过滤，允许更多高波动交易
VOL_WINDOW = 24
VOL_THRESHOLD_PCTL = 0.995   # 从 0.90 放宽到 0.95

# Model thresholds & fallback：放宽阈值以提升交易频率
BUY_PROB_DEFAULT = 0.50
MIN_CONFIDENCE = 0.10       # 从 0.35 放宽到 0.25（允许较低置信也进场）
MIN_PROBA_FOR_SIGNAL = 0.47  # 下限从 0.50 调为 0.48，略放宽

# Costs
COMMISSION = 0.0005  # proportional
SLIPPAGE = 0.0005

REGIME_WEIGHTS = {"bull": 1.2, "neutral": 1.0, "bear": 0.9} # 略调 regime 权重：中性提高

# Enable shorting
ENABLE_SHORTS = True

# Drawdown-based dynamic risk scaling -> 保持阈值，但我们将在回测中禁用（固定 dd_factor=1.0）
DD_SCALE_THRESHOLDS = [0.20, 0.35, 0.50]
DD_SCALE_FACTORS = [0.95, 0.8, 0.6]
# ===================================================


# ---------------- Utilities ----------------
def load_latest_model(model_dir: str):
    pkl_files = sorted(
        [f for f in os.listdir(model_dir) if f.endswith(".pkl")],
        key=lambda f: os.path.getmtime(os.path.join(model_dir, f)),
        reverse=True,
    )
    if not pkl_files:
        raise FileNotFoundError(f"No .pkl model files found in {model_dir}")
    for f in pkl_files:
        path = os.path.join(model_dir, f)
        try:
            with open(path, "rb") as fd:
                obj = pickle.load(fd)
            print(f"✅ Loaded model file: {f}")
            if isinstance(obj, dict):
                return obj
            else:
                return {"model": obj}
        except Exception as e:
            print(f"⚠️ Failed to load {f}: {e}")
    raise RuntimeError("No valid model found.")

def safe_feature_align(X: pd.DataFrame, model_meta: dict) -> pd.DataFrame:
    Xc = X.copy()
    feat_required = None
    if isinstance(model_meta, dict) and model_meta.get("feature_cols"):
        feat_required = list(model_meta["feature_cols"])
    else:
        model = None
        if isinstance(model_meta, dict):
            model = model_meta.get("calibrated_model") or model_meta.get("raw_ensemble") or model_meta.get("model")
        else:
            model = model_meta
        if model is not None and hasattr(model, "feature_names_in_"):
            feat_required = list(model.feature_names_in_)
    if feat_required is not None:
        missing = [f for f in feat_required if f not in Xc.columns]
        if missing:
            print(f"⚠️ Missing features (filling 0): {missing[:10]}{'...' if len(missing)>10 else ''}")
            for m in missing:
                Xc[m] = 0.0
        Xc = Xc[feat_required]
    Xc = Xc.replace([np.inf, -np.inf], np.nan)
    Xc = Xc.fillna(Xc.median().fillna(0.0))
    Xc = Xc.clip(lower=-1e6, upper=1e6)
    return Xc

def model_predict_proba(model_meta, X: pd.DataFrame) -> Optional[np.ndarray]:
    """Try to get probabilities from saved metadata dict or raw model."""
    if isinstance(model_meta, dict):
        for key in ("calibrated_model", "raw_ensemble", "model"):
            m = model_meta.get(key)
            if m is None:
                continue
            try:
                return m.predict_proba(X)
            except Exception:
                # try averaging estimator probs if ensemble-like
                try:
                    ests = getattr(m, "estimators_", getattr(m, "estimators", []))
                    probs = []
                    for e in ests:
                        try:
                            p = e.predict_proba(X)
                            probs.append(p)
                        except Exception:
                            continue
                    if probs:
                        return sum(probs) / len(probs)
                except Exception:
                    pass
    else:
        try:
            return model_meta.predict_proba(X)
        except Exception:
            try:
                ests = getattr(model_meta, "estimators_", [])
                probs = []
                for e in ests:
                    try:
                        p = e.predict_proba(X)
                        probs.append(p)
                    except Exception:
                        continue
                if probs:
                    return sum(probs) / len(probs)
            except Exception:
                pass
    return None

# ATR helper
def compute_atr(df: pd.DataFrame, window: int = ATR_WINDOW, col_high="High", col_low="Low", col_close="Close"):
    h = df[col_high]
    l = df[col_low]
    c = df[col_close].shift(1)
    hl = h - l
    hc = (h - c).abs()
    lc = (l - c).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    atr = tr.rolling(window, min_periods=3).mean()
    return atr.bfill().fillna(0.0)

# backtest metrics (same style)
def compute_backtest_metrics(equity_series: pd.Series, strategy_rets: pd.Series, freq_per_year: int = 252*24):
    total_return = equity_series.iloc[-1] / equity_series.iloc[0] - 1.0
    n_periods = len(strategy_rets)
    years = n_periods / freq_per_year if freq_per_year>0 else np.nan
    cagr = (equity_series.iloc[-1] / equity_series.iloc[0]) ** (1 / years) - 1 if years > 0 else np.nan
    ann_vol = strategy_rets.std() * np.sqrt(freq_per_year) if n_periods>1 else 0.0
    ann_ret = strategy_rets.mean() * freq_per_year
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
    neg = strategy_rets[strategy_rets < 0]
    downside = neg.std() * np.sqrt(freq_per_year) if len(neg)>0 else 0.0
    sortino = ann_ret / downside if downside > 0 else np.nan
    roll_max = equity_series.cummax()
    max_dd = (equity_series / roll_max - 1.0).min()
    win_rate = (strategy_rets > 0).mean()
    return {
        "total_return": float(total_return),
        "cagr": float(cagr) if not np.isnan(cagr) else None,
        "annualized_return": float(ann_ret),
        "annualized_vol": float(ann_vol) if not np.isnan(ann_vol) else None,
        "sharpe": float(sharpe) if not np.isnan(sharpe) else None,
        "sortino": float(sortino) if not np.isnan(sortino) else None,
        "max_drawdown": float(max_dd),
        "win_rate": float(win_rate)
    }

def visualize_equity(dates, equity, outpath, title="Equity Curve"):
    plt.figure(figsize=(12,5))
    plt.plot(dates, equity, label="Equity")
    plt.fill_between(dates, equity, pd.Series(equity).cummax(), where=(equity < pd.Series(equity).cummax()), color="red", alpha=0.15)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Equity")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()
    print(f"Saved chart → {outpath}")

def get_dynamic_risk_factor(current_equity, equity_history):
    """
    Returns multiplier for BASE_RISK_PER_TRADE_FRAC based on drawdown from peak.
    Equity_history should be list-like historical equities including current_equity.
    """
    if not equity_history:
        return 1.0
    peak = max(equity_history)
    if peak <= 0:
        return 1.0
    drawdown = (peak - current_equity) / peak
    for thr, factor in zip(DD_SCALE_THRESHOLDS[::-1], DD_SCALE_FACTORS[::-1]):  # check largest threshold first
        if drawdown >= thr:
            return factor
    return 1.0

# ---------------- Main simulation ----------------
def simulate_backtest(
    data_path: str = DATA_PATH,
    model_path: str = MODEL_PATH,
    initial_capital: float = INITIAL_CAPITAL,
    save_dir: str = SAVE_DIR
):

    # load data
    if not os.path.exists(data_path):
        raise FileNotFoundError(data_path)

    df = pd.read_csv(data_path)
    # compat rename
    rename_map = {"GOLD_Close":"Close","GOLD_Open":"Open","GOLD_High":"High","GOLD_Low":"Low"}
    df = df.rename(columns=rename_map)
    if "Close" not in df.columns:
        raise ValueError("CSV must include Close column")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    # compute returns and ATR, vol
    df["return"] = df["Close"].pct_change().fillna(0.0)
    df["logret"] = np.log(df["Close"] / df["Close"].shift(1)).fillna(0.0)
    df["ATR"] = compute_atr(df, window=ATR_WINDOW, col_high="High", col_low="Low", col_close="Close")
    df["vol_24h"] = df["return"].rolling(VOL_WINDOW).std().bfill().fillna(0.0)

    # Attempt to import user's feature generator
    add_feats = None
    try:
        sys.path.insert(0, os.path.join(ROOT, "src", "model"))
        from train_xgb_v2 import add_technical_features as add_feats
        print("✅ Loaded add_technical_features from train_xgb_v2")
    except Exception as e:
        print(f"⚠️ Failed to import add_technical_features from train_xgb_v2: {e}")

    if add_feats:
        try:
            df = add_feats(df)
            print("✅ Applied add_technical_features successfully.")
        except Exception as e:
            print(f"⚠️ Feature generator error while applying: {e}")
    else:
        print("⚠️ No valid add_technical_features found — may cause missing features.")

    # add regime if exists
    if "regime" not in df.columns:
        try:
            from utils.market_regime_detector import MarketRegimeDetector
            det = MarketRegimeDetector(ma_fast=20, ma_slow=50)
            r = det.detect_regime(df)
            if "regime" in r.columns:
                df = pd.merge(df, r[["Date","regime"]], on="Date", how="left")
                print("✅ Added regime from utils.")
            else:
                df["regime"] = "neutral"
        except Exception:
            df["regime"] = "neutral"

    # load model
    with open(model_path, "rb") as f:
        model_meta = pickle.load(f)
    if not isinstance(model_meta, dict):
        model_meta = {"model": model_meta}
    print(f"✅ Loaded specific model: {os.path.basename(model_path)}")
    model_obj = model_meta.get("calibrated_model") or model_meta.get("raw_ensemble") or model_meta.get("model")

    # prepare feature matrix: keep numeric columns (exclude date, target cols)
    exclude = {"Date","target_bin","target_ret","future_ret","regime"}
    feats = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    X = df[feats].copy().fillna(0.0)
    X = safe_feature_align(X, model_meta)

    # try predict proba
    probs = model_predict_proba(model_obj, X)
    if probs is not None:
        if probs.ndim == 2 and probs.shape[1] > 1:
            buy_prob = probs[:, -1]
            sell_prob = probs[:, 0]
        else:
            buy_prob = probs.ravel()
            sell_prob = 1.0 - buy_prob
        df["buy_prob"] = buy_prob
        df["sell_prob"] = sell_prob
        saved_thr = model_meta.get("threshold") if isinstance(model_meta, dict) else None
        buy_thr = float(saved_thr) if (saved_thr is not None and saved_thr>0) else BUY_PROB_DEFAULT
        sell_thr = buy_thr - 0.12
        print(f"Using buy_thr={buy_thr:.3f}, sell_thr={sell_thr:.3f}")
    else:
        print("Model has no predict_proba -> using predict fallback")
        preds = model_obj.predict(X)
        df["buy_prob"] = np.where(preds==1, 0.75, 0.25)
        df["sell_prob"] = 1.0 - df["buy_prob"]
        buy_thr, sell_thr = BUY_PROB_DEFAULT, BUY_PROB_DEFAULT - 0.12

    # prepare volatility cutoff (dynamic)
    vol_cutoff = df["vol_24h"].quantile(VOL_THRESHOLD_PCTL)
    print(f"Volatility cutoff (p{int(VOL_THRESHOLD_PCTL*100)}): {vol_cutoff:.6f}")

    # simulation state
    equity = initial_capital
    equities = []
    equity_history_for_dd = [equity]
    position = 0.0         # +units for long, -units for short
    entry_price = 0.0
    entry_atr = 0.0
    entry_value = 0.0
    trade_log = []

    # iterate bars: use previous period signal to enter next bar
    for i in range(1, len(df)):
        date = df.loc[i, "Date"]
        price = float(df.loc[i, "Close"])
        prev_buy_prob = float(df.loc[i-1, "buy_prob"])
        prev_sell_prob = float(df.loc[i-1, "sell_prob"]) if "sell_prob" in df.columns else (1.0 - prev_buy_prob)
        regime = df.loc[i, "regime"] if "regime" in df.columns else "neutral"
        vol_now = float(df.loc[i, "vol_24h"])
        atr_now = float(df.loc[i, "ATR"] if "ATR" in df.columns else 0.0)

        # dynamic risk factor based on drawdown from historical high
        dd_factor = 1.0
        # effective risk per trade after drawdown scaling
        eff_risk_frac = BASE_RISK_PER_TRADE_FRAC 

        # compute "confidence" from probability: map [0.5,1] -> [0,1]
        conf_long_raw = max(0.0, (prev_buy_prob - 0.5) * 2.0)  # 0..1
        conf_short_raw = max(0.0, (prev_sell_prob - 0.5) * 2.0)
        vol_block = vol_now > vol_cutoff
        rweight = REGIME_WEIGHTS.get(regime, 0.9)

        # decide wants
        want_long = (prev_buy_prob >= buy_thr) and (prev_buy_prob >= MIN_PROBA_FOR_SIGNAL) and (conf_long_raw >= MIN_CONFIDENCE) and (not vol_block) and (rweight>0)
        want_short = ENABLE_SHORTS and (prev_sell_prob >= (1.0 - buy_thr)) and (prev_sell_prob >= MIN_PROBA_FOR_SIGNAL) and (conf_short_raw >= MIN_CONFIDENCE) and (not vol_block) and (rweight>0)

        # compute stop/tp distances
        stop_distance_long = max(atr_now * ATR_MULTIPLIER_SL, price * MIN_STOP_PCT)
        tp_distance_long = max(atr_now * ATR_MULTIPLIER_TP, price * MIN_STOP_PCT * 2)
        stop_distance_short = stop_distance_long
        tp_distance_short = tp_distance_long

        # ENTRY: flat -> try enter long or short (prioritize side with stronger confidence)
        if position == 0:
            if want_long or want_short:
                if want_long and want_short:
                    chosen_side = "long" if conf_long_raw >= conf_short_raw else "short"
                elif want_long:
                    chosen_side = "long"
                else:
                    chosen_side = "short"

                if chosen_side == "long" and want_long:
                    confidence = conf_long_raw
                    # effective risk scaled by confidence and regime and drawdown factor
                    risk_amount = equity * eff_risk_frac * (0.5 + 0.5*confidence) * rweight
                    if stop_distance_long <= 1e-8:
                        stop_distance_long = price * 0.002
                    units = max(0.0, risk_amount / stop_distance_long)
                    max_invest = equity * MAX_POSITION_FRAC * rweight
                    invest = min(units * price, max_invest)
                    if invest > 0:
                        units = invest / price
                        entry_price = price * (1.0 + SLIPPAGE)
                        fee_entry = invest * COMMISSION
                        entry_time = date
                        position = units
                        entry_atr = atr_now
                        entry_value = units * entry_price
                        equity -= fee_entry
                        trade_log.append({
                            "side": "long",
                            "entry_idx": i, "entry_date": entry_time, "entry_price": entry_price,
                            "units": units, "invest": invest, "fee_entry": fee_entry, "confidence": confidence,
                            "regime": regime, "stop_distance": stop_distance_long, "tp_distance": tp_distance_long,
                            "dd_factor": dd_factor
                        })
                elif chosen_side == "short" and want_short:
                    confidence = conf_short_raw
                    # apply short-specific risk multiplier
                    risk_amount = equity * eff_risk_frac * SHORT_RISK_MULT * (0.5 + 0.5*confidence) * rweight
                    if stop_distance_short <= 1e-8:
                        stop_distance_short = price * 0.002
                    units = max(0.0, risk_amount / stop_distance_short)
                    max_invest = equity * MAX_POSITION_FRAC * rweight
                    invest = min(units * price, max_invest)
                    if invest > 0:
                        units = invest / price
                        entry_price = price * (1.0 - SLIPPAGE)  # short entry a bit lower
                        fee_entry = invest * COMMISSION
                        entry_time = date
                        position = -units
                        entry_atr = atr_now
                        entry_value = units * entry_price
                        equity -= fee_entry
                        trade_log.append({
                            "side": "short",
                            "entry_idx": i, "entry_date": entry_time, "entry_price": entry_price,
                            "units": units, "invest": invest, "fee_entry": fee_entry, "confidence": confidence,
                            "regime": regime, "stop_distance": stop_distance_short, "tp_distance": tp_distance_short,
                            "dd_factor": dd_factor
                        })

        # If in position: evaluate exit conditions (both long and short)
        if position != 0 and trade_log:
            last_trade = trade_log[-1]
            if position > 0:
                # long
                mark_price = price * (1.0 - SLIPPAGE)
                current_value = position * mark_price
                invested = last_trade["invest"]
                # stop
                if mark_price <= (last_trade["entry_price"] - last_trade["stop_distance"]):
                    exit_price = mark_price
                    fee_exit = current_value * COMMISSION
                    pnl = (current_value - (position * last_trade["entry_price"])) - fee_exit
                    equity += pnl
                    last_trade.update({
                        "exit_idx": i, "exit_date": date, "exit_price": exit_price,
                        "fee_exit": fee_exit, "pnl": pnl, "return": pnl / invested if invested>0 else None, "exit_reason": "stop"
                    })
                    position = 0.0
                # take profit
                elif mark_price >= (last_trade["entry_price"] + last_trade["tp_distance"]):
                    exit_price = mark_price
                    fee_exit = current_value * COMMISSION
                    pnl = (current_value - (position * last_trade["entry_price"])) - fee_exit
                    equity += pnl
                    last_trade.update({
                        "exit_idx": i, "exit_date": date, "exit_price": exit_price,
                        "fee_exit": fee_exit, "pnl": pnl, "return": pnl / invested if invested>0 else None, "exit_reason": "takeprofit"
                    })
                    position = 0.0
                else:
                    # signal-based exit using current buy prob
                    current_prob = float(df.loc[i, "buy_prob"])
                    if current_prob <= (buy_thr - 0.10):
                        exit_price = mark_price
                        fee_exit = current_value * COMMISSION
                        pnl = (current_value - (position * last_trade["entry_price"])) - fee_exit
                        equity += pnl
                        last_trade.update({
                            "exit_idx": i, "exit_date": date, "exit_price": exit_price,
                            "fee_exit": fee_exit, "pnl": pnl, "return": pnl / invested if invested>0 else None, "exit_reason": "signal_exit"
                        })
                        position = 0.0
            else:
                # short
                units = abs(position)
                mark_price = price * (1.0 + SLIPPAGE)
                current_value = units * mark_price
                invested = last_trade["invest"]
                # stop for short (price rose)
                if mark_price >= (last_trade["entry_price"] + last_trade["stop_distance"]):
                    exit_price = mark_price
                    fee_exit = current_value * COMMISSION
                    pnl = ((last_trade["entry_price"] * units) - current_value) - fee_exit
                    equity += pnl
                    last_trade.update({
                        "exit_idx": i, "exit_date": date, "exit_price": exit_price,
                        "fee_exit": fee_exit, "pnl": pnl, "return": pnl / invested if invested>0 else None, "exit_reason": "stop"
                    })
                    position = 0.0
                # takeprofit for short (price fell)
                elif mark_price <= (last_trade["entry_price"] - last_trade["tp_distance"]):
                    exit_price = mark_price
                    fee_exit = current_value * COMMISSION
                    pnl = ((last_trade["entry_price"] * units) - current_value) - fee_exit
                    equity += pnl
                    last_trade.update({
                        "exit_idx": i, "exit_date": date, "exit_price": exit_price,
                        "fee_exit": fee_exit, "pnl": pnl, "return": pnl / invested if invested>0 else None, "exit_reason": "takeprofit"
                    })
                    position = 0.0
                else:
                    # signal-based exit using current sell prob
                    current_sell_prob = float(df.loc[i, "sell_prob"]) if "sell_prob" in df.columns else (1.0 - float(df.loc[i, "buy_prob"]))
                    if current_sell_prob <= (1.0 - buy_thr - 0.10):
                        exit_price = mark_price
                        fee_exit = current_value * COMMISSION
                        pnl = ((last_trade["entry_price"] * units) - current_value) - fee_exit
                        equity += pnl
                        last_trade.update({
                            "exit_idx": i, "exit_date": date, "exit_price": exit_price,
                            "fee_exit": fee_exit, "pnl": pnl, "return": pnl / invested if invested>0 else None, "exit_reason": "signal_exit"
                        })
                        position = 0.0

        # record equity this bar (mark to market)
        if position > 0:
            mark_price = price * (1.0 - SLIPPAGE)
            equities.append(equity + position * mark_price)
        elif position < 0:
            units = abs(position)
            mark_price = price * (1.0 + SLIPPAGE)
            equities.append(equity + (last_trade["entry_price"] * units) - (units * mark_price))
        else:
            equities.append(equity)

        # update equity history for drawdown calc
        equity_history_for_dd.append(equities[-1])

    # finalize DataFrame alignment
    if len(equities) < len(df):
        equities += [equities[-1]] * (len(df) - len(equities))
    df = df.iloc[:len(equities)].copy().reset_index(drop=True)
    df["equity"] = equities
    df["strategy_ret"] = df["equity"].pct_change().fillna(0.0)

    # metrics
    if len(df) >= 2:
        delta_days = (df.loc[1, "Date"] - df.loc[0, "Date"]).total_seconds() / (3600*24)
        freq = int(round(365.0 / delta_days)) if delta_days > 0 else 24*252
    else:
        freq = 24*252
    metrics = compute_backtest_metrics(df["equity"], df["strategy_ret"], freq_per_year=freq)

    # save logs
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    trades_df = pd.DataFrame(trade_log)
    trades_path = os.path.join(save_dir, f"trades_profitboost_v6_2_{ts}.csv")
    trades_df.to_csv(trades_path, index=False)
    df_path = os.path.join(save_dir, f"backtest_profitboost_v6_2_{ts}.csv")
    df.to_csv(df_path, index=False)
    chart_path = os.path.join(save_dir, f"equity_profitboost_v6_2_{ts}.png")
    visualize_equity(df["Date"], df["equity"], chart_path, "Profit-Boost v6.2 Equity Curve")

    # print metrics
    print("\n===== BACKTEST METRICS (v6.2) =====")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k:25s}: {v:.6f}")
        else:
            print(f"{k:25s}: {v}")
    print(f"Trades saved → {trades_path}")
    print(f"Backtest CSV → {df_path}")
    print(f"Chart → {chart_path}")

    return {"metrics": metrics, "trades_path": trades_path, "df_path": df_path, "chart_path": chart_path}

if __name__ == "__main__":
    simulate_backtest()
    print("Done.")
