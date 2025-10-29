#!/usr/bin/env python3
"""
simulate_trading_v6_profit_boost_updated.py
-------------------------------------------
Updated Profit-Boost backtester for Gold Scalper (v6 -> v6.1)
Changes aimed to increase realized profit while keeping risk controls:
 - Larger risk per trade (but still fraction-based & confidence-weighted)
 - Higher max position fraction
 - Smaller ATR stop, larger ATR take-profit (improve RR)
 - Looser signal / vol filters to increase trade frequency
 - Optional shorting enabled (symmetric logic)
 - Keeps commission/slippage modeling and compounding
Author: adapted for Lim Jia Xuan (Profit-Boost v6.1)
"""

import os, sys, pickle
from datetime import datetime
from typing import Optional, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- CONFIG ----------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODEL_DIR = os.path.join(ROOT, "models")
DATA_PATH = os.path.join(ROOT, "data", "final", "final_dataset_hourly.csv")
SAVE_DIR = os.path.join(ROOT, "backtest_results")
os.makedirs(SAVE_DIR, exist_ok=True)

SEED = 42
np.random.seed(SEED)

INITIAL_CAPITAL = 10000.0

# Risk sizing (more aggressive but still capped)
RISK_PER_TRADE_FRAC = 0.05   # increased from 0.03 -> risk 5% of equity base (scaled by confidence)
MAX_POSITION_FRAC = 0.50     # cap: don't allocate more than 50% of equity to a single position

# ATR / stop / tp (improve RR)
ATR_WINDOW = 14
ATR_MULTIPLIER_SL = 1.8     # tighter stop
ATR_MULTIPLIER_TP = 3.2     # larger take-profit
MIN_STOP_PCT = 0.001        # minimum stop distance as fraction of price (0.1%)

# Volatility filter
VOL_WINDOW = 24             # hours
VOL_THRESHOLD_PCTL = 0.90   # skip trading if recent vol > 90th percentile (looser than before)

# Model thresholds & fallback (looser)
BUY_PROB_DEFAULT = 0.52 
SELL_PROB_DEFAULT = 0.48
MIN_CONFIDENCE = 0.35        # require confidence >= this to trade (after mapping)
MIN_PROBA_FOR_SIGNAL = 0.50

# Costs
COMMISSION = 0.0005  # proportional
SLIPPAGE = 0.0005

REGIME_WEIGHTS = {"bull": 1.0, "neutral": 0.9, "bear": 0.7}

# Enable shorting (symmetric)
ENABLE_SHORTS = True

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
    return atr.fillna(method="bfill").fillna(0.0)

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

# ---------------- Main simulation ----------------
def simulate_backtest(
    data_path: str = DATA_PATH,
    model_dir: str = MODEL_DIR,
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
    # compute ATR
    df["ATR"] = compute_atr(df, window=ATR_WINDOW, col_high="High", col_low="Low", col_close="Close")
    # rolling vol (std of returns)
    df["vol_24h"] = df["return"].rolling(VOL_WINDOW).std().fillna(method="bfill").fillna(0.0)
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
    model_meta = load_latest_model(model_dir)
    model_obj = model_meta.get("calibrated_model") or model_meta.get("raw_ensemble") or model_meta.get("model")

    # prepare feature matrix: keep numeric columns (exclude date, target cols)
    exclude = {"Date","target_bin","target_ret","future_ret","regime"}
    feats = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    X = df[feats].copy().fillna(0.0)
    X = safe_feature_align(X, model_meta)

    # try predict proba
    probs = model_predict_proba(model_obj, X)
    if probs is not None:
        # if binary, probs.shape[1]==2 expected
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
        buy_thr, sell_thr = BUY_PROB_DEFAULT, SELL_PROB_DEFAULT

    # prepare volatility cutoff (dynamic)
    vol_cutoff = df["vol_24h"].quantile(VOL_THRESHOLD_PCTL)
    print(f"Volatility cutoff (p{int(VOL_THRESHOLD_PCTL*100)}): {vol_cutoff:.6f}")

    # simulation state
    equity = initial_capital
    equities = []
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

        # compute "confidence" from probability: map [0.5,1] -> [0,1]; negative side ignored for longs
        conf_long_raw = max(0.0, (prev_buy_prob - 0.5) * 2.0)  # 0..1
        conf_short_raw = max(0.0, (prev_sell_prob - 0.5) * 2.0)
        # final confidence depends on which side we consider later
        vol_block = vol_now > vol_cutoff
        rweight = REGIME_WEIGHTS.get(regime, 0.9)

        # decide wants
        want_long = (prev_buy_prob >= buy_thr) and (prev_buy_prob >= MIN_PROBA_FOR_SIGNAL) and (conf_long_raw >= MIN_CONFIDENCE) and (not vol_block) and (rweight>0)
        want_short = ENABLE_SHORTS and (prev_sell_prob >= (1.0 - buy_thr + 0.0)) and (prev_sell_prob >= MIN_PROBA_FOR_SIGNAL) and (conf_short_raw >= MIN_CONFIDENCE) and (not vol_block) and (rweight>0)

        # compute stop/tp distances
        stop_distance_long = max(atr_now * ATR_MULTIPLIER_SL, price * MIN_STOP_PCT)
        tp_distance_long = max(atr_now * ATR_MULTIPLIER_TP, price * MIN_STOP_PCT * 2)
        stop_distance_short = stop_distance_long
        tp_distance_short = tp_distance_long

        # ENTRY: flat -> try enter long or short (prioritize side with stronger confidence)
        if position == 0:
            # decide side priority
            if want_long or want_short:
                # compare confidences to pick side (if both true)
                if want_long and want_short:
                    # choose larger raw confidence
                    chosen_side = "long" if conf_long_raw >= conf_short_raw else "short"
                elif want_long:
                    chosen_side = "long"
                else:
                    chosen_side = "short"

                if chosen_side == "long" and want_long:
                    confidence = conf_long_raw
                    # position sizing: risk_amount = equity * RISK_PER_TRADE_FRAC * (0.5 + 0.5*confidence) * rweight
                    risk_amount = equity * RISK_PER_TRADE_FRAC * (0.5 + 0.5*confidence) * rweight
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
                            "regime": regime, "stop_distance": stop_distance_long, "tp_distance": tp_distance_long
                        })
                elif chosen_side == "short" and want_short:
                    confidence = conf_short_raw
                    risk_amount = equity * RISK_PER_TRADE_FRAC * (0.5 + 0.5*confidence) * rweight
                    if stop_distance_short <= 1e-8:
                        stop_distance_short = price * 0.002
                    units = max(0.0, risk_amount / stop_distance_short)
                    max_invest = equity * MAX_POSITION_FRAC * rweight
                    invest = min(units * price, max_invest)
                    if invest > 0:
                        units = invest / price
                        entry_price = price * (1.0 - SLIPPAGE)  # short entry at slightly lower
                        fee_entry = invest * COMMISSION
                        entry_time = date
                        position = -units  # negative units indicates short
                        entry_atr = atr_now
                        entry_value = units * entry_price
                        equity -= fee_entry
                        trade_log.append({
                            "side": "short",
                            "entry_idx": i, "entry_date": entry_time, "entry_price": entry_price,
                            "units": units, "invest": invest, "fee_entry": fee_entry, "confidence": confidence,
                            "regime": regime, "stop_distance": stop_distance_short, "tp_distance": tp_distance_short
                        })

        # If in position: evaluate exit conditions (both long and short)
        if position != 0:
            last_trade = trade_log[-1]
            # mark to market (account for exit slippage)
            if position > 0:
                # long
                mark_price = price * (1.0 - SLIPPAGE)
                current_value = position * mark_price
                invested = last_trade["invest"]
                # check stop
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
                    entry_price = 0.0
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
                    entry_price = 0.0
                else:
                    # signal-based exit (use current prob)
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
                        entry_price = 0.0
            else:
                # short (position < 0)
                units = abs(position)
                mark_price = price * (1.0 + SLIPPAGE)
                current_value = units * mark_price
                invested = last_trade["invest"]
                # for short, stop is when price rises above entry + stop_distance
                if mark_price >= (last_trade["entry_price"] + last_trade["stop_distance"]):
                    exit_price = mark_price
                    fee_exit = current_value * COMMISSION
                    pnl = ( (last_trade["entry_price"] * units) - current_value ) - fee_exit
                    equity += pnl
                    last_trade.update({
                        "exit_idx": i, "exit_date": date, "exit_price": exit_price,
                        "fee_exit": fee_exit, "pnl": pnl, "return": pnl / invested if invested>0 else None, "exit_reason": "stop"
                    })
                    position = 0.0
                    entry_price = 0.0
                elif mark_price <= (last_trade["entry_price"] - last_trade["tp_distance"]):
                    exit_price = mark_price
                    fee_exit = current_value * COMMISSION
                    pnl = ( (last_trade["entry_price"] * units) - current_value ) - fee_exit
                    equity += pnl
                    last_trade.update({
                        "exit_idx": i, "exit_date": date, "exit_price": exit_price,
                        "fee_exit": fee_exit, "pnl": pnl, "return": pnl / invested if invested>0 else None, "exit_reason": "takeprofit"
                    })
                    position = 0.0
                    entry_price = 0.0
                else:
                    # signal-based exit for short (use current sell_prob)
                    current_sell_prob = float(df.loc[i, "sell_prob"]) if "sell_prob" in df.columns else (1.0 - float(df.loc[i, "buy_prob"]))
                    if current_sell_prob <= (1.0 - buy_thr - 0.10):
                        exit_price = mark_price
                        fee_exit = current_value * COMMISSION
                        pnl = ( (last_trade["entry_price"] * units) - current_value ) - fee_exit
                        equity += pnl
                        last_trade.update({
                            "exit_idx": i, "exit_date": date, "exit_price": exit_price,
                            "fee_exit": fee_exit, "pnl": pnl, "return": pnl / invested if invested>0 else None, "exit_reason": "signal_exit"
                        })
                        position = 0.0
                        entry_price = 0.0

        # record equity this bar (mark to market)
        if position > 0:
            mark_price = price * (1.0 - SLIPPAGE)
            equities.append(equity + position * mark_price)
        elif position < 0:
            units = abs(position)
            mark_price = price * (1.0 + SLIPPAGE)
            # short current P&L: entry*units - current_value
            equities.append(equity + (last_trade["entry_price"] * units) - (units * mark_price))
        else:
            equities.append(equity)

    # finalize DataFrame alignment
    if len(equities) < len(df):
        equities += [equities[-1]] * (len(df) - len(equities))
    df = df.iloc[:len(equities)].copy().reset_index(drop=True)
    df["equity"] = equities
    df["strategy_ret"] = df["equity"].pct_change().fillna(0.0)

    # metrics
    if len(df) >= 2:
        delta_days = (df.loc[1,"Date"] - df.loc[0,"Date"]).total_seconds() / (3600*24)
        freq = int(round(365.0 / delta_days)) if delta_days>0 else 24*252
    else:
        freq = 24*252
    metrics = compute_backtest_metrics(df["equity"], df["strategy_ret"], freq_per_year=freq)

    # save logs
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    trades_df = pd.DataFrame(trade_log)
    trades_path = os.path.join(save_dir, f"trades_profitboost_v6_1_{ts}.csv")
    trades_df.to_csv(trades_path, index=False)
    df_path = os.path.join(save_dir, f"backtest_profitboost_v6_1_{ts}.csv")
    df.to_csv(df_path, index=False)
    chart_path = os.path.join(save_dir, f"equity_profitboost_v6_1_{ts}.png")
    visualize_equity(df["Date"], df["equity"], chart_path, "Profit-Boost v6.1 Equity Curve")

    # print metrics
    print("\n===== BACKTEST METRICS (v6.1) =====")
    for k,v in metrics.items():
        if isinstance(v, float):
            print(f"{k:25s}: {v:.6f}")
        else:
            print(f"{k:25s}: {v}")
    print(f"Trades saved → {trades_path}")
    print(f"Backtest CSV → {df_path}")
    print(f"Chart → {chart_path}")

    return {"metrics":metrics, "trades_path":trades_path, "df_path":df_path, "chart_path":chart_path}

if __name__ == "__main__":
    simulate_backtest()
    print("Done.")
