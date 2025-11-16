#!/usr/bin/env python3
"""
strategy_v6_6_stable.py
-----------------------
Profit-Boost v6.6 (Stable)

Design goals:
- Conservative changes vs v6.4 (minimize performance regressions)
- Fix lookahead / feature-shift mismatch
- Use next-bar open for execution, simulate spread/slippage/commissions
- Default: keep v6.4-like exit behavior (no TP/trailing). TP/trailing available but off.
- Robust feature alignment & predict_proba with numpy arrays
- Use ffill/bfill to avoid deprecated fillna(method=...)
Author: adapted for Lim Jia Xuan
"""
import os
import pickle
from datetime import datetime
from typing import Dict, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------
# CONFIG (tune these conservatively)
# -------------------------
INITIAL_CAPITAL = 10000.0
BASE_RISK_PER_TRADE_FRAC = 0.02   # 2% risk per trade by default
MAX_POSITION_FRAC = 0.5
ATR_WINDOW = 14
ATR_MULTIPLIER_SL = 1.5
ENABLE_TAKE_PROFIT = False       # v6.6 default: OFF (safe)
ATR_MULTIPLIER_TP = 4.0
ENABLE_TRAILING = False          # v6.6 default: OFF (safe)
TRAILING_ATR_MULT = 1.2
MIN_STOP_PCT = 0.0005
VOL_WINDOW = 24
VOL_THRESHOLD_PCTL = 0.995
BUY_PROB_DEFAULT = 0.50
COMMISSION_PER_SIDE = 0.0005
SLIPPAGE_PCT = 0.0006
SPREAD_PCT = 0.0006
REGIME_WEIGHTS = {"bull": 1.2, "neutral": 1.0, "bear": 0.9}
ENABLE_SHORTS = True

# -------------------------
class ProfitBoostStrategy:
    def __init__(self, model_meta: Dict, save_dir: str, shift_features: bool = True):
        """
        model_meta: dict loaded from your saved pickle (should include calibrated_model/raw_ensemble and optionally feature_cols)
        save_dir: folder to save outputs
        shift_features: If True, generated rolling features are shifted by 1 (prevents lookahead). 
                        IMPORTANT: training must match this choice.
        """
        self.model_meta = model_meta
        self.save_dir = save_dir
        self.shift_features = bool(shift_features)
        os.makedirs(save_dir, exist_ok=True)

    # -------------------------
    def safe_feature_align_df(self, X_df: pd.DataFrame) -> pd.DataFrame:
        feat_required = None
        mm = self.model_meta
        if isinstance(mm, dict) and mm.get("feature_cols"):
            feat_required = list(mm["feature_cols"])
        else:
            model = mm.get("calibrated_model") or mm.get("raw_ensemble") or mm.get("model")
            if model is not None and hasattr(model, "feature_names_in_"):
                feat_required = list(model.feature_names_in_)
        if feat_required is not None:
            for f in feat_required:
                if f not in X_df.columns:
                    X_df[f] = 0.0
            X_df = X_df[feat_required].copy()
        X_df = X_df.replace([np.inf, -np.inf], np.nan)
        # fill with medians then fill remaining with 0
        X_df = X_df.fillna(X_df.median().fillna(0.0)).ffill().bfill().fillna(0.0)
        X_df = X_df.clip(lower=-1e9, upper=1e9)
        return X_df

    def model_predict_proba(self, X_df: pd.DataFrame) -> Optional[np.ndarray]:
        m = self.model_meta.get("calibrated_model") or self.model_meta.get("raw_ensemble") or self.model_meta.get("model")
        if m is None:
            return None
        # pass numpy array to avoid sklearn warning when StandardScaler was fitted without feature names
        try:
            X_arr = X_df.values
            proba = m.predict_proba(X_arr)
            return np.asarray(proba)
        except Exception:
            try:
                proba = m.predict_proba(X_df)
                return np.asarray(proba)
            except Exception:
                # try averaging estimators
                try:
                    ests = getattr(m, "estimators_", []) or getattr(m, "estimators", [])
                    probs = []
                    for e in ests:
                        if hasattr(e, "predict_proba"):
                            try:
                                probs.append(e.predict_proba(X_arr))
                            except Exception:
                                try:
                                    probs.append(e.predict_proba(X_df))
                                except Exception:
                                    continue
                    if probs:
                        return np.mean(probs, axis=0)
                except Exception:
                    pass
        return None

    @staticmethod
    def compute_atr(df: pd.DataFrame, window: int = ATR_WINDOW) -> pd.Series:
        high = df["High"]
        low = df["Low"]
        prev_close = df["Close"].shift(1)
        tr = pd.concat([(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
        atr = tr.rolling(window, min_periods=3).mean()
        return atr

    def _rsi_shifted(self, series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        up = delta.clip(lower=0).rolling(period, min_periods=3).mean()
        down = (-delta.clip(upper=0)).rolling(period, min_periods=3).mean()
        rs = up / (down + 1e-9)
        rsi = 100 - (100 / (1 + rs))
        return rsi.shift(1) if self.shift_features else rsi

    # -------------------------
    def simulate(self, df: pd.DataFrame) -> Dict:
        """
        Run backtest.
        Expects df with Date, GOLD_Open, GOLD_High, GOLD_Low, GOLD_Close (or Open/High/Low/Close).
        """
        df = df.rename(columns={
            "GOLD_Close": "Close",
            "GOLD_Open": "Open",
            "GOLD_High": "High",
            "GOLD_Low": "Low"
        }).copy()

        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

        # build features — shift if requested (must match training)
        shift = 1 if self.shift_features else 0
        df["Return"] = df["Close"].pct_change().shift(shift)
        df["SMA_20"] = df["Close"].rolling(20, min_periods=5).mean().shift(shift)
        df["SMA_60"] = df["Close"].rolling(60, min_periods=10).mean().shift(shift)
        df["EMA_12"] = df["Close"].ewm(span=12, adjust=False).mean().shift(shift)
        df["EMA_26"] = df["Close"].ewm(span=26, adjust=False).mean().shift(shift)
        df["MACD"] = (df["EMA_12"] - df["EMA_26"]).shift(shift)
        df["RSI_14"] = self._rsi_shifted(df["Close"], 14)
        df["ATR"] = self.compute_atr(df).shift(shift).ffill().bfill().fillna(0.0)
        df["vol_24h"] = df["Close"].pct_change().rolling(VOL_WINDOW).std().shift(shift).ffill().bfill().fillna(0.0)
        df["momentum_ok"] = (df["SMA_20"] > df["SMA_60"]).astype(bool)

        # prepare numeric frame for model and align features
        numeric_df = df.select_dtypes(include=[np.number]).copy().fillna(0.0)
        X_df = self.safe_feature_align_df(numeric_df)

        proba = self.model_predict_proba(X_df)
        if proba is not None:
            if proba.ndim == 2 and proba.shape[1] > 1:
                df["buy_prob"] = np.clip(proba[:, -1], 0.0, 1.0)
            else:
                df["buy_prob"] = np.clip(proba.ravel(), 0.0, 1.0)
        else:
            df["buy_prob"] = BUY_PROB_DEFAULT

        buy_thr = float(self.model_meta.get("threshold", BUY_PROB_DEFAULT))
        vol_cutoff = df["vol_24h"].quantile(VOL_THRESHOLD_PCTL)

        # bookkeeping
        capital = float(INITIAL_CAPITAL)
        cash = capital
        position_units = 0.0
        position_side = None
        entry_price = None
        equities = []
        trade_log = []

        # iterate: execute signals at next bar open (i => look at i, execute at i+1)
        for i in range(0, len(df) - 1):
            cur = df.loc[i]
            nxt = df.loc[i + 1]
            exec_price = nxt["Open"] if not pd.isna(nxt["Open"]) else nxt["Close"]
            buy_prob = float(cur.get("buy_prob", BUY_PROB_DEFAULT))
            vol_now = float(cur.get("vol_24h", 0.0))
            atr_now = float(cur.get("ATR", 0.0))
            momentum_ok = bool(cur.get("momentum_ok", True))
            regime = cur.get("regime", "neutral")
            rweight = REGIME_WEIGHTS.get(regime, 1.0)

            # compute stop distance (price units)
            stop_dist = max(atr_now * ATR_MULTIPLIER_SL, exec_price * MIN_STOP_PCT)
            if stop_dist <= 0:
                stop_dist = exec_price * MIN_STOP_PCT

            # sizing by risk
            adaptive_risk_amt = capital * BASE_RISK_PER_TRADE_FRAC * rweight
            vol_scale = np.clip((df["vol_24h"].median() + 1e-9) / (vol_now + 1e-9), 0.5, 1.5)
            adaptive_risk_amt *= vol_scale
            desired_units = adaptive_risk_amt / (stop_dist + 1e-12)
            max_units = (capital * MAX_POSITION_FRAC) / (exec_price + 1e-12)
            units = float(np.clip(desired_units, 0.0, max_units))

            want_long = (buy_prob >= buy_thr) and momentum_ok and (vol_now <= vol_cutoff)
            want_short = ENABLE_SHORTS and (buy_prob <= (1.0 - buy_thr)) and (not momentum_ok) and (vol_now <= vol_cutoff)

            # ENTRY (only if flat)
            if position_side is None:
                if want_long and units >= 1e-6:
                    # compute realistic entry including slippage and half spread
                    entry_price = exec_price * (1.0 + SLIPPAGE_PCT + SPREAD_PCT / 2.0)
                    position_units = units
                    position_side = "long"
                    # record entry commission
                    entry_comm = entry_price * position_units * COMMISSION_PER_SIDE
                    cash -= entry_comm
                    # set TP/SL optionally
                    sl_price = max(entry_price - stop_dist, entry_price * (1 - 0.1))
                    tp_price = entry_price + ATR_MULTIPLIER_TP * atr_now if ENABLE_TAKE_PROFIT else None
                    trail = entry_price - TRAILING_ATR_MULT * atr_now if ENABLE_TRAILING else None
                    trade_log.append({
                        "side": "long",
                        "entry_time": nxt["Date"],
                        "entry_price": entry_price,
                        "units": position_units,
                        "sl_price": sl_price,
                        "tp_price": tp_price,
                        "trail_stop": trail,
                        "entry_commission": entry_comm,
                        "entry_prob": buy_prob
                    })
                elif want_short and units >= 1e-6:
                    entry_price = exec_price * (1.0 - SLIPPAGE_PCT + SPREAD_PCT / 2.0)
                    position_units = units
                    position_side = "short"
                    entry_comm = entry_price * position_units * COMMISSION_PER_SIDE
                    cash -= entry_comm
                    sl_price = min(entry_price + stop_dist, entry_price * (1 + 0.1))
                    tp_price = entry_price - ATR_MULTIPLIER_TP * atr_now if ENABLE_TAKE_PROFIT else None
                    trail = entry_price + TRAILING_ATR_MULT * atr_now if ENABLE_TRAILING else None
                    trade_log.append({
                        "side": "short",
                        "entry_time": nxt["Date"],
                        "entry_price": entry_price,
                        "units": position_units,
                        "sl_price": sl_price,
                        "tp_price": tp_price,
                        "trail_stop": trail,
                        "entry_commission": entry_comm,
                        "entry_prob": buy_prob
                    })

            # MONITOR / EXIT using next bar OHLC (check intrabar TP/SL)
            if position_side is not None:
                # use current next bar OHLC to see if TP/SL hit
                high = nxt.get("High", exec_price)
                low = nxt.get("Low", exec_price)
                exited = False
                exit_price = None
                exit_reason = None

                last_trade = trade_log[-1]
                # Check TP first (if enabled)
                if last_trade.get("tp_price") is not None:
                    if position_side == "long" and high >= last_trade["tp_price"]:
                        exit_price = last_trade["tp_price"] * (1.0 - SLIPPAGE_PCT)
                        exit_reason = "TP"
                        exited = True
                    if position_side == "short" and low <= last_trade["tp_price"]:
                        exit_price = last_trade["tp_price"] * (1.0 + SLIPPAGE_PCT)
                        exit_reason = "TP"
                        exited = True

                # Check SL
                if not exited:
                    if position_side == "long" and low <= last_trade["sl_price"]:
                        exit_price = last_trade["sl_price"] * (1.0 - SLIPPAGE_PCT)
                        exit_reason = "SL"
                        exited = True
                    if position_side == "short" and high >= last_trade["sl_price"]:
                        exit_price = last_trade["sl_price"] * (1.0 + SLIPPAGE_PCT)
                        exit_reason = "SL"
                        exited = True

                # Trailing stop if enabled
                if not exited and ENABLE_TRAILING and last_trade.get("trail_stop") is not None:
                    if position_side == "long" and low <= last_trade["trail_stop"]:
                        exit_price = last_trade["trail_stop"] * (1.0 - SLIPPAGE_PCT)
                        exit_reason = "TRAIL"
                        exited = True
                    if position_side == "short" and high >= last_trade["trail_stop"]:
                        exit_price = last_trade["trail_stop"] * (1.0 + SLIPPAGE_PCT)
                        exit_reason = "TRAIL"
                        exited = True

                if exited:
                    if position_side == "long":
                        pnl = (exit_price - last_trade["entry_price"]) * position_units
                    else:
                        pnl = (last_trade["entry_price"] - exit_price) * position_units
                    exit_comm = exit_price * position_units * COMMISSION_PER_SIDE
                    cash += pnl - exit_comm
                    last_trade.update({
                        "exit_time": nxt["Date"],
                        "exit_price": exit_price,
                        "exit_commission": exit_comm,
                        "pnl": pnl,
                        "exit_reason": exit_reason
                    })
                    # reset position
                    position_side = None
                    position_units = 0.0
                    entry_price = None
                else:
                    # if still holding and trailing enabled, update trailing stop based on mark
                    if ENABLE_TRAILING and last_trade.get("trail_stop") is not None:
                        cur_mark = exec_price
                        if position_side == "long":
                            new_trail = max(last_trade["trail_stop"], cur_mark - TRAILING_ATR_MULT * atr_now)
                            last_trade["trail_stop"] = new_trail
                        else:
                            new_trail = min(last_trade["trail_stop"], cur_mark + TRAILING_ATR_MULT * atr_now)
                            last_trade["trail_stop"] = new_trail

            # mark-to-market equity
            unrealized = 0.0
            if position_side == "long":
                unrealized = (exec_price - trade_log[-1]["entry_price"]) * position_units
            elif position_side == "short":
                unrealized = (trade_log[-1]["entry_price"] - exec_price) * position_units
            equity = cash + unrealized
            equities.append(equity)

        # fill final equities length
        if len(equities) < len(df):
            equities += [equities[-1]] * (len(df) - len(equities))

        df["equity"] = equities
        df["strategy_ret"] = pd.Series(df["equity"]).pct_change().fillna(0.0)

        # compute freq
        if len(df) >= 2:
            delta = (df.loc[1, "Date"] - df.loc[0, "Date"]).total_seconds()
            freq = 24 * 252 if delta <= 4000 else (252 if delta <= 90000 else 12 * 252)
        else:
            freq = 252

        metrics = self._compute_metrics(pd.Series(df["equity"]), pd.Series(df["strategy_ret"]), freq_per_year=freq)

        # save logs + chart
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        trades_path = os.path.join(self.save_dir, f"trades_profitboost_v6_6_{ts}.csv")
        df_path = os.path.join(self.save_dir, f"backtest_profitboost_v6_6_{ts}.csv")
        chart_path = os.path.join(self.save_dir, f"equity_profitboost_v6_6_{ts}.png")
        try:
            pd.DataFrame(trade_log).to_csv(trades_path, index=False)
            df.to_csv(df_path, index=False)
            self._plot_equity(df["Date"], df["equity"], chart_path)
        except Exception as e:
            print("Warning saving outputs:", e)

        print(f"Trades saved → {trades_path}")
        print(f"Backtest CSV → {df_path}")
        print(f"Chart → {chart_path}")
        return metrics

    @staticmethod
    def _plot_equity(dates, equity, outpath):
        plt.figure(figsize=(12, 5))
        plt.plot(dates, equity, label="Equity")
        peak = pd.Series(equity).cummax()
        plt.fill_between(dates, equity, peak, where=(pd.Series(equity) < peak), alpha=0.15)
        plt.title("ProfitBoost v6.6 Equity")
        plt.xlabel("Date")
        plt.ylabel("Equity")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(outpath)
        plt.close()

    @staticmethod
    def _compute_metrics(equity_series: pd.Series, strategy_rets: pd.Series, freq_per_year: int = 252):
        # safe metrics (avoid invalid power)
        total_return = float(equity_series.iloc[-1] / equity_series.iloc[0] - 1.0) if len(equity_series) > 1 else 0.0
        years = len(strategy_rets) / freq_per_year if freq_per_year > 0 else np.nan
        try:
            cagr = (equity_series.iloc[-1] / equity_series.iloc[0]) ** (1 / years) - 1 if years > 0 else np.nan
        except Exception:
            cagr = None
        mean_ret = float(strategy_rets.mean()) if len(strategy_rets) > 0 else 0.0
        vol = float(strategy_rets.std()) if len(strategy_rets) > 0 else np.nan
        ann_return = (1 + mean_ret) ** freq_per_year - 1 if not np.isnan(mean_ret) else np.nan
        ann_vol = vol * np.sqrt(freq_per_year) if not np.isnan(vol) else np.nan
        sharpe = (mean_ret / vol) * np.sqrt(freq_per_year) if vol and vol > 0 else np.nan
        downside = float(strategy_rets[strategy_rets < 0].std()) if (strategy_rets < 0).any() else np.nan
        sortino = (mean_ret / downside) * np.sqrt(freq_per_year) if downside and downside > 0 else np.nan
        max_dd = float((equity_series / equity_series.cummax() - 1.0).min())
        win_rate = float((strategy_rets > 0).mean()) if len(strategy_rets) > 0 else 0.0
        return {
            "total_return": total_return,
            "cagr": float(cagr) if cagr is not None else None,
            "annualized_return": ann_return,
            "annualized_vol": ann_vol,
            "sharpe": sharpe,
            "sortino": sortino,
            "max_drawdown": max_dd,
            "win_rate": win_rate
        }
