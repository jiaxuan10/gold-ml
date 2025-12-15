#!/usr/bin/env python3
"""
strategy_v6_6_stable.py
-----------------------
Profit-Boost v6.6 (Stable) - STRICT SYNC with Live Engine

Changes:
- REMOVED 'vol_scale' logic (Position sizing now matches inference_service.py exactly).
- Kept Zero Fees (Commission/Slippage = 0.0).
"""
import os
import pickle
from datetime import datetime
from typing import Dict, Optional, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------
# CONFIG (Optimized for Out-of-Sample Profit)
# -------------------------
INITIAL_CAPITAL = 10000.0
BASE_RISK_PER_TRADE_FRAC = 0.05
MAX_POSITION_FRAC = 0.9

ATR_MULTIPLIER_SL = 3.0  

ENABLE_TAKE_PROFIT = True       
ATR_MULTIPLIER_TP = 4.5  

BUY_PROB_DEFAULT = 0.52

ATR_WINDOW = 14
ENABLE_TRAILING = False          
TRAILING_ATR_MULT = 1.2
MIN_STOP_PCT = 0.0005
VOL_WINDOW = 24
VOL_THRESHOLD_PCTL = 0.995

# Zero fees for FYP
COMMISSION_PER_SIDE = 0.0
SLIPPAGE_PCT = 0.0
SPREAD_PCT = 0.0

REGIME_WEIGHTS = {"bull": 1.2, "neutral": 1.0, "bear": 0.9}
ENABLE_SHORTS = False

# -------------------------
class ProfitBoostStrategy:
    def __init__(self, model_meta: Dict, save_dir: str, shift_features: bool = True):
        self.model_meta = model_meta
        self.save_dir = save_dir
        self.shift_features = bool(shift_features)
        os.makedirs(save_dir, exist_ok=True)

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
        X_df = X_df.fillna(X_df.median().fillna(0.0)).ffill().bfill().fillna(0.0)
        X_df = X_df.clip(lower=-1e9, upper=1e9)
        return X_df

    def model_predict_proba(self, X_df: pd.DataFrame) -> Optional[np.ndarray]:
        m = self.model_meta.get("calibrated_model") or self.model_meta.get("raw_ensemble") or self.model_meta.get("model")
        if m is None: return None
        try:
            X_arr = X_df.values
            proba = m.predict_proba(X_arr)
            return np.asarray(proba)
        except Exception: return None

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

    def simulate(self, df: pd.DataFrame) -> Dict:
        if "Close" not in df.columns and "GOLD_Close" in df.columns:
            # 创建标准列名的副本（不删除原始列）
            df["Close"] = df["GOLD_Close"]
            df["Open"] = df.get("GOLD_Open", df["Close"])
            df["High"] = df.get("GOLD_High", df["Close"])
            df["Low"] = df.get("GOLD_Low", df["Close"])
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

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

        # buy_thr = float(self.model_meta.get("threshold", BUY_PROB_DEFAULT))
        buy_thr = float(BUY_PROB_DEFAULT)
        vol_cutoff = df["vol_24h"].quantile(VOL_THRESHOLD_PCTL)

        initial_capital = float(INITIAL_CAPITAL)
        cash = initial_capital
        position_units = 0.0
        position_side = None
        equities = []
        trade_log = []

        for i in range(0, len(df) - 1):
            cur = df.loc[i]
            nxt = df.loc[i + 1]
            exec_price = nxt["Open"] if not pd.isna(nxt["Open"]) else nxt["Close"]
            buy_prob = float(cur.get("buy_prob", BUY_PROB_DEFAULT))
            
            vol_now = float(cur.get("vol_24h", cur.get("Volatility_20", 0.0)))

            atr_now = float(cur.get("ATR_14", cur.get("ATR", 0.0)))

            if "momentum_ok" in cur:
                momentum_ok = bool(cur["momentum_ok"])
            else:
                sma20 = float(cur.get("SMA_20", 0))
                sma50 = float(cur.get("SMA_50", 0))
                if sma20 > 0 and sma50 > 0:
                    momentum_ok = (sma20 > sma50)
                else:
                    momentum_ok = True 
            
            # -----------------------------------------------------------

            regime = cur.get("regime", "neutral")
            rweight = REGIME_WEIGHTS.get(regime, 1.0)

            if atr_now <= 0: atr_now = exec_price * 0.01 # 兜底 1% 波动
            stop_dist = max(atr_now * ATR_MULTIPLIER_SL, exec_price * MIN_STOP_PCT)
            
            model_signal = 1 if buy_prob > buy_thr else 0
            # want_long = (model_signal == 1) and momentum_ok and (vol_now <= vol_cutoff)
            want_long = (model_signal == 1)
            
            # 1. EXIT LOGIC
            if position_side == "long":
                last_trade = trade_log[-1]
                exited = False
                exit_price = None
                exit_reason = None
                
                high = nxt.get("High", exec_price)
                low = nxt.get("Low", exec_price)
                
                # A. Check SL/TP
                if low <= last_trade["sl_price"]:
                    exit_price = last_trade["sl_price"] * (1.0 - SLIPPAGE_PCT)
                    exit_reason = "SL"
                    exited = True
                elif ENABLE_TAKE_PROFIT and high >= last_trade["tp_price"]:
                    exit_price = last_trade["tp_price"] * (1.0 - SLIPPAGE_PCT)
                    exit_reason = "TP"
                    exited = True
                
                # B. Check Strategy Exit
                if not exited and not want_long:
                    exit_price = exec_price * (1.0 - SLIPPAGE_PCT)
                    exit_reason = "Strategy Exit"
                    exited = True
                
                if exited:
                    pnl = (exit_price - last_trade["entry_price"]) * position_units
                    exit_comm = exit_price * position_units * COMMISSION_PER_SIDE
                    cash += pnl - exit_comm
                    last_trade.update({
                        "exit_time": nxt["Date"],
                        "exit_price": exit_price,
                        "exit_commission": exit_comm,
                        "pnl": pnl,
                        "exit_reason": exit_reason
                    })
                    position_side = None
                    position_units = 0.0

            # 2. ENTRY LOGIC
            if position_side is None:
                if want_long:
                    # 🔥🔥🔥 Sizing (Synced with Live Engine) 🔥🔥🔥
                    # Removed 'vol_scale' to match inference_service.py
                    adaptive_risk_amt = initial_capital * BASE_RISK_PER_TRADE_FRAC * rweight
                    desired_units = adaptive_risk_amt / (stop_dist + 1e-12)
                    max_units = (initial_capital * MAX_POSITION_FRAC) / (exec_price + 1e-12)
                    units = float(np.clip(desired_units, 0.0, max_units))
                    
                    if units >= 1e-6:
                        entry_price = exec_price * (1.0 + SLIPPAGE_PCT + SPREAD_PCT / 2.0)
                        position_units = units
                        position_side = "long"
                        entry_comm = entry_price * position_units * COMMISSION_PER_SIDE
                        cash -= entry_comm
                        
                        sl_price = max(entry_price - stop_dist, entry_price * (1 - 0.1))
                        tp_price = entry_price + ATR_MULTIPLIER_TP * atr_now if ENABLE_TAKE_PROFIT else 999999.0
                        
                        trade_log.append({
                            "side": "long",
                            "entry_time": nxt["Date"],
                            "entry_price": entry_price,
                            "units": position_units,
                            "sl_price": sl_price,
                            "tp_price": tp_price,
                            "entry_commission": entry_comm,
                            "entry_prob": buy_prob
                        })

            # Equity Curve
            unrealized = 0.0
            if position_side == "long":
                unrealized = (exec_price - trade_log[-1]["entry_price"]) * position_units
            equity = cash + unrealized
            equities.append(equity)

        if len(equities) < len(df):
            equities += [equities[-1]] * (len(df) - len(equities))

        df["equity"] = equities
        df["strategy_ret"] = pd.Series(df["equity"]).pct_change().fillna(0.0)

        if len(df) >= 2:
            delta = (df.loc[1, "Date"] - df.loc[0, "Date"]).total_seconds()
            freq = 24 * 252 if delta <= 4000 else 252
        else:
            freq = 252

        metrics = self._compute_metrics(pd.Series(df["equity"]), pd.Series(df["strategy_ret"]), trade_log, freq_per_year=freq)

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
        return metrics

    @staticmethod
    def _plot_equity(dates, equity, outpath):
        plt.figure(figsize=(12, 5))
        plt.plot(dates, equity, label="Equity")
        plt.title("ProfitBoost v6.6 Equity (Zero Fee)")
        plt.grid(alpha=0.3)
        plt.savefig(outpath)
        plt.close()

    @staticmethod
    def _compute_metrics(equity_series: pd.Series, strategy_rets: pd.Series, trade_log: List, freq_per_year: int = 252):
        """
        Calculates metrics including Real Trade Win Rate.
        """
        total_return = float(equity_series.iloc[-1] / equity_series.iloc[0] - 1.0) if len(equity_series) > 1 else 0.0
        mean_ret = float(strategy_rets.mean()) if len(strategy_rets) > 0 else 0.0
        vol = float(strategy_rets.std()) if len(strategy_rets) > 0 else np.nan
        sharpe = (mean_ret / vol) * np.sqrt(freq_per_year) if vol and vol > 0 else np.nan
        max_dd = float((equity_series / equity_series.cummax() - 1.0).min())
        
        # --- Real Trade Win Rate ---
        closed_trades = [t for t in trade_log if t.get("pnl") is not None]
        if len(closed_trades) > 0:
            winning_trades = [t for t in closed_trades if t["pnl"] > 0]
            win_rate = len(winning_trades) / len(closed_trades)
        else:
            win_rate = 0.0
            
        return {
            "total_return": total_return,
            "sharpe": sharpe,
            "max_drawdown": max_dd,
            "win_rate": win_rate,
            "total_trades": len(closed_trades)
        }