# streamlit_app/app.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os
import json
import time
from plotly.subplots import make_subplots

# --- PAGE CONFIG ---
st.set_page_config(page_title="Gold AI Scalper", layout="wide", page_icon="⚡")

# --- PATHS ---
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(ROOT, "data", "final")

LATEST_CSV = os.path.join(DATA_DIR, "latest_hour_features.csv")
PRED_JSON = os.path.join(DATA_DIR, "latest_prediction.json")
PRED_LOG = os.path.join(DATA_DIR, "prediction_log.csv")
TRADE_LOG = os.path.join(DATA_DIR, "trade_log.csv")
PORTFOLIO_FILE = os.path.join(DATA_DIR, "portfolio_state.json")

# --- CSS STYLING ---
st.markdown("""
<style>
    .metric-card {background-color: #1e1e1e; padding: 15px; border-radius: 10px; border: 1px solid #333;}
    .buy-signal {color: #00ff00; font-weight: bold;}
    .sell-signal {color: #ff0044; font-weight: bold;}
</style>
""", unsafe_allow_html=True)

st.title("⚡ Realtime AI Gold Auto-Trader")

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ System Control")
    refresh_rate = st.slider("Refresh Rate (s)", 2, 60, 5)
    ma_len = st.number_input("MA Filter", value=20)
    st.info("Data Source: Yahoo Finance (1h)\nModel: Ensemble (RF+GB+MLP)")
    
    if st.button("Reload All Data"):
        st.cache_data.clear()
        st.rerun()

# --- AUTO REFRESH ---
from streamlit_autorefresh import st_autorefresh
st_autorefresh(interval=refresh_rate * 1000, key="datarefresh")

# --- DATA LOADERS ---
def load_data():
    if not os.path.exists(LATEST_CSV): return pd.DataFrame()
    df = pd.read_csv(LATEST_CSV)
    df["Date"] = pd.to_datetime(df["Date"])
    # Convert to KL Time
    df["Date"] = df["Date"].dt.tz_convert("Asia/Kuala_Lumpur")
    return df.sort_values("Date")

def load_json(path):
    if os.path.exists(path):
        try:
            with open(path, "r") as f: return json.load(f)
        except: return {}
    return {}

def load_trades():
    if os.path.exists(TRADE_LOG):
        df = pd.read_csv(TRADE_LOG)
        df["Date"] = pd.to_datetime(df["Date"])
        # Fix timezones if mixed
        if df["Date"].dt.tz is None:
             df["Date"] = df["Date"].dt.tz_localize("UTC").dt.tz_convert("Asia/Kuala_Lumpur")
        else:
             df["Date"] = df["Date"].dt.tz_convert("Asia/Kuala_Lumpur")
        return df
    return pd.DataFrame()

# --- LOAD ASSETS ---
df = load_data()
pred = load_json(PRED_JSON)
portfolio = load_json(PORTFOLIO_FILE)
trades = load_trades()

# --- MAIN DASHBOARD ---
if df.empty:
    st.warning("⚠️ Waiting for Data Pipeline to initialize...")
    st.stop()

# 1. KPI ROW
last_price = df.iloc[-1]["GOLD_Close"]
prev_price = df.iloc[-2]["GOLD_Close"]
price_chg = last_price - prev_price

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Gold Price (USD)", f"${last_price:,.2f}", f"{price_chg:.2f}")

with col2:
    prob = pred.get("probability", 0)
    signal = pred.get("signal", 0)
    color = "normal"
    if prob > 0.6: color = "normal" 
    st.metric("AI Confidence (Buy)", f"{prob:.1%}", 
              delta="BUY SIGNAL" if signal==1 else "WAIT/SELL", 
              delta_color="normal" if signal==0 else "inverse")

with col3:
    equity = portfolio.get("equity", 10000)
    bal = portfolio.get("balance", 10000)
    pnl_pct = (equity - 10000) / 10000 * 100
    st.metric("Simulated Equity", f"${equity:,.2f}", f"{pnl_pct:.2f}%")

with col4:
    pos = portfolio.get("position", 0)
    st.metric("Current Position", f"{pos} Units", f"Entry: {portfolio.get('entry_price', 0):.2f}" if pos > 0 else "Flat")

# 2. CHART ROW
st.subheader("📈 Realtime Market & AI Signals")

# Create Subplots (Price on Top, Probabilities on Bottom)
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                    vertical_spacing=0.03, row_heights=[0.7, 0.3])

# Candlestick
fig.add_trace(go.Candlestick(x=df["Date"], open=df["GOLD_Open"], high=df["GOLD_High"],
                             low=df["GOLD_Low"], close=df["GOLD_Close"], name="Gold"), row=1, col=1)

# MA
df["MA"] = df["GOLD_Close"].rolling(ma_len).mean()
fig.add_trace(go.Scatter(x=df["Date"], y=df["MA"], line=dict(color='orange', width=1), name=f"SMA {ma_len}"), row=1, col=1)

# Trade Markers
if not trades.empty:
    buys = trades[trades["Side"] == "BUY"]
    sells = trades[trades["Side"] == "SELL"]
    
    fig.add_trace(go.Scatter(x=buys["Date"], y=buys["Price"], mode='markers', 
                             marker=dict(symbol='triangle-up', size=12, color='green'), name="Buy Exec"), row=1, col=1)
    fig.add_trace(go.Scatter(x=sells["Date"], y=sells["Price"], mode='markers', 
                             marker=dict(symbol='triangle-down', size=12, color='red'), name="Sell Exec"), row=1, col=1)

# Probability Heatmap (Bottom Chart)
# Use prediction log for history
if os.path.exists(PRED_LOG):
    pred_hist = pd.read_csv(PRED_LOG)
    pred_hist["Date"] = pd.to_datetime(pred_hist["Date"]).dt.tz_convert("Asia/Kuala_Lumpur")
    # Align with df dates
    merged = pd.merge_asof(df[["Date"]], pred_hist.sort_values("Date"), on="Date")
    
    fig.add_trace(go.Scatter(x=merged["Date"], y=merged["probability"], 
                             fill='tozeroy', line=dict(color='#00ccff', width=1), name="AI Buy Prob"), row=2, col=1)
    # Threshold line
    fig.add_hline(y=pred.get("threshold", 0.5), line_dash="dot", row=2, col=1, annotation_text="Threshold")

fig.update_layout(height=600, margin=dict(l=0, r=0, t=0, b=0), template="plotly_dark")
fig.update_yaxes(range=[0, 1], row=2, col=1)
st.plotly_chart(fig, use_container_width=True)

# 3. LOGS ROW
c1, c2 = st.columns(2)
with c1:
    st.subheader("📜 Trade Log")
    if not trades.empty:
        st.dataframe(trades.sort_values("Date", ascending=False).head(10), hide_index=True, use_container_width=True)
    else:
        st.info("No trades executed yet.")

with c2:
    st.subheader("🧠 AI Feature Analysis (Latest)")
    # Show top features from latest row
    ignore = ["Date", "GOLD_Open", "GOLD_High", "GOLD_Low", "GOLD_Close", "GOLD_Volume", "Return"]
    feat_data = df.iloc[-1].drop(labels=ignore, errors='ignore')
    feat_df = pd.DataFrame(feat_data).reset_index()
    feat_df.columns = ["Feature", "Value"]
    feat_df["Value"] = pd.to_numeric(feat_df["Value"], errors='coerce')
    st.dataframe(feat_df.dropna().style.background_gradient(cmap="viridis"), use_container_width=True, height=300)