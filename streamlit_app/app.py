# streamlit_app/app.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import json
from datetime import datetime, timedelta

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Gold AI Quant System",
    layout="wide",
    page_icon="🏆",
    initial_sidebar_state="collapsed"
)

# --- CONFIG ---
REFRESH_RATE = 5   
MA_PERIOD = 20     

# --- PATHS ---
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(ROOT, "data", "final")
LATEST_CSV = os.path.join(DATA_DIR, "latest_hour_features.csv")
PRED_JSON = os.path.join(DATA_DIR, "latest_prediction.json")
SENTIMENT_JSON = os.path.join(DATA_DIR, "current_sentiment.json")
NEWS_LIST_CSV = os.path.join(DATA_DIR, "latest_news_headlines.csv") # 🆕 新闻列表
PRED_LOG = os.path.join(DATA_DIR, "prediction_log.csv")
TRADE_LOG = os.path.join(DATA_DIR, "trade_log.csv")
PORTFOLIO_FILE = os.path.join(DATA_DIR, "portfolio_state.json")

# --- CUSTOM CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&display=swap');
    html, body, [class*="css"] { font-family: 'JetBrains Mono', monospace; }
    div[data-testid="stMetric"] {
        background-color: #161b22; border: 1px solid #30363d; padding: 15px;
        border-radius: 8px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    div[data-testid="stMetric"]:hover { border-color: #58a6ff; }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] { background-color: #0d1117; border-radius: 4px 4px 0 0; color: #8b949e; }
    .stTabs [aria-selected="true"] { background-color: #161b22; color: #58a6ff; border-color: #30363d; border-bottom: #161b22; }
</style>
""", unsafe_allow_html=True)

# --- HEADER ---
c_title, c_status = st.columns([3, 1])
with c_title: st.title("🏆 Gold AI Quant System")
with c_status:
    st.markdown("### 🟢 SYSTEM ONLINE")
    st.caption(f"Refresh: {REFRESH_RATE}s | Mode: LIVE")

from streamlit_autorefresh import st_autorefresh
st_autorefresh(interval=REFRESH_RATE * 1000, key="datarefresh")

# --- DATA LOADERS ---
def load_data():
    if not os.path.exists(LATEST_CSV): return pd.DataFrame()
    try:
        df = pd.read_csv(LATEST_CSV)
        df["Date"] = pd.to_datetime(df["Date"], utc=True)
        return df.sort_values("Date")
    except: return pd.DataFrame()

def load_json(path):
    if os.path.exists(path):
        try:
            with open(path, "r") as f: return json.load(f)
        except: return {}
    return {}

def load_csv(path):
    if os.path.exists(path):
        try: return pd.read_csv(path)
        except: pass
    return pd.DataFrame()

# --- MAIN LOGIC ---
df = load_data()
pred = load_json(PRED_JSON)
sent = load_json(SENTIMENT_JSON)
news_list = load_csv(NEWS_LIST_CSV) 
portfolio = load_json(PORTFOLIO_FILE)
trades = load_csv(TRADE_LOG)

if df.empty:
    st.info("⏳ Waiting for data stream...")
    st.stop()

# --- 1. KPI DASHBOARD ---
last_price = df.iloc[-1]["GOLD_Close"]
prev_price = df.iloc[-2]["GOLD_Close"] if len(df) > 1 else last_price
price_chg = last_price - prev_price

balance = portfolio.get("balance", 10000)
equity = portfolio.get("equity", 10000)
pos_size = portfolio.get("position", 0)
entry_price = portfolio.get("entry_price", 0)
sl_price = portfolio.get("sl", 0)
tp_price = portfolio.get("tp", 0)
unrealized_pnl = (last_price - entry_price) * pos_size if pos_size > 0 else 0
total_return = (equity - 10000) / 10000 * 100

# AI & News Logic
ai_prob = pred.get('probability', 0)
final_thr = pred.get('final_threshold', 0.5)
news_score = sent.get("sentiment_score", 0.0)
news_status = sent.get("status", "Neutral")

news_color = "normal"
if news_score > 0.1: news_color = "normal" 
elif news_score < -0.1: news_color = "inverse" 

# Layout: 5 Columns
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("💛 Gold Price", f"${last_price:,.2f}", f"{price_chg:.2f}")
m2.metric("🤖 AI Confidence", f"{ai_prob:.1%}", f"Thr: {final_thr:.2f}")
m3.metric("📰 News Sentiment", f"{news_score:.2f}", news_status, delta_color=news_color)
m4.metric("💰 Net Equity", f"${equity:,.2f}", f"{total_return:.2f}%")
m5.metric("📊 Position", f"{pos_size:.4f} oz", f"${unrealized_pnl:.2f}" if pos_size > 0 else "FLAT")

# --- 2. LIVE TRADE MONITOR ---
if pos_size > 0:
    st.markdown("### 🔥 Live Trade Monitor")
    c1, c2 = st.columns([1, 3])
    with c1: st.info(f"**Entry:** ${entry_price:.2f}")
    with c2:
        if tp_price > sl_price:
            prog = min(max((last_price - sl_price) / (tp_price - sl_price), 0.0), 1.0)
            st.write(f"Risk/Reward (SL: {sl_price:.1f} | TP: {tp_price:.1f})")
            st.progress(prog)

# --- 3. CHARTING ---
st.markdown("### 📈 Market Analysis")
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.75, 0.25])
fig.add_trace(go.Candlestick(x=df["Date"], open=df["GOLD_Open"], high=df["GOLD_High"], low=df["GOLD_Low"], close=df["GOLD_Close"], name="Price"), row=1, col=1)
df["MA20"] = df["GOLD_Close"].rolling(MA_PERIOD).mean()
fig.add_trace(go.Scatter(x=df["Date"], y=df["MA20"], line=dict(color='#FFA500', width=1), name="SMA 20"), row=1, col=1)

if not trades.empty:
    trades["Date"] = pd.to_datetime(trades["Date"])
    buys = trades[trades["Side"] == "BUY"]
    sells = trades[trades["Side"].isin(["SELL", "Stop Loss", "Take Profit"])]
    if not buys.empty:
        fig.add_trace(go.Scatter(x=buys["Date"], y=buys["Price"], mode='markers', marker=dict(symbol='triangle-up', size=10, color='#00FF00'), name="Buy"), row=1, col=1)
    if not sells.empty:
        fig.add_trace(go.Scatter(x=sells["Date"], y=sells["Price"], mode='markers', marker=dict(symbol='triangle-down', size=10, color='#FF4444'), name="Sell"), row=1, col=1)

# AI Prob Chart
if os.path.exists(PRED_LOG):
    try:
        ph = pd.read_csv(PRED_LOG)
        ph["Date"] = pd.to_datetime(ph["Date"], utc=True)
        m = pd.merge_asof(df[["Date"]], ph.sort_values("Date"), on="Date")
        fig.add_trace(go.Scatter(x=m["Date"], y=m["probability"], fill='tozeroy', line=dict(color='#9932CC'), name="AI Prob"), row=2, col=1)
        # 优先使用 final_threshold
        fig.add_trace(go.Scatter(x=m["Date"], y=m.get("final_threshold", m.get("threshold", 0.54)), line=dict(color='white', dash='dash'), name="Threshold"), row=2, col=1)
    except: pass

fig.update_layout(height=500, margin=dict(l=10, r=10, t=10, b=10), template="plotly_dark", xaxis_rangeslider_visible=False)
st.plotly_chart(fig, use_container_width=True)

# --- 4. LOGS & NEWS & ANALYSIS ---
# 恢复了 4 个 Tab
t1, t2, t3, t4 = st.tabs(["📜 Trade Log", "📰 Live News", "🧠 AI Analysis", "🔢 Raw Data"])

with t1:
    if not trades.empty:
        show_all = st.checkbox("Show Wait/Hold Logs", False)
        d = trades if show_all else trades[~trades["Side"].isin(["WAIT", "HOLD"])]
        st.dataframe(d.sort_values("Date", ascending=False), use_container_width=True, hide_index=True)
    else: st.info("No trades yet.")

with t2:
    if not news_list.empty:
        st.caption("Latest FinBERT Analysis (Past 24h)")
        def highlight_sentiment(val):
            if val == 'Positive': return 'color: #00FF00'
            elif val == 'Negative': return 'color: #FF4444'
            return ''
        st.dataframe(
            news_list.style.map(highlight_sentiment, subset=['Label']),
            use_container_width=True,
            hide_index=True,
            column_config={
                "URL": st.column_config.LinkColumn("Read Article"),
                "Score": st.column_config.ProgressColumn("Sentiment Score", min_value=-1, max_value=1)
            }
        )
    else: st.info("No news fetched yet. Waiting for next cycle...")

with t3:
    # === 仪表盘 UI 优化版 ===
    col_gauge, col_factors = st.columns([1, 1.5]) # 调整比例让表格更宽一点
    
    if not df.empty:
        latest = df.iloc[-1]
        curr_prob = pred.get('probability', 0.5)
        curr_thr = pred.get('final_threshold', 0.54)

        with col_gauge:
            st.markdown("#### 🤖 Confidence Gauge")
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = curr_prob * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                delta = {'reference': curr_thr * 100, 'increasing': {'color': "green"}},
                gauge = {
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white"},
                    'bar': {'color': "#9932CC"}, 
                    'bgcolor': "rgba(0,0,0,0)", # 透明背景
                    'borderwidth': 2,
                    'bordercolor': "#333",
                    'steps': [
                        {'range': [0, 40], 'color': 'rgba(255, 0, 0, 0.2)'},
                        {'range': [40, 60], 'color': 'rgba(128, 128, 128, 0.2)'},
                        {'range': [60, 100], 'color': 'rgba(0, 255, 0, 0.2)'}
                    ],
                    'threshold': {
                        'line': {'color': "yellow", 'width': 4},
                        'thickness': 0.75,
                        'value': curr_thr * 100
                    }
                }
            ))
            # 调整 margin 消除空白
            fig_gauge.update_layout(height=250, margin=dict(l=30, r=30, t=30, b=30), paper_bgcolor="rgba(0,0,0,0)", font={'color': "white"})
            st.plotly_chart(fig_gauge, use_container_width=True)
            
            # 信号文字
            if curr_prob > curr_thr:
                st.success(f"✅ **BUY SIGNAL**")
            else:
                st.warning(f"🛑 **WAIT** (Conf < Thr)")

        with col_factors:
            st.markdown("#### 🔍 Factor Analysis")
            
            factors = []
            # RSI Logic
            rsi = latest.get("RSI_14", 50)
            rsi_status = "Bullish" if rsi < 30 else "Bearish" if rsi > 70 else "Neutral"
            factors.append({"Factor": "RSI (14)", "Value": f"{rsi:.1f}", "Signal": rsi_status})
            
            # Trend Logic
            sma20 = latest.get("SMA_20", 0)
            sma50 = latest.get("SMA_50", 0)
            trend = "Bullish" if sma20 > sma50 else "Bearish"
            factors.append({"Factor": "Trend (SMA)", "Value": "Golden Cross" if sma20>sma50 else "Death Cross", "Signal": trend})
            
            # Volatility Logic
            vol = latest.get("Volatility_20", 0)
            vol_status = "High Risk" if vol > 0.002 else "Stable"
            factors.append({"Factor": "Volatility", "Value": f"{vol:.4f}", "Signal": vol_status})
            
            # News Logic
            sent_impact = pred.get('sentiment_impact', 'None')
            sent_signal = "Bearish" if "Bad" in sent_impact or "Bear" in sent_impact else "Bullish" if "Good" in sent_impact else "Neutral"
            factors.append({"Factor": "News Impact", "Value": sent_impact, "Signal": sent_signal})

            def color_signal(val):
                if "Bull" in val: return 'color: #00FF00; font-weight: bold'
                if "Bear" in val or "High" in val: return 'color: #FF4444; font-weight: bold'
                return 'color: #888'

            df_factors = pd.DataFrame(factors)
            st.dataframe(
                df_factors.style.map(color_signal, subset=['Signal']),
                use_container_width=True,
                hide_index=True
            )

with t4:
    st.dataframe(df.tail(10), use_container_width=True)