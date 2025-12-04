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
    page_title="Gold ML Auto-Trader",
    layout="wide",
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
    
    /* Global Font */
    html, body, [class*="css"] { 
        font-family: 'JetBrains Mono', monospace; 
    }

    /* 1. Make Tabs Bigger */
    button[data-baseweb="tab"] {
        font-size: 24px !important;   /* Increase tab text size */
        font-weight: bold !important;
        padding: 10px 20px !important; /* Make tabs wider/taller */
    }
    
    /* 2. Make Metric Labels Bigger (Gold Price, AI Confidence...) */
    div[data-testid="stMetricLabel"] {
        font-size: 18px !important;   /* Label size */
        font-weight: bold !important;
        color: #8b949e !important;    /* Light gray text */
    }
    
    /* 3. Make Metric Values Bigger ($2,650.50...) */
    div[data-testid="stMetricValue"] {
        font-size: 32px !important;   /* Value size */
        color: #ffffff !important;    /* White text */
    }

    /* Metric Card Box Style */
    div[data-testid="stMetric"] {
        background-color: #161b22; 
        border: 1px solid #30363d; 
        padding: 15px;
        border-radius: 8px; 
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
    }
    div[data-testid="stMetric"]:hover { 
        transform: translateY(-2px);
        border-color: #58a6ff; 
    }
    
    /* Tab Container Style */
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] { 
        background-color: #0d1117; 
        border-radius: 4px 4px 0 0; 
        color: #8b949e; 
        border: 1px solid transparent;
    }
    .stTabs [aria-selected="true"] { 
        background-color: #161b22; 
        color: #58a6ff; 
        border-color: #30363d; 
        border-bottom-color: #161b22; 
    }
</style>
""", unsafe_allow_html=True)

# --- HEADER ---
c_title, c_status = st.columns([3, 1])
with c_title: st.title("Gold ML Auto-Trader Dashboard")
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
    st.info("Waiting for data stream...")
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
m1.metric("Gold Price", f"${last_price:,.2f}", f"{price_chg:.2f}")
m2.metric("AI Confidence", f"{ai_prob:.3%}", f"Thr: {final_thr:.3f}")
m3.metric("News Sentiment", f"{news_score:.3f}", news_status, delta_color=news_color)
m4.metric("Net Equity", f"${equity:,.2f}", f"{total_return:.2f}%")
m5.metric("Position", f"{pos_size:.4f} oz", f"${unrealized_pnl:.2f}" if pos_size > 0 else "FLAT")

# --- 2. LIVE TRADE MONITOR ---
if pos_size > 0:
    st.markdown("### Live Trade Monitor")
    c1, c2 = st.columns([1, 3])
    with c1: st.info(f"**Entry:** ${entry_price:.2f}")
    with c2:
        if tp_price > sl_price:
            prog = min(max((last_price - sl_price) / (tp_price - sl_price), 0.0), 1.0)
            st.write(f"Risk/Reward (SL: {sl_price:.1f} | TP: {tp_price:.1f})")
            st.progress(prog)
# --- 3. CHARTING ---
st.markdown("### Market Analysis")
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.75, 0.25])

# 1. K线图
fig.add_trace(go.Candlestick(x=df["Date"], open=df["GOLD_Open"], high=df["GOLD_High"], low=df["GOLD_Low"], close=df["GOLD_Close"], name="Price"), row=1, col=1)

# 2. SMA 20 (快线 - 橙色)
df["MA20"] = df["GOLD_Close"].rolling(MA_PERIOD).mean()
fig.add_trace(go.Scatter(x=df["Date"], y=df["MA20"], line=dict(color='#FFA500', width=1), name="SMA 20"), row=1, col=1)

# 3. SMA 50 (慢线 - 蓝色) 🔥🔥🔥【新增部分】
df["MA50"] = df["GOLD_Close"].rolling(50).mean()
fig.add_trace(go.Scatter(x=df["Date"], y=df["MA50"], line=dict(color='#00BFFF', width=1), name="SMA 50"), row=1, col=1)

# 4. 买卖点标记
if not trades.empty:
    trades["Date"] = pd.to_datetime(trades["Date"])
    buys = trades[trades["Side"] == "BUY"]
    sells = trades[trades["Side"].isin(["SELL", "Stop Loss", "Take Profit"])]
    if not buys.empty:
        fig.add_trace(go.Scatter(x=buys["Date"], y=buys["Price"], mode='markers', marker=dict(symbol='triangle-up', size=10, color='#00FF00'), name="Buy"), row=1, col=1)
    if not sells.empty:
        fig.add_trace(go.Scatter(x=sells["Date"], y=sells["Price"], mode='markers', marker=dict(symbol='triangle-down', size=10, color='#FF4444'), name="Sell"), row=1, col=1)

# ... (后面的 AI Prob Chart 和 update_layout 保持不变) ...
# 如果你需要完整的后半部分代码来复制，请告诉我。
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
# 1. 设置布局 (原有的)
fig.update_layout(
    height=500, 
    margin=dict(l=10, r=10, t=10, b=10), 
    template="plotly_dark", 
    xaxis_rangeslider_visible=False
)

# 这会让 Plotly 强制把周五和周一拼在一起，不留空隙
fig.update_xaxes(
    rangebreaks=[
        dict(bounds=["sat", "mon"]), # 隐藏 周六 到 周一 之间的空档
    ]
)

st.plotly_chart(fig, use_container_width=True)

# --- 4. LOGS & NEWS & ANALYSIS ---
# 恢复了 4 个 Tab
t1, t2, t3, t4, t5 = st.tabs(["Trade Log", "Live News", "AI Analysis", "Raw Data", "Model Insights"])
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

# ---------------------------------------------------------
# 替换 app.py 中的 with t3: 部分
# ---------------------------------------------------------

with t3:
    st.markdown("### AI Decision Core (XGBoost/Ensemble)")
    
    if not df.empty:
        latest = df.iloc[-1]
        curr_prob = pred.get('probability', 0.5)
        curr_thr = pred.get('final_threshold', 0.54)

        # === 第一行：仪表盘 + 关键因子 ===
        c1, c2 = st.columns([1, 1.5])
        
        with c1:
            # 1. 仪表盘
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = curr_prob * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                delta = {'reference': curr_thr * 100, 'increasing': {'color': "green"}},
                gauge = {
                    'axis': {'range': [0, 100], 'tickwidth': 1},
                    'bar': {'color': "#9932CC"}, 
                    'bgcolor': "rgba(0,0,0,0)",
                    'borderwidth': 2,
                    'bordercolor': "#333",
                    'steps': [
                        {'range': [0, 40], 'color': 'rgba(255, 0, 0, 0.2)'},
                        {'range': [40, 60], 'color': 'rgba(128, 128, 128, 0.2)'},
                        {'range': [60, 100], 'color': 'rgba(0, 255, 0, 0.2)'}
                    ],
                    'threshold': {'line': {'color': "yellow", 'width': 4}, 'thickness': 0.75, 'value': curr_thr * 100}
                }
            ))
            fig_gauge.update_layout(height=220, margin=dict(l=20, r=20, t=20, b=20), paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_gauge, use_container_width=True)
            
            if curr_prob > curr_thr:
                st.success(f"✅ **BUY SIGNAL**")
            else:
                st.warning(f"🛑 **WAIT** (Conf < Thr)")

        with c2:
            st.markdown("#### Factor Analysis")
            
            factors = []
            
            # 1. RSI
            rsi = latest.get("RSI_14", 50)
            rsi_status = "Bullish" if rsi < 30 else "Bearish" if rsi > 70 else "Neutral"
            factors.append({"Factor": "RSI (14)", "Value": f"{rsi:.1f}", "Signal": rsi_status})
            
            # 2. Trend
            sma20 = latest.get("SMA_20", 0)
            sma50 = latest.get("SMA_50", 0)
            trend = "Bullish" if sma20 > sma50 else "Bearish"
            factors.append({"Factor": "Trend (SMA)", "Value": "Golden Cross" if sma20>sma50 else "Death Cross", "Signal": trend})
            
            # 3. Volatility
            vol = latest.get("Volatility_20", 0)
            vol_status = "High Risk" if vol > 0.002 else "Stable"
            factors.append({"Factor": "Volatility", "Value": f"{vol:.4f}", "Signal": vol_status})
            
            # 4. News Logic (万能兼容版) 🔥
            # 获取原始字符串，例如 "BearNews(-0.92)" 或 "Thr +0.01"
            sent_impact = pred.get('sentiment_impact', 'NeutralNews') 
            
            # --- 智能翻译逻辑 ---
            # 只要包含 "Bear" (大空) 或 "Thr +" (门槛升高) -> 判定为 Threshold +0.01
            if "Bear" in sent_impact or "Thr +" in sent_impact:
                sent_signal = "Threshold +0.01"
                
            # 只要包含 "Bad" (小空) -> 判定为 Threshold +0.005
            elif "Bad" in sent_impact:
                sent_signal = "Threshold +0.005"
                
            # 只要包含 "Bull" (大多) 或 "Thr -" (门槛降低) -> 判定为 Threshold -0.01
            elif "Bull" in sent_impact or "Thr -" in sent_impact:
                sent_signal = "Threshold -0.01"
                
            # 只要包含 "Good" (小多) -> 判定为 Threshold -0.005
            elif "Good" in sent_impact:
                sent_signal = "Threshold -0.005"
                
            # 其他情况 (Neutral, NoNews)
            else:
                sent_signal = "No Change"
                
            factors.append({"Factor": "News Impact", "Value": sent_impact, "Signal": sent_signal})

            # 5. 配色逻辑
            def color_signal(val):
                # 绿色：利好 / 稳定 / 门槛降低
                if any(s in val for s in ['Oversold', 'Bullish', 'Stable', 'Threshold -']): 
                    return 'color: #00FF00; font-weight: bold'
                # 红色：利空 / 风险 / 门槛升高
                if any(s in val for s in ['Overbought', 'Bearish', 'High', 'Threshold +']): 
                    return 'color: #FF4444; font-weight: bold'
                # 灰色
                return 'color: #888'

            df_factors = pd.DataFrame(factors)
            st.dataframe(
                df_factors.style.map(color_signal, subset=['Signal']),
                use_container_width=True,
                hide_index=True
            )

        st.divider()

        # # === 第二行：AI 的大脑结构 (Feature Importance) ===
        # st.markdown("#### 🧠 What is the AI looking at? (Feature Importance)")
        
        # # 尝试寻找最新的 feature importance 文件
        # fi_df = pd.DataFrame()
        # try:
        #     # 找到最新的 run 文件夹
        #     runs = sorted([d for d in os.listdir(os.path.join(ROOT, "models")) if d.startswith("run_")])
        #     if runs:
        #         latest_run = runs[-1]
        #         # 找 feature_importance_svc.csv 或其他
        #         fi_path = os.path.join(ROOT, "models", latest_run)
        #         csvs = [f for f in os.listdir(fi_path) if f.startswith("feature_importance")]
        #         if csvs:
        #             # 读取第一个找到的 importance 文件
        #             fi_df = pd.read_csv(os.path.join(fi_path, csvs[0]))
        #             fi_df = fi_df.sort_values("importance", ascending=True).tail(10) # 取最重要的 10 个
        # except:
        #     pass

        # if not fi_df.empty:
        #     # 画横向条形图
        #     fig_fi = go.Figure(go.Bar(
        #         x=fi_df['importance'],
        #         y=fi_df['feature'],
        #         orientation='h',
        #         marker=dict(color='#58a6ff')
        #     ))
        #     fig_fi.update_layout(
        #         height=300, 
        #         margin=dict(l=10, r=10, t=10, b=10), 
        #         paper_bgcolor="rgba(0,0,0,0)", 
        #         plot_bgcolor="rgba(0,0,0,0)",
        #         xaxis=dict(showgrid=False),
        #         yaxis=dict(showgrid=False)
        #     )
        #     st.plotly_chart(fig_fi, use_container_width=True)
        #     st.caption(f"Top 10 features driving the AI model (Source: {latest_run})")
        # else:
        #     st.info("No feature importance data found yet. (Run training to generate)")

        # === 第三行：折叠的详细数据 ===
        with st.expander("View All 28 Real-time Features"):
             # 转置显示，方便查看
            raw_features = latest.drop(labels=["Date", "GOLD_Open", "GOLD_High", "GOLD_Low", "GOLD_Close", "GOLD_Volume"], errors='ignore')
            st.dataframe(raw_features.to_frame().T, use_container_width=True)

    else:
        st.info("Waiting for data stream...")

with t4:
    st.dataframe(df.tail(10), use_container_width=True)

# ---------------------------------------------------------
# 替换 app.py 中的 with t5: 部分
# ---------------------------------------------------------

# ---------------------------------------------------------
# REPLACEMENT FOR "with t5:" in app.py
# ---------------------------------------------------------

with t5:
    st.markdown("### Model Training Insights")
    
    # 1. Auto-locate the latest report
    report_root = os.path.join(ROOT, "models")
    latest_report_dir = None
    try:
        runs = sorted([d for d in os.listdir(report_root) if d.startswith("run_")])
        if runs:
            latest_report_dir = os.path.join(report_root, runs[-1])
    except: pass

    if latest_report_dir:
        json_path = os.path.join(latest_report_dir, "comprehensive_report.json")
        
        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                report = json.load(f)
            
            # --- A. KEY METRICS CARD ---
            meta = report.get("metadata", {})
            perf = report.get("ensemble_performance", {})
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Training Rows", f"{meta.get('rows_loaded', 0):,}")
            c2.metric("Features", f"{meta.get('features_count', 0)}")
            c3.metric("Ensemble Precision", f"{perf.get('precision', 0):.1%}")
            c4.metric("Best Threshold", f"{meta.get('best_threshold_validation', 0):.4f}")
            
            st.divider()
            
            # --- B. VISUALIZATION ROW 1: The "Brain" Composition ---
            col_pie, col_perf = st.columns([1, 1.5])
            
            with col_pie:
                st.markdown("#### Ensemble Composition (Voting Weights)")
                models = perf.get("selected_models", [])
                weights = perf.get("voting_weights", [])
                
                # Create Donut Chart
                fig_pie = go.Figure(data=[go.Pie(
                    labels=[m.upper() for m in models], 
                    values=weights, 
                    hole=.5,
                    textinfo='label+percent',
                    marker=dict(colors=['#00CC96', '#636EFA', '#EF553B', '#AB63FA'])
                )])
                fig_pie.update_layout(
                    height=300, 
                    margin=dict(t=20, b=20, l=20, r=20), 
                    paper_bgcolor="rgba(0,0,0,0)",
                    showlegend=False
                )
                st.plotly_chart(fig_pie, use_container_width=True)

            with col_perf:
                st.markdown("#### ⚔️ Model Performance Battle")
                st.caption("Comparing Win Rate (Precision) vs Accuracy vs Activity (Recall)")
                
                base_perfs = report.get("base_models_performance", {})
                
                # Prepare data structures
                model_names = []
                precisions = []
                accuracies = [] # 🆕 新增 Accuracy
                recalls = []
                f1s = []
                
                # Extract data
                for m_name, m_data in base_perfs.items():
                    model_names.append(m_name.upper())
                    precisions.append(m_data.get("test_precision", 0))
                    accuracies.append(m_data.get("test_acc", 0)) # 🆕 获取数据
                    recalls.append(m_data.get("test_recall", 0))
                    f1s.append(m_data.get("test_f1", 0))
                
                # Create Advanced Grouped Bar Chart
                fig_bar = go.Figure()
                
                # 1. Precision Bar (Win Rate - Green/Teal)
                fig_bar.add_trace(go.Bar(
                    x=model_names, y=precisions, name='Precision (Win Rate)',
                    marker_color='#00CC96', # Modern Teal
                    text=[f"{p:.1%}" for p in precisions],
                    textposition='auto'
                ))
                
                # 2. Accuracy Bar (Overall Correctness - Cyan) 🆕
                fig_bar.add_trace(go.Bar(
                    x=model_names, y=accuracies, name='Accuracy (Overall)',
                    marker_color='#19D3F3', # Cyan
                    text=[f"{a:.1%}" for a in accuracies],
                    textposition='auto'
                ))

                # 3. Recall Bar (Activity - Purple)
                fig_bar.add_trace(go.Bar(
                    x=model_names, y=recalls, name='Recall (Opportunity)',
                    marker_color='#AB63FA', # Vivid Purple
                    text=[f"{r:.1%}" for r in recalls],
                    textposition='auto'
                ))
                
                # 4. F1 Score Bar (Balance - Orange/Gold)
                fig_bar.add_trace(go.Bar(
                    x=model_names, y=f1s, name='F1 Score (Balance)',
                    marker_color='#FFA15A', # Soft Orange
                    text=[f"{f:.2f}" for f in f1s],
                    textposition='auto'
                ))
                
                fig_bar.update_layout(
                    barmode='group',
                    height=320,
                    margin=dict(t=30, b=20, l=40, r=10),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    legend=dict(
                        orientation="h", 
                        y=1.15, 
                        x=0.5, 
                        xanchor='center',
                        font=dict(size=10)
                    ),
                    yaxis=dict(
                        title="Score", 
                        showgrid=True, 
                        gridcolor='#333', 
                        range=[0, 1.1] # 稍微留高一点给文字
                    ),
                    font=dict(family="JetBrains Mono"),
                    hovermode="x unified"
                )
                st.plotly_chart(fig_bar, use_container_width=True)


            st.divider()

            # --- C. VISUALIZATION ROW 2: Feature Importance ---
            st.markdown("#### Top Drivers of Prediction")
            
            # Extract feature importance from XGB (usually the most interpretable)
            # If XGB is not available, fallback to RF or the first available
            all_fi = report.get("feature_importances", {})
            fi_data = all_fi.get("xgb", all_fi.get("rf", {}))
            
            if fi_data:
                # Sort and take top 10
                sorted_fi = sorted(fi_data.items(), key=lambda x: x[1], reverse=True)[:10]
                feats, scores = zip(*sorted_fi)
                
                fig_fi = go.Figure(go.Bar(
                    x=scores,
                    y=feats,
                    orientation='h',
                    marker=dict(color=scores, colorscale='Viridis'),
                ))
                
                fig_fi.update_layout(
                    height=350,
                    margin=dict(t=10, b=10, l=10, r=10),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    yaxis=dict(autorange="reversed"), # Put top feature at top
                    xaxis=dict(title="Relative Importance Score")
                )
                st.plotly_chart(fig_fi, use_container_width=True)
            else:
                st.info("No feature importance data available in report.")

            # --- D. RAW DATA EXPANDER ---
            with st.expander("View Raw JSON Report"):
                st.json(report)
        
        else:
            st.error(f"Report file not found at: {json_path}")
    else:
        st.info("No training runs found. Please run `train_gold_model_v3_enhanced.py` first.")