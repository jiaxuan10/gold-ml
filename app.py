import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import pytz

st.set_page_config(page_title="Gold Live Monitor", layout="wide")
st.title("Gold Live Monitor — 实时/近实时黄金价格可视化")

# ---- SIDEBAR 控件 ----
with st.sidebar:
    st.markdown("## 设置")
    ticker = st.text_input("Ticker (Yahoo)", value="GC=F", help="默认GC=F 为黄金期货（Yahoo）")
    period = st.selectbox("Period", options=["1d","5d","1mo","3mo","6mo","1y","5y","max"], index=0)
    interval = st.selectbox("Interval", options=["1m","2m","5m","15m","30m","60m","90m","1h","1d"], index=0)
    ma_short = st.number_input("MA 短期 (periods)", value=5, min_value=1)
    ma_long = st.number_input("MA 长期 (periods)", value=20, min_value=1)
    auto_refresh = st.checkbox("自动刷新（需要 streamlit-autorefresh，可选）", value=False)
    refresh_seconds = st.number_input("刷新间隔（秒）", min_value=1, value=10)
    st.markdown("---")
    st.write("注意：Yahoo (yfinance) 提供的是接近实时的公共数据，但不是交易所级别 tick feed。若需要更高频、低延迟数据，考虑 Alpha Vantage / Polygon / 交易所 feed（需API key）。")

# ---- 数据抓取（缓存短时） ----
@st.cache_data(ttl=10)
def fetch_data(ticker: str, period: str, interval: str) -> pd.DataFrame:
    df = yf.download(tickers=ticker, period=period, interval=interval, progress=False)

    if df is None or df.empty:
        return pd.DataFrame()

    # reset index, rename时间列
    df = df.reset_index()
    if "Datetime" in df.columns:
        df = df.rename(columns={"Datetime": "date"})
    elif "Date" in df.columns:
        df = df.rename(columns={"Date": "date"})

    # --- 自动扁平化列名 ---
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [str(c[0]) for c in df.columns]

    # --- 转换为 float ---
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # --- 处理时区（转马来西亚时间） ---
    if pd.api.types.is_datetime64_any_dtype(df["date"]):
        if df["date"].dt.tz is None:
            df["date"] = df["date"].dt.tz_localize("UTC").dt.tz_convert("Asia/Kuala_Lumpur")
        else:
            df["date"] = df["date"].dt.tz_convert("Asia/Kuala_Lumpur")

    # --- 确保所需列存在 ---
    expected = ["date", "Open", "High", "Low", "Close", "Volume"]
    for c in expected:
        if c not in df.columns:
            df[c] = pd.NA

    return df

# optional auto refresh helper
if auto_refresh:
    try:
        from streamlit_autorefresh import st_autorefresh
        # st_autorefresh returns an incrementing counter; each run will re-execute fetch_data
        st_autorefresh(interval=refresh_seconds * 1000, key="gold_autorefresh")
    except Exception:
        st.warning("若要启用自动刷新，请先 pip install streamlit-autorefresh。")

df = fetch_data(ticker, period, interval)

if df.empty:
    st.error("未取得数据 — 请检查 ticker、period、interval，或网络/代理。")
    st.write("常见原因：Yahoo 对某些间隔/代码不提供数据；或本地网络阻断。")
    st.stop()

# ---- 顶部数值卡 ----
latest = df.iloc[-1]
prev = df.iloc[-2] if len(df) > 1 else latest

col1, col2, col3 = st.columns([1, 2, 1])
col1.metric("最新收盘价", f"{float(latest['Close']):.2f}", delta=f"{(float(latest['Close'])-float(prev['Close'])):.2f}")
col2.write(f"最新时间：{pd.to_datetime(latest['date'])}")

# 修复版本
vol_value = latest["Volume"]

# 如果还是 Series，就取第一个值
if isinstance(vol_value, pd.Series):
    vol_value = vol_value.iloc[0]

if pd.isna(vol_value):
    vol_display = "N/A"
else:
    vol_display = f"{int(vol_value)}"

col3.metric("最近成交量", vol_display)

# ---- 计算技术指标 ----
df["MA_short"] = df["Close"].astype(float).rolling(ma_short, min_periods=1).mean()
df["MA_long"] = df["Close"].astype(float).rolling(ma_long, min_periods=1).mean()

# ---- 绘图 (Plotly Candlestick + MA) ----
# 确保数值列是 float
for col in ["Open", "High", "Low", "Close"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

fig = go.Figure()
fig.add_trace(go.Candlestick(
    x=df["date"],
    open=df["Open"],
    high=df["High"],
    low=df["Low"],
    close=df["Close"],
    name="Price",
    increasing_line_color="green",
    increasing_fillcolor="green",
    decreasing_line_color="red",
    decreasing_fillcolor="red"
))
fig.add_trace(go.Scatter(x=df["date"], y=df["MA_short"], mode="lines", name=f"MA{ma_short}"))
fig.add_trace(go.Scatter(x=df["date"], y=df["MA_long"], mode="lines", name=f"MA{ma_long}"))

fig.update_layout(
    xaxis_title="Time",
    yaxis_title="Price",
    xaxis_rangeslider_visible=False,
    height=600,
    margin=dict(l=10, r=10, t=40, b=10)
)

st.plotly_chart(fig, use_container_width=True)

# ---- 额外信息（表格） ----
with st.expander("查看原始数据（最后 50 条）"):
    st.dataframe(df.tail(50))

# ---- 手动刷新按钮 ----
# ---- 手动刷新按钮（兼容最新 Streamlit） ----
if "refresh_counter" not in st.session_state:
    st.session_state.refresh_counter = 0

if st.button("手动刷新"):
    st.session_state.refresh_counter += 1

# 利用 session_state 触发重新执行
_ = st.session_state.refresh_counter
st.markdown("---")
st.caption("提示：若要在后台长期运行此界面，可部署到 Streamlit Cloud / Docker / VPS。")
