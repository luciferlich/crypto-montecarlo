import streamlit as st
import ccxt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from PIL import Image

# --- PAGE CONFIG ---
st.set_page_config(page_title="KuCoin Crypto Simulator", layout="centered")

# --- HEADER STYLE ---
st.markdown("""
<style>
    .header-container {
        display: flex; align-items: center; justify-content: center;
        gap: 20px; margin-bottom: 10px;
    }
    .big-title {
        font-size: 4rem; font-weight: 900;
        font-family: 'Arial Black', Gadget, sans-serif;
        color: #1F77B4; user-select: none;
    }
</style>
""", unsafe_allow_html=True)

# --- HEADER CONTENT ---
col1, col2, col3 = st.columns([1, 2, 1])
with col1:
    try:
        logo = Image.open("logo.png")
        st.image(logo, width=80)
    except:
        st.write("")
with col2:
    st.markdown('<div class="big-title">ADEX</div>', unsafe_allow_html=True)
with col3:
    st.write("")

st.title("📊 KuCoin Monte Carlo Return Simulator")

# --- USER INPUT ---
symbol_input = st.text_input("Enter Symbol (e.g. BTC/USDT, ETH/USDT)", "BTC/USDT").upper().strip()
holding_days = st.slider("Holding Period (days)", 10, 180, 60)
simulations = st.number_input("Number of Simulations", min_value=1000, max_value=100000, value=5000, step=1000)

kucoin = ccxt.kucoin()

@st.cache_data(show_spinner=False)
def get_kucoin_data(symbol):
    try:
        since = kucoin.parse8601((datetime.utcnow() - timedelta(days=365)).strftime("%Y-%m-%dT%H:%M:%S"))
        ohlcv = kucoin.fetch_ohlcv(symbol, timeframe='1d', since=since)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

@st.cache_data
def is_symbol_supported(symbol):
    try:
        markets = kucoin.load_markets()
        return symbol in markets
    except:
        return False

if not symbol_input:
    st.warning("Please enter a symbol like BTC/USDT")
    st.stop()

if not is_symbol_supported(symbol_input):
    st.error(f"❌ {symbol_input} is not supported on KuCoin.")
    st.stop()

st.write(f"⏳ Fetching 1 year of daily data for {symbol_input} from KuCoin...")
data = get_kucoin_data(symbol_input)

if data.empty or len(data) < 30:
    st.error("❌ Not enough historical data found or invalid symbol. Try another one.")
    st.stop()

st.success(f"✅ Loaded {len(data)} days of historical prices.")

# --- MONTE CARLO SIMULATION ---
returns = np.log(data['close'] / data['close'].shift(1)).dropna()
mu, sigma = returns.mean(), returns.std()
start_price = data['close'].iloc[-1]

st.write(f"Running {simulations} Monte Carlo simulations for {holding_days} days...")
progress = st.progress(0)

results = []
simulated_prices = []
batch_size = 5000
dt = 1

for start in range(0, simulations, batch_size):
    end = min(start + batch_size, simulations)
    size = end - start
    rand_matrix = np.random.normal(0, 1, size=(size, holding_days))
    daily_returns = np.exp((mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * rand_matrix)
    price_paths = start_price * daily_returns.cumprod(axis=1)
    simulated_returns = price_paths[:, -1] / start_price - 1

    results.extend(simulated_returns)
    simulated_prices.append(price_paths)
    progress.progress(end / simulations)

progress.empty()

results = np.array(results)
simulated_prices = np.vstack(simulated_prices)

# --- METRICS ---
st.subheader(f"Results for {symbol_input}")
col1, col2, col3 = st.columns(3)
col1.metric("Expected Return", f"{results.mean():.2%}")
col2.metric("Risk (Std Dev)", f"{results.std():.2%}")
col3.metric("Chance of Gain", f"{(results > 0).mean() * 100:.1f}%")

# --- RETURN DISTRIBUTION PLOT ---
fig, ax = plt.subplots()
ax.hist(results, bins=100, color='skyblue', edgecolor='black')
r5, r95 = np.percentile(results, [5, 95])
ax.axvline(r5, color='red', linestyle='--', label='5th percentile')
ax.axvline(r95, color='green', linestyle='--', label='95th percentile')
ax.set_title(f"Simulated Return Distribution for {symbol_input}")
ax.set_xlabel("Return")
ax.set_ylabel("Frequency")
ax.legend()
st.pyplot(fig)

# --- PRICE PROJECTION ---
prices = start_price * (1 + results)
expected_price = prices.mean()
price_std = prices.std()
p5, p95 = np.percentile(prices, [5, 95])

st.subheader(f"Price Projection after {holding_days} days")
c1, c2, c3 = st.columns(3)
c1.metric("Expected Price", f"${expected_price:,.2f}")
c2.metric("5th Percentile", f"${p5:,.2f}")
c3.metric("95th Percentile", f"${p95:,.2f}")

fig2, ax2 = plt.subplots()
ax2.hist(prices, bins=100, color='lightgreen', edgecolor='black')
ax2.axvline(p5, color='red', linestyle='--')
ax2.axvline(p95, color='green', linestyle='--')
ax2.set_title("Simulated Price Distribution")
ax2.set_xlabel("Price")
ax2.set_ylabel("Frequency")
st.pyplot(fig2)

# --- SMART ALERTS ---
st.markdown("## 🔔 Smart Alerts & Bias Detection")
bias_alerts = []

if holding_days >= 2:
    median_day_2 = np.median(simulated_prices[:, 1])
    if median_day_2 < start_price:
        bias_alerts.append("⚠️ Expected drop in 2 days — consider rebalancing.")

if holding_days >= 10:
    day10 = simulated_prices[:, 9]
    upside_prob = (day10 > start_price * 1.15).mean()
    if upside_prob > 0.5:
        bias_alerts.append(f"🚀 Simulated upside of +15% over 10 days for {symbol_input} with {upside_prob:.0%} probability.")

if bias_alerts:
    for alert in bias_alerts:
        st.warning(alert)
else:
    st.info("✅ No significant alert detected.")
