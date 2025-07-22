import streamlit as st
import ccxt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

st.set_page_config(page_title="Binance Crypto Simulator", layout="centered")
st.title("📊 Binance Monte Carlo Return Simulator")

# --- User Inputs ---
symbol_input = st.text_input("Enter Binance Symbol (e.g. BTC/USDT, ETH/USDT)", "BTC/USDT").upper().strip()
holding_days = st.slider("Holding Period (days)", 10, 180, 60)
simulations = st.number_input("Number of Simulations", min_value=1000, max_value=100000, value=1000, step=1000)

binance = ccxt.binance()

@st.cache_data(show_spinner=False)
def get_binance_data(symbol):
    try:
        since = binance.parse8601((datetime.utcnow() - timedelta(days=365)).strftime("%Y-%m-%dT%H:%M:%S"))
        ohlcv = binance.fetch_ohlcv(symbol, timeframe='1d', since=since)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

if not symbol_input:
    st.warning("Please enter a Binance symbol like BTC/USDT")
    st.stop()

st.write(f"⏳ Fetching 1 year of daily data for {symbol_input} from Binance...")
data = get_binance_data(symbol_input)

if data.empty or len(data) < 30:
    st.error("❌ Not enough historical data found for this coin or invalid symbol. Try another one.")
    st.stop()

st.write(f"✅ Data loaded: {len(data)} days of prices available.")

# Calculate log returns
returns = np.log(data['close'] / data['close'].shift(1)).dropna()
mu = returns.mean()
sigma = returns.std()

st.write(f"Running {simulations} Monte Carlo simulations for {holding_days} days holding period...")
progress_bar = st.progress(0)

dt = 1  # one day timestep
batch_size = 5000

results = []
for start in range(0, simulations, batch_size):
    end = min(start + batch_size, simulations)
    size = end - start
    rand_matrix = np.random.normal(0, 1, size=(size, holding_days))
    daily_returns = np.exp((mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * rand_matrix)
    simulated_returns = daily_returns.prod(axis=1) - 1
    results.extend(simulated_returns)
    progress_bar.progress(end / simulations)

progress_bar.empty()

results = np.array(results)
expected_return = results.mean()
std_dev = results.std()
prob_gain = (results > 0).mean()
r5, r95 = np.percentile(results, [5, 95])

st.subheader(f"Results for {symbol_input}")
col1, col2, col3 = st.columns(3)
col1.metric("Expected Return", f"{expected_return:.2%}")
col2.metric("Risk (Std Dev)", f"{std_dev:.2%}")
col3.metric("Chance of Gain", f"{prob_gain * 100:.1f}%")

# Plot histogram
fig, ax = plt.subplots()
ax.hist(results, bins=100, color='skyblue', edgecolor='black')
ax.axvline(r5, color='red', linestyle='--', label='5th percentile')
ax.axvline(r95, color='green', linestyle='--', label='95th percentile')
ax.set_title(f"Simulated Return Distribution for {symbol_input}")
ax.set_xlabel("Return")
ax.set_ylabel("Frequency")
ax.legend()
st.pyplot(fig)
