import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from PIL import Image
from pycoingecko import CoinGeckoAPI

st.set_page_config(page_title="ADEX Crypto Simulator", layout="centered")

# Header setup (logo + title)
st.markdown("""
<style>
.header-container { display: flex; align-items: center; justify-content: center; gap: 20px; margin-bottom: 10px; }
.big-title { font-size: 4rem; font-weight: 900; font-family: 'Arial Black', Gadget, sans-serif; color: #1F77B4; user-select: none; }
</style>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1,2,1])
with col1:
    try:
        logo = Image.open("logo.png")
        st.image(logo, width=80)
    except:
        pass
with col2:
    st.markdown('<div class="big-title">ADEX</div>', unsafe_allow_html=True)
with col3:
    st.write("")

st.title("📊 Monte Carlo Return Simulator (CoinGecko Data)")

# --- Inputs ---
symbol_input = st.text_input("Enter CoinGecko coin ID (e.g., bitcoin, ethereum)", "bitcoin").lower().strip()
holding_days = st.slider("Holding Period (days)", 10, 180, 60)
simulations = st.number_input("Number of Simulations", 1000, 50000, 5000, 1000)

cg = CoinGeckoAPI()
@st.cache_data(show_spinner=False)
def get_gecko_data(coin_id):
    try:
        data = cg.get_coin_ohlc_by_id(id=coin_id, vs_currency='usd', days='365')
        df = pd.DataFrame(data, columns=['timestamp','open','high','low','close'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    except Exception as e:
        st.error(f"Error fetching data from CoinGecko: {e}")
        return pd.DataFrame()

st.write(f"⏳ Fetching 1 year of daily data for {symbol_input}...")
data = get_gecko_data(symbol_input)
if data.empty or len(data) < 30:
    st.error("Not enough historical data—check the coin ID or try another.")
    st.stop()
st.write(f"✅ Retrieved {len(data)} days of data.")

# Compute log returns
returns = np.log(data['close'] / data['close'].shift(1)).dropna()
mu, sigma = returns.mean(), returns.std()
start_price = data['close'].iloc[-1]

# Run Monte Carlo
st.write(f"Running {simulations} simulations for {holding_days} days...")
progress = st.progress(0)
results = []
batch = 5000

for i in range(0, simulations, batch):
    n = min(batch, simulations - i)
    Z = np.random.randn(n, holding_days)
    daily = np.exp((mu - 0.5 * sigma**2) + sigma * Z)
    P = start_price * daily.cumprod(axis=1)
    results.extend(P[:, -1] / start_price - 1)
    progress.progress((i + n) / simulations)
progress.empty()

# Metrics & histogram
res = np.array(results)
st.subheader("Results")
c1, c2, c3 = st.columns(3)
c1.metric("Exp. Return", f"{res.mean():.2%}")
c2.metric("Std Dev", f"{res.std():.2%}")
c3.metric("Chance Gain", f"{(res > 0).mean() * 100:.1f}%")

fig, ax = plt.subplots()
ax.hist(res, bins=100, color='skyblue', edgecolor='black')
st.pyplot(fig)

# Price projections
prices = start_price * (1 + res)
st.subheader(f"Price Projection after {holding_days} days")
c4, c5, c6 = st.columns(3)
c4.metric("Expected Price", f"${prices.mean():,.2f}")
c5.metric("5th Percentile", f"${np.percentile(prices,5):,.2f}")
c6.metric("95th Percentile", f"${np.percentile(prices,95):,.2f}")

fig2, ax2 = plt.subplots()
ax2.hist(prices, bins=100, color='lightgreen', edgecolor='black')
st.pyplot(fig2)
