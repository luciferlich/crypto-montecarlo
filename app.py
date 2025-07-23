import streamlit as st
import ccxt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from PIL import Image

# Page config
st.set_page_config(page_title="ADEX Crypto Simulator", layout="centered", page_icon="logo.png")

# CSS for centering logo and ADEX text below it
st.markdown(
    """
    <style>
    .logo-container {
        display: flex;
        justify-content: center;
        margin-top: 20px;
        margin-bottom: 10px;
    }
    .adex-text {
        color: white;
        font-size: 48px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 30px;
        font-family: 'Arial Black', Gadget, sans-serif;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load logo image
logo = Image.open("logo.png")

# Show logo centered
st.markdown('<div class="logo-container">', unsafe_allow_html=True)
st.image(logo, width=150)
st.markdown('</div>', unsafe_allow_html=True)

# Show ADEX text below logo
st.markdown('<div class="adex-text">ADEX</div>', unsafe_allow_html=True)

# --- User Inputs ---
symbol_input = st.text_input("Enter Binance Symbol (e.g. BTC/USDT, ETH/USDT)", "BTC/USDT").upper().strip()
holding_days = st.slider("Holding Period (days)", 10, 180, 60)
simulations = st.number_input("Number of Simulations", min_value=1000, max_value=100000, value=1000, step=1000)

# Simulation types (LSTM removed)
sim_type = st.selectbox(
    "Select Simulation Type",
    [
        "Geometric Brownian Motion (GBM)",
        "Standard Brownian Motion",
        "Jump Diffusion",
        "Historical Bootstrap",
        "GARCH + GBM hybrid"
    ]
)

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

# Calculate log returns and stats
returns = np.log(data['close'] / data['close'].shift(1)).dropna()
mu = returns.mean()
sigma = returns.std()
start_price = data['close'].iloc[-1]

st.write(f"Running {simulations} simulations for {holding_days} days holding period using {sim_type}...")

progress_bar = st.progress(0)
dt = 1
results = []
simulated_prices = []
batch_size = 5000

for start in range(0, simulations, batch_size):
    end = min(start + batch_size, simulations)
    size = end - start

    if sim_type == "Geometric Brownian Motion (GBM)":
        rand_matrix = np.random.normal(0, 1, size=(size, holding_days))
        daily_returns = np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * rand_matrix)
        prices = start_price * daily_returns.cumprod(axis=1)
    elif sim_type == "Standard Brownian Motion":
        increments = np.random.normal(mu * dt, sigma * np.sqrt(dt), size=(size, holding_days))
        log_prices = np.cumsum(increments, axis=1) + np.log(start_price)
        prices = np.exp(log_prices)
    elif sim_type == "Jump Diffusion":
        jump_lambda = 0.1
        jump_mu = -0.01
        jump_sigma = 0.02
        rand_matrix = np.random.normal(0, 1, size=(size, holding_days))
        jumps = np.random.poisson(jump_lambda, size=(size, holding_days)) * np.random.normal(jump_mu, jump_sigma, size=(size, holding_days))
        daily_returns = np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * rand_matrix + jumps)
        prices = start_price * daily_returns.cumprod(axis=1)
    elif sim_type == "Historical Bootstrap":
        bootstrapped_returns = np.random.choice(returns, size=(size, holding_days), replace=True)
        prices = start_price * np.exp(np.cumsum(bootstrapped_returns, axis=1))
    elif sim_type == "GARCH + GBM hybrid":
        # Simple GARCH imitation: change volatility over time randomly
        garch_vol = np.clip(sigma + np.random.normal(0, 0.01, (size, holding_days)), 0, 1)
        rand_matrix = np.random.normal(0, 1, size=(size, holding_days))
        daily_returns = np.exp((mu - 0.5 * garch_vol**2) * dt + garch_vol * np.sqrt(dt) * rand_matrix)
        prices = start_price * daily_returns.cumprod(axis=1)
    elif sim_type == "REGRESSION":
        # Linear projection based on historical return mean
        trend = mu * np.arange(1, holding_days + 1)
        noise = np.random.normal(0, sigma, (size, holding_days))
        prices = start_price * np.exp(trend + noise)
    else:
        prices = np.full((size, holding_days), start_price)

    final_returns = prices[:, -1] / start_price - 1
    results.extend(final_returns)
    simulated_prices.append(prices)
    progress_bar.progress(end / simulations)

progress_bar.empty()

results = np.array(results)
simulated_prices = np.vstack(simulated_prices)

expected_return = results.mean()
std_dev = results.std()
prob_gain = (results > 0).mean()

st.subheader(f"Simulation Results for {symbol_input}")

col1, col2, col3 = st.columns(3)
col1.metric("Expected Return", f"{expected_return:.2%}")
col2.metric("Risk (Std Dev)", f"{std_dev:.2%}")
col3.metric("Chance of Gain", f"{prob_gain * 100:.1f}%")

# Show probabilistic price forecast with adaptive decimal formatting
percentiles = [10, 25, 40, 50, 60, 75, 90]
price_percentiles = np.percentile(simulated_prices[:, -1], percentiles)

decimal_places = 2 if start_price < 20 else 0
price_format = f"{{:,.{decimal_places}f}}"

prob_summary = f"Price Prediction Probabilities after {holding_days} days:\n"
for p, price in zip(percentiles, price_percentiles):
    prob = 100 - p  # probability price >= this percentile price
    formatted_price = price_format.format(price)
    prob_summary += f"  • Price ≥ ${formatted_price} with {prob}% probability\n"

st.markdown("### Probabilistic Price Forecast")
st.text(prob_summary)

# Plot histogram of returns
fig, ax = plt.subplots()
ax.hist(results, bins=100, color='skyblue', edgecolor='black')
ax.axvline(np.percentile(results, 10), color='red', linestyle='--', label='10th percentile')
ax.axvline(np.percentile(results, 90), color='green', linestyle='--', label='90th percentile')
ax.set_title(f"Simulated Return Distribution for {symbol_input}")
ax.set_xlabel("Return")
ax.set_ylabel("Frequency")
ax.legend()
st.pyplot(fig)

# Plot sample simulated price paths
fig2, ax2 = plt.subplots(figsize=(10, 5))
for i in range(min(10, simulations)):
    ax2.plot(simulated_prices[i], lw=1)
ax2.set_title(f"Sample Simulated Price Paths for {symbol_input} ({sim_type})")
ax2.set_xlabel("Days")
ax2.set_ylabel("Price")
ax2.grid(True)
st.pyplot(fig2)
# --- Target Price Prediction ---
st.markdown("## 🎯 Target Price Prediction")

target_model = st.selectbox("Select Prediction Model", ["Linear Regression", "XGBoost"])
target_return_pct = st.number_input("Enter Target Return (%)", min_value=1.0, max_value=100.0, value=5.0, step=0.5)

# Estimate time to reach target return using simplified volatility model
log_target = np.log(1 + target_return_pct / 100)

# Expected daily return and std dev (volatility)
daily_mu = mu
daily_sigma = sigma

if daily_mu <= 0:
    estimated_days = log_target / (0.5 * daily_sigma ** 2)  # conservative fallback
else:
    estimated_days = log_target / daily_mu

if estimated_days < 0:
    result_msg = "🔴 Based on current trend, reaching this target might not be feasible in the short term."
else:
    result_msg = f"📈 Estimated time to reach {target_return_pct:.1f}% gain: **~{estimated_days:.1f} days**"

st.success(result_msg)
