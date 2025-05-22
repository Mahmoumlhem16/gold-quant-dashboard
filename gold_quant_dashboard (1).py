
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from statsmodels.tsa.arima.model import ARIMA
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def compute_indicators(df):
    df['SMA_50'] = df['Gold_Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Gold_Close'].rolling(window=200).mean()
    df['RSI'] = compute_rsi(df['Gold_Close'])
    return df

def generate_signals(df):
    signals = []
    for i in range(1, len(df)):
        row = df.iloc[i]
        prev_row = df.iloc[i-1]
        signal = 'HOLD'
        if row['RSI'] < 30 and row['Gold_Close'] > row['SMA_50']:
            signal = 'BUY'
        elif row['RSI'] > 70 and row['Gold_Close'] < row['SMA_50']:
            signal = 'SELL'
        if prev_row['SMA_50'] < prev_row['SMA_200'] and row['SMA_50'] > row['SMA_200']:
            signal = 'BUY (Golden Cross)'
        elif prev_row['SMA_50'] > prev_row['SMA_200'] and row['SMA_50'] < row['SMA_200']:
            signal = 'SELL (Death Cross)'
        signals.append(signal)
    df = df.iloc[1:]
    df['Signal'] = signals
    return df

def monte_carlo_simulation(start_price, mu, sigma, days=60, sims=500):
    results = np.zeros((days, sims))
    for s in range(sims):
        price = start_price
        for d in range(days):
            shock = np.random.normal(loc=mu, scale=sigma)
            price *= (1 + shock)
            results[d, s] = price
    return results

st.set_page_config(layout="wide")
st.title("GoldQuant Pro – نموذج كوانتي احترافي للذهب")

with st.spinner("تحميل البيانات..."):
    end = datetime.today()
    start = end - timedelta(days=365 * 5)
    gold = yf.download("GLD", start=start, end=end)
    dxy = yf.download("DX-Y.NYB", start=start, end=end)
    tnx = yf.download("^TNX", start=start, end=end)

    data = pd.DataFrame({
        'Gold_Close': gold['Close'],
        'DXY': dxy['Close'],
        '10Y_Yield': tnx['Close']
    }).dropna()

data = compute_indicators(data)
data['Daily_Return'] = data['Gold_Close'].pct_change()
volatility = data['Daily_Return'].std() * np.sqrt(252)
VaR_95 = np.percentile(data['Daily_Return'].dropna(), 5)

arima_model = ARIMA(data['Gold_Close'], order=(5,1,0)).fit()
forecast = arima_model.forecast(steps=10)

data['Target'] = (data['Gold_Close'].shift(-1) > data['Gold_Close']).astype(int)
data.dropna(inplace=True)
features = ['DXY', '10Y_Yield', 'RSI', 'SMA_50', 'SMA_200']
X = data[features]
y = data['Target']
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)
model = XGBClassifier()
model.fit(X_train, y_train)
data['AI_Signal'] = model.predict(data[features])
data['AI_Signal'] = data['AI_Signal'].map({1: 'BUY', 0: 'SELL'})

data = generate_signals(data)

def combine_signals(row):
    if row['Signal'].startswith('BUY') and row['AI_Signal'] == 'BUY':
        return 'STRONG BUY'
    elif row['Signal'].startswith('SELL') and row['AI_Signal'] == 'SELL':
        return 'STRONG SELL'
    else:
        return 'HOLD'

data['Final_Signal'] = data.apply(combine_signals, axis=1)

st.subheader("سعر الذهب والمؤشرات")
st.line_chart(data[['Gold_Close', 'SMA_50', 'SMA_200']])

st.subheader("إشارات التداول")
st.dataframe(data[['Gold_Close', 'RSI', 'Signal', 'AI_Signal', 'Final_Signal']].tail(10))

st.subheader("تحليل المخاطر")
st.write(f"**التذبذب السنوي:** {volatility:.2%}")
st.write(f"**قيمة في خطر (VaR 95%):** {VaR_95:.2%}")

st.subheader("توقع ARIMA (10 أيام)")
st.line_chart(forecast)

st.subheader("محاكاة مونت كارلو (60 يوم)")
simulated = monte_carlo_simulation(
    start_price=data['Gold_Close'].iloc[-1],
    mu=data['Daily_Return'].mean(),
    sigma=data['Daily_Return'].std(),
    days=60,
    sims=300
)
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(simulated, color='skyblue', alpha=0.2)
ax.set_title("Monte Carlo Simulation – 60 Day Gold Price")
st.pyplot(fig)
