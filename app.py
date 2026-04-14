import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="AAPL Stock Price Predictor", layout="wide")
st.title("AAPL Stock Price Prediction")
st.markdown("**Machine Learning Algorithms Course | MO 3208**")
st.markdown("---")

@st.cache_data
def load_data():
    try:
        ticker = yf.Ticker("AAPL")
        raw = ticker.history(start="2020-01-01", end="2026-04-13", auto_adjust=True)
    except Exception as e:
        st.error(f"❌ Ошибка при загрузке данных: {e}")
        st.stop()

    if raw is None or raw.empty:
        st.error("❌ Yahoo Finance не вернул данные. Попробуй Manage App → Reboot.")
        st.stop()

    raw = raw.reset_index()

    if 'Datetime' in raw.columns:
        raw = raw.rename(columns={'Datetime': 'Date'})

    # ✅ Убираем timezone из даты — иначе matplotlib падает
    raw['Date'] = pd.to_datetime(raw['Date']).dt.tz_localize(None)

    df = raw[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].copy()
    df['Price_Range'] = df['High'] - df['Low']
    df['Price_Change'] = df['Close'] - df['Open']
    df['MA_7'] = df['Close'].rolling(window=7).mean()
    df['MA_30'] = df['Close'].rolling(window=30).mean()
    df['Volatility'] = df['Close'].rolling(window=7).std()
    df = df.dropna().reset_index(drop=True)
    return df

@st.cache_resource
def train_models(_df):
    features = ['Open', 'High', 'Low', 'Volume', 'Price_Range', 'MA_7', 'MA_30']
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X = scaler_X.fit_transform(_df[features])
    y = scaler_y.fit_transform(_df[['Close']])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False)
    ridge = Ridge(alpha=0.001)
    ridge.fit(X_train, y_train.ravel())
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train.ravel())
    knn = KNeighborsRegressor(n_neighbors=5)
    knn.fit(X_train, y_train.ravel())
    return ridge, rf, knn, scaler_X, scaler_y, X_test, y_test

with st.spinner("Loading data..."):
    df = load_data()
    ridge, rf, knn, scaler_X, scaler_y, X_test, y_test = train_models(df)

st.sidebar.title("Settings")
model_choice = st.sidebar.selectbox(
    "Choose a model:",
    ["Ridge Regression (Best)", "Random Forest", "KNN Regressor"]
)
st.sidebar.markdown("---")
st.sidebar.info("Course: MO 3208\nDataset: AAPL Yahoo Finance\nPeriod: 2020-2026\nTask: Regression")

st.header("Dataset Overview")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Records", str(len(df)))
col2.metric("Date Range", "2020 - 2026")
col3.metric("Features Used", "7")
col4.metric("Latest Close", "$" + str(round(float(df['Close'].iloc[-1]), 2)))

with st.expander("View Raw Data"):
    st.dataframe(df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].tail(20))

st.header("AAPL Closing Price History")
fig1, ax1 = plt.subplots(figsize=(12, 4))
ax1.plot(df['Date'], df['Close'], color='steelblue', linewidth=1.5, label='Close')
ax1.plot(df['Date'], df['MA_7'], color='orange', linewidth=1, label='MA 7')
ax1.plot(df['Date'], df['MA_30'], color='red', linewidth=1, label='MA 30')
ax1.set_title('AAPL Closing Price with Moving Averages')
ax1.set_xlabel('Date')
ax1.set_ylabel('Price (USD)')
ax1.legend()
ax1.grid(alpha=0.3)
plt.tight_layout()
st.pyplot
