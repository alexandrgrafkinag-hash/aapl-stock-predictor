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

st.set_page_config(page_title="AAPL Stock Price Predictor", page_icon="📈", layout="wide")
st.title("📈 AAPL Stock Price Prediction")
st.markdown("**Machine Learning Algorithms Course | MO 3208**")
st.markdown("---")

@st.cache_data
def load_data():
    # Download AAPL data from Yahoo Finance
    raw = yf.download("AAPL", start="2020-01-01", end="2026-04-13", auto_adjust=True)
    
    # Flatten multi-level columns if present
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    
    # Reset index to get Date as column
    df = raw.reset_index()
    
    # Keep only needed columns
    df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].copy()
    df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    
    # Feature engineering
    df['Price_Range'] = df['High'] - df['Low']
    df['Price_Change'] = df['Close'] - df['Open']
    df['MA_7']  = df['Close'].rolling(window=7).mean()
    df['MA_30'] = df['Close'].rolling(window=30).mean()
    df['Volatility'] = df['Close'].rolling(window=7).std()
    df = df.dropna().reset_index(drop=True)
    return df

@st.cache_resource
def train_models(df):
    features = ['Open', 'High', 'Low', 'Volume', 'Price_Range', 'MA_7', 'MA_30']
    target = 'Close'
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X = scaler_X.fit_transform(df[features])
    y = scaler_y.fit_transform(df[[target]])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False)
    ridge = Ridge(alpha=0.001, fit_intercept=True)
    ridge.fit(X_train, y_train.ravel())
    rf = RandomForestRegressor(n_estimators=200, max_depth=None,
                                max_features='sqrt', random_state=42)
    rf.fit(X_train, y_train.ravel())
    knn = KNeighborsRegressor(n_neighbors=5)
    knn.fit(X_train, y_train.ravel())
    return ridge, rf, knn, scaler_X, scaler_y, X_test, y_test

with st.spinner("Loading AAPL data from Yahoo Finance..."):
    df = load_data()
    ridge, rf, knn, scaler_X, scaler_y, X_test, y_test = train_models(df)

# Sidebar
st.sidebar.title("⚙️ Settings")
model_choice = st.sidebar.selectbox(
    "Choose a model:",
    ["Ridge Regression (Best)", "Random Forest", "KNN Regressor"]
)
st.sidebar.markdown("---")
st.sidebar.info("""
**Course:** MO 3208
**Dataset:** AAPL - Yahoo Finance
**Period:** 2020–2026
**Task:** Regression
""")

# Section 1: Overview
st.header("📊 Dataset Overview")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Records", f"{len(df):,}")
col2.metric("Date Range", "2020 – 2026")
col3.metric("Features Used", "7")
col4.metric("Latest Close", f"${df['Close'].iloc[-1]:.2f}")

with st.expander("View Raw Data"):
    st.dataframe(df[['Date','Open','High','Low','Close','Volume']].tail(20))

# Section 2: Price Chart
st.header("📉 AAPL Closing Price History")
fig1, ax1 = plt.subplots(figsize=(12, 4))
ax1.plot(df['Date'], df['Close'], color='steelblue', linewidth=1.5, label='Close')
ax1.plot(df['Date'], df['MA_7'],  color='orange', linewidth=1, label='MA 7')
ax1.plot(df['Date'], df['MA_30'], color='red', linewidth=1, label='MA 30')
ax1.set_title('AAPL Closing Price with Moving Averages')
ax1.set_xlabel('Date')
ax1.set_ylabel('Price (USD)')
ax1.legend()
ax1.grid(alpha=0.3)
plt.tight_layout()
st.pyplot(fig1)

# Section 3: Model Performance
st.header("🤖 Model Performance Comparison")
y_test_inv = scaler_y.inverse_transform(y_test)

def get_metrics(model, X_test, y_test_inv, scaler_y):
    pred = model.predict(X_test)
    pred_inv = scaler_y.inverse_transform(pred.reshape(-1,1))
    return {
        'MAE': round(mean_absolute_error(y_test_inv, pred_inv), 4),
        'MSE': round(mean_squared_error(y_test_inv, pred_inv), 4),
        'R²':  round(r2_score(y_test_inv, pred_inv), 4),
        'pred_inv': pred_inv
    }

ridge_m = get_metrics(ridge, X_test, y_test_inv, scaler_y)
rf_m    = get_metrics(rf,    X_test, y_test_inv, scaler_y)
knn_m   = get_metrics(knn,   X_test, y_test_inv, scaler_y)

results_df = pd.DataFrame({
    'Model': ['Ridge Regression', 'Random Forest', 'KNN Regressor'],
    'MAE':   [ridge_m['MAE'], rf_m['MAE'], knn_m['MAE']],
    'MSE':   [ridge_m['MSE'], rf_m['MSE'], knn_m['MSE']],
    'R²':    [ridge_m['R²'],  rf_m['R²'],  knn_m['R²']]
})
st.dataframe(results_df, use_container_width=True)

# Chart based on selected model
if model_choice == "Ridge Regression (Best)":
    selected_pred, selected_name, color = ridge_m['pred_inv'], "Ridge Regression", 'steelblue'
elif model_choice == "Random Forest":
    selected_pred, selected_name, color = rf_m['pred_inv'], "Random Forest", 'green'
else:
    selected_pred, selected_name, color = knn_m['pred_inv'], "KNN Regressor", 'red'

fig2, ax2 = plt.subplots(figsize=(12, 4))
ax2.plot(y_test_inv,    label='Actual Price', color='black', linewidth=2)
