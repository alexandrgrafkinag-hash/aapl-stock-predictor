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
    df = yf.download("AAPL", start="2020-01-01", end="2026-04-13")
    df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    df.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
    ridge = Ridge(alpha=0.001, fit_intercept=True)
    ridge.fit(X_train, y_train.ravel())
    rf = RandomForestRegressor(n_estimators=200, max_depth=None, max_features='sqrt', random_state=42)
    rf.fit(X_train, y_train.ravel())
    knn = KNeighborsRegressor(n_neighbors=5)
    knn.fit(X_train, y_train.ravel())
    return ridge, rf, knn, scaler_X, scaler_y, X_test, y_test

with st.spinner("Loading AAPL data from Yahoo Finance..."):
    df = load_data()
    ridge, rf, knn, scaler_X, scaler_y, X_test, y_test = train_models(df)

st.sidebar.title("⚙️ Settings")
st.sidebar.markdown("### Select Model")
model_choice = st.sidebar.selectbox(
    "Choose a model for prediction:",
    ["Ridge Regression (Best)", "Random Forest", "KNN Regressor"]
)
st.sidebar.markdown("---")
st.sidebar.info("""
**Course:** MO 3208 - Machine Learning Algorithms
**Dataset:** Apple Inc. (AAPL)
Yahoo Finance | 2020–2026
**Problem Type:** Regression
**Goal:** Predict AAPL closing price
""")

st.header("📊 Dataset Overview")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Records", f"{len(df):,}")
col2.metric("Date Range", "2020 – 2026")
col3.metric("Features Used", "7")
col4.metric("Latest Close", f"${df['Close'].iloc[-1]:.2f}")

with st.expander("View Raw Data"):
    st.dataframe(df[['Date','Open','High','Low','Close','Volume']].tail(20))

st.header("📉 AAPL Closing Price History")
fig1, ax1 = plt.subplots(figsize=(12, 4))
ax1.plot(df['Date'], df['Close'], color='steelblue', linewidth=1.5, label='Close Price')
ax1.plot(df['Date'], df['MA_7'],  color='orange', linewidth=1, label='MA 7')
ax1.plot(df['Date'], df['MA_30'], color='red', linewidth=1, label='MA 30')
ax1.set_title('AAPL Closing Price with Moving Averages')
ax1.set_xlabel('Date')
ax1.set_ylabel('Price (USD)')
ax1.legend()
ax1.grid(alpha=0.3)
plt.tight_layout()
st.pyplot(fig1)

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
    'Model':  ['Ridge Regression', 'Random Forest', 'KNN Regressor'],
    'MAE':    [ridge_m['MAE'], rf_m['MAE'], knn_m['MAE']],
    'MSE':    [ridge_m['MSE'], rf_m['MSE'], knn_m['MSE']],
    'R²':     [ridge_m['R²'],  rf_m['R²'],  knn_m['R²']]
})
st.dataframe(results_df, use_container_width=True)

if model_choice == "Ridge Regression (Best)":
    selected_pred = ridge_m['pred_inv']
    selected_name = "Ridge Regression"
    color = 'steelblue'
elif model_choice == "Random Forest":
    selected_pred = rf_m['pred_inv']
    selected_name = "Random Forest"
    color = 'green'
else:
    selected_pred = knn_m['pred_inv']
    selected_name = "KNN Regressor"
    color = 'red'

fig2, ax2 = plt.subplots(figsize=(12, 4))
ax2.plot(y_test_inv,    label='Actual Price', color='black', linewidth=2)
ax2.plot(selected_pred, label=f'{selected_name} Prediction', color=color, linewidth=1.5, alpha=0.85)
ax2.set_title(f'{selected_name}: Predicted vs Actual Closing Price')
ax2.set_xlabel('Days (Test Set)')
ax2.set_ylabel('Price (USD)')
ax2.legend()
ax2.grid(alpha=0.3)
plt.tight_layout()
st.pyplot(fig2)

st.header("🔮 Predict Closing Price")
st.markdown("Enter stock data manually to get a predicted closing price:")

col1, col2 = st.columns(2)
with col1:
    open_price = st.number_input("Open Price (USD)",  min_value=50.0,    max_value=500.0,       value=259.0)
    high_price = st.number_input("High Price (USD)",  min_value=50.0,    max_value=500.0,       value=261.0)
with col2:
    low_price  = st.number_input("Low Price (USD)",   min_value=50.0,    max_value=500.0,       value=257.0)
    volume     = st.number_input("Volume",            min_value=1000000, max_value=500000000,   value=28000000, step=1000000)

price_range = high_price - low_price
ma_7  = df['Close'].tail(7).mean()
ma_30 = df['Close'].tail(30).mean()

if st.button("🔮 Predict Closing Price", use_container_width=True):
    input_data   = np.array([[open_price, high_price, low_price, volume, price_range, ma_7, ma_30]])
    input_scaled = scaler_X.transform(input_data)
    if model_choice == "Ridge Regression (Best)":
        model = ridge
    elif model_choice == "Random Forest":
        model = rf
    else:
        model = knn
    pred_scaled = model.predict(input_scaled)
    prediction  = scaler_y.inverse_transform(pred_scaled.reshape(-1,1))[0][0]
    st.success(f"### 📌 Predicted Closing Price: **${prediction:.2f}**")
    c1, c2, c3 = st.columns(3)
    c1.metric("Open Price",       f"${open_price:.2f}")
    c2.metric("Predicted Close",  f"${prediction:.2f}", delta=f"{prediction - open_price:+.2f}")
    c3.metric("Model Used",       selected_name)

st.markdown("---")
st.caption("MO 3208 Machine Learning Algorithms | Astana IT University 2026")
