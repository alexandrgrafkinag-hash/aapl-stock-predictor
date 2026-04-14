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
    raw = yf.download("AAPL", start="2020-01-01", end="2026-04-13", auto_adjust=True)

    # ✅ Проверка что данные пришли
    if raw is None or raw.empty:
        return None

    # ✅ Исправляем MultiIndex колонки
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = raw.reset_index()
    df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].copy()

    # Инженерия признаков
    df['Price_Range'] = df['High'] - df['Low']
    df['Price_Change'] = df['Close'] - df['Open']
    df['MA_7'] = df['Close'].rolling(window=7).mean()
    df['MA_30'] = df['Close'].rolling(window=30).mean()
    df['Volatility'] = df['Close'].rolling(window=7).std()
    df = df.dropna().reset_index(drop=True)
    return df

@st.cache_resource
def train_models(_df):
    features = ['Open', 'High', 'Low', 'Volume', 'Price_Range', 'Price_Change', 'MA_7', 'MA_30', 'Volatility']
    target = 'Close'

    X = _df[features].values
    y = _df[target].values

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

    # Обучение моделей
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)

    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    knn = KNeighborsRegressor(n_neighbors=5)
    knn.fit(X_train, y_train)

    return ridge, rf, knn, scaler_X, scaler_y, X_test, y_test

def get_metrics(model, X_test, y_test, scaler_y):
    y_pred_scaled = model.predict(X_test)
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
    y_true = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2, y_true, y_pred

# ──────────────────────────────────────────────
# ЗАГРУЗКА ДАННЫХ
# ──────────────────────────────────────────────
with st.spinner("Загружаем данные AAPL с Yahoo Finance..."):
    df = load_data()

if df is None or df.empty:
    st.error("❌ Не удалось загрузить данные. Проверьте интернет-соединение или попробуйте позже.")
    st.stop()

st.success(f"✅ Данные загружены: {len(df)} строк")

# ──────────────────────────────────────────────
# ПРОСМОТР ДАННЫХ
# ──────────────────────────────────────────────
with st.expander("📊 Просмотр данных"):
    st.dataframe(df.tail(20), use_container_width=True)

# ──────────────────────────────────────────────
# ГРАФИК ЦЕН
# ──────────────────────────────────────────────
st.subheader("📈 История цен закрытия AAPL")
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(df['Date'], df['Close'], color='royalblue', linewidth=1.5, label='Close Price')
ax.plot(df['Date'], df['MA_7'], color='orange', linewidth=1, linestyle='--', label='MA 7')
ax.plot(df['Date'], df['MA_30'], color='green', linewidth=1, linestyle='--', label='MA 30')
ax.set_xlabel("Дата")
ax.set_ylabel("Цена ($)")
ax.legend()
ax.grid(True, alpha=0.3)
st.pyplot(fig)

# ──────────────────────────────────────────────
# ОБУЧЕНИЕ МОДЕЛЕЙ
# ──────────────────────────────────────────────
st.subheader("🤖 Сравнение моделей машинного обучения")

with st.spinner("Обучаем модели..."):
    ridge, rf, knn, scaler_X, scaler_y, X_test, y_test = train_models(df)

model_names = ["Ridge Regression", "Random Forest", "KNN"]
models = [ridge, rf, knn]

results = []
predictions = {}

for name, model in zip(model_names, models):
    mae, rmse, r2, y_true, y_pred = get_metrics(model, X_test, y_test, scaler_y)
    results.append({"Модель": name, "MAE": round(mae, 4), "RMSE": round(rmse, 4), "R²": round(r2, 4)})
    predictions[name] = (y_true, y_pred)

results_df = pd.DataFrame(results)
st.dataframe(results_df, use_container_width=True)

# ──────────────────────────────────────────────
# ГРАФИКИ ПРЕДСКАЗАНИЙ
# ──────────────────────────────────────────────
st.subheader("🎯 Предсказания vs Реальные значения")

fig, axes = plt.subplots(1, 3, figsize=(16, 4))
colors = ['royalblue', 'green', 'orange']

for ax, (name, (y_true, y_pred)), color in zip(axes, predictions.items(), colors):
    ax.scatter(y_true, y_pred, alpha=0.4, color=color, s=10)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1)
    ax.set_title(name)
    ax.set_xlabel("Реальное")
    ax.set_ylabel("Предсказанное")
    ax.grid(True, alpha=0.3)

plt.tight_layout()
st.pyplot(fig)

# ──────────────────────────────────────────────
# ВЫВОД ЛУЧШЕЙ МОДЕЛИ
# ──────────────────────────────────────────────
best = results_df.loc[results_df['R²'].idxmax()]
st.success(f"🏆 Лучшая модель по R²: **{best['Модель']}** | R² = {best['R²']} | MAE = {best['MAE']}")
