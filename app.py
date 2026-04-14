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
    df = None

    # Способ 1: yf.download
    try:
        raw = yf.download(
            "AAPL",
            start="2020-01-01",
            end="2026-04-13",
            auto_adjust=True,
            progress=False,
            threads=False
        )
        if raw is not None and not raw.empty:
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            raw = raw.reset_index()
            raw['Date'] = pd.to_datetime(raw['Date']).dt.tz_localize(None)
            df = raw[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].copy()
    except Exception:
        pass

    # Способ 2: ticker.history (если первый не сработал)
    if d
