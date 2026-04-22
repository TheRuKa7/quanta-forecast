"""Streamlit demo — upload CSV, pick model, see forecast."""
from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="quanta-forecast", layout="wide")

st.title("📈 quanta-forecast")
st.caption("Multi-model time-series forecasting — classical · ML · deep · foundation.")

st.info(
    "🚧 Demo UI scaffold. Full functionality ships in P5 "
    "(see [PLAN.md](https://github.com/TheRuKa7/quanta-forecast/blob/main/docs/PLAN.md))."
)

with st.sidebar:
    st.header("Configuration")
    model = st.selectbox(
        "Model",
        ["chronos (foundation)", "tft (deep)", "lightgbm (ml)", "arima (classical)"],
    )
    horizon = st.slider("Horizon (steps)", 7, 90, 14)
    st.caption("Not financial advice.")

col1, col2 = st.columns([2, 1])
with col1:
    st.subheader("Forecast")
    st.write("Upload a CSV with a datetime index and one target column to begin.")
with col2:
    st.subheader("Metrics")
    st.write("sMAPE, MASE, CRPS, coverage@80/95 will appear here after backtest.")
