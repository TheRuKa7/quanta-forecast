"""Streamlit demo — pick a dataset, pick a backend, see a forecast.

The demo talks to the library directly (not the FastAPI service) so it
stays runnable with just ``uv sync --extra demo``. Heavy backends
(Chronos, TFT) are listed in the dropdown but gracefully surface an
ImportError banner if their optional extra isn't installed, so the demo
stays useful on CPU-only / lightweight environments.
"""
from __future__ import annotations

import io
import traceback

import numpy as np
import pandas as pd
import streamlit as st

from quanta import (
    TimeSeries,
    create,
    load_airline_passengers,
    load_csv,
    make_synthetic_seasonal,
)
from quanta.eval.backtest import rolling_origin_backtest
from quanta.eval.metrics import coverage, mae, mase, rmse, smape

st.set_page_config(page_title="quanta-forecast", layout="wide")

st.title("quanta-forecast")
st.caption(
    "Multi-model time-series forecasting — classical · ML · deep · foundation. "
    "Point + probabilistic forecasts behind one unified API."
)


# --- Dataset selector ---------------------------------------------------

def _load_dataset(choice: str, uploaded: io.BytesIO | None) -> TimeSeries:
    if choice == "Airline passengers (monthly, 1949-1960)":
        return load_airline_passengers()
    if choice == "Synthetic seasonal":
        return make_synthetic_seasonal(
            n=240, period=12, amplitude=10.0, trend_slope=0.1, noise=1.0, seed=0
        )
    if choice == "Upload CSV" and uploaded is not None:
        # Save to a tmp file-like path that load_csv can read.
        df = pd.read_csv(uploaded)
        # load_csv expects a path; we replicate its validation inline.
        if "ds" not in df.columns or "y" not in df.columns:
            raise KeyError("Uploaded CSV must have `ds` (date) and `y` (value) columns.")
        df["ds"] = pd.to_datetime(df["ds"])
        df = df.sort_values("ds").set_index("ds")
        freq = pd.infer_freq(df.index)
        return TimeSeries(
            values=df["y"].to_numpy(dtype=np.float64),
            index=pd.DatetimeIndex(df.index),
            freq=freq,
            name="y",
        )
    raise ValueError(f"Unknown dataset choice: {choice}")


# --- Model catalogue ----------------------------------------------------

# (label, registry_key, needs_extra, defaults, supports_quantiles)
_CATALOGUE: list[tuple[str, str, str | None, dict, bool]] = [
    ("Seasonal naive (baseline)", "seasonal_naive", None, {"season": 12}, False),
    ("Drift (baseline)", "drift", None, {}, False),
    ("ARIMA (classical)", "arima", "classical",
        {"order": (1, 1, 1), "seasonal_order": (1, 1, 1, 12), "trend": "c"}, True),
    ("Holt-Winters (classical)", "holt_winters", "classical",
        {"season": 12, "trend": "add", "seasonal": "add"}, True),
    ("LightGBM (ml)", "lightgbm", "ml", {}, True),
    ("Chronos (foundation, zero-shot)", "chronos", "foundation",
        {"model_name": "amazon/chronos-t5-tiny", "num_samples": 50}, True),
]

_CATALOGUE_BY_LABEL = {lbl: row for (lbl, *_), row in zip(_CATALOGUE, _CATALOGUE)}


# --- Sidebar ------------------------------------------------------------

with st.sidebar:
    st.header("Configuration")

    dataset_choice = st.selectbox(
        "Dataset",
        [
            "Airline passengers (monthly, 1949-1960)",
            "Synthetic seasonal",
            "Upload CSV",
        ],
    )
    uploaded = None
    if dataset_choice == "Upload CSV":
        uploaded = st.file_uploader(
            "CSV with columns `ds` (date) and `y` (value)", type="csv"
        )

    labels = [row[0] for row in _CATALOGUE]
    model_label = st.selectbox("Backend", labels, index=0)
    horizon = st.slider("Horizon (steps)", 4, 60, 12)
    show_quantiles = st.checkbox("Probabilistic (quantile) output", value=True)
    run_backtest = st.checkbox("Rolling-origin backtest", value=False)
    if run_backtest:
        min_train = st.slider("Min training window", 24, 200, 60)
        step = st.slider("Fold step", 1, 24, 12)
        max_folds = st.slider("Max folds", 2, 12, 4)

    st.caption("Not financial advice. Markets are near-efficient.")


# --- Main ---------------------------------------------------------------

try:
    series = _load_dataset(dataset_choice, uploaded)
except Exception as e:  # noqa: BLE001
    st.error(f"Failed to load dataset: {e}")
    st.stop()

st.subheader(f"Data — {len(series)} observations @ freq `{series.freq}`")
hist_df = pd.DataFrame(
    {"y": series.values}, index=pd.Index(series.index, name="ds")
)
st.line_chart(hist_df, height=260)

label, key, extra, defaults, supports_q = _CATALOGUE_BY_LABEL[model_label]


def _build_model():
    try:
        return create(key, **defaults)
    except ImportError as e:
        st.warning(
            f"The `{extra}` extra is not installed — this backend is unavailable.\n\n"
            f"`pip install 'quanta-forecast[{extra}]'`\n\n"
            f"Import error: {e}"
        )
        return None
    except Exception as e:  # noqa: BLE001
        st.error(f"Could not build backend `{key}`: {e}")
        return None


col1, col2 = st.columns([2, 1])

with col1:
    st.subheader(f"Forecast — {label}")
    model = _build_model()
    if model is None:
        st.stop()
    try:
        model.fit(series)
        if show_quantiles and supports_q:
            out = model.predict_quantiles(horizon=horizon, quantiles=(0.1, 0.5, 0.9))
        else:
            out = model.predict(horizon=horizon)
    except Exception as e:  # noqa: BLE001
        st.error(f"Fit/predict failed: {e}")
        st.code(traceback.format_exc())
        st.stop()

    hist = pd.DataFrame({"y": series.values}, index=pd.Index(series.index, name="ds"))
    fc_df = pd.DataFrame({"forecast": out.point}, index=pd.Index(out.index, name="ds"))
    if out.quantiles:
        fc_df["q10"] = out.quantiles[0.1]
        fc_df["q50"] = out.quantiles[0.5]
        fc_df["q90"] = out.quantiles[0.9]

    plot_df = pd.concat([hist, fc_df], axis=1)
    st.line_chart(plot_df, height=360)

with col2:
    st.subheader("Last-fold sanity check")
    # Simple holdout: last `horizon` observations.
    if len(series) <= horizon + 12:
        st.info("Series too short for a holdout sanity check.")
    else:
        holdout_actual = series.values[-horizon:]
        train_ts = TimeSeries(
            values=series.values[:-horizon],
            index=series.index[:-horizon],
            freq=series.freq,
            name=series.name,
        )
        sanity = _build_model()
        if sanity is not None:
            try:
                sanity.fit(train_ts)
                sanity_out = sanity.predict(horizon=horizon)
                metrics = {
                    "MAE": mae(holdout_actual, sanity_out.point),
                    "RMSE": rmse(holdout_actual, sanity_out.point),
                    "sMAPE (%)": smape(holdout_actual, sanity_out.point),
                }
                try:
                    metrics["MASE"] = mase(
                        holdout_actual,
                        sanity_out.point,
                        train_ts.values,
                        season=12 if series.freq and "M" in str(series.freq) else 1,
                    )
                except ValueError:
                    metrics["MASE"] = float("nan")
                if supports_q and show_quantiles:
                    q_sanity = sanity.predict_quantiles(
                        horizon=horizon, quantiles=(0.1, 0.9)
                    )
                    metrics["Coverage@80"] = coverage(
                        holdout_actual, q_sanity.quantiles[0.1], q_sanity.quantiles[0.9]
                    )
                st.dataframe(
                    pd.DataFrame(
                        {"metric": list(metrics.keys()), "value": list(metrics.values())}
                    ),
                    hide_index=True,
                    use_container_width=True,
                )
            except Exception as e:  # noqa: BLE001
                st.warning(f"Sanity check failed: {e}")


if run_backtest:
    st.divider()
    st.subheader(
        f"Rolling-origin backtest — horizon={horizon}, min_train={min_train}, "
        f"step={step}, max_folds={max_folds}"
    )
    try:
        result = rolling_origin_backtest(
            factory=lambda: create(key, **defaults),
            series=series,
            horizon=horizon,
            min_train=min_train,
            step=step,
            max_folds=max_folds,
            season=12 if series.freq and "M" in str(series.freq) else 1,
        )
        st.write(f"**Folds run:** {result.folds}")
        st.dataframe(result.per_fold, use_container_width=True)
        st.write("**Summary (mean across folds)**")
        st.dataframe(result.summary().to_frame("value"))
    except Exception as e:  # noqa: BLE001
        st.error(f"Backtest failed: {e}")
        st.code(traceback.format_exc())

st.divider()
st.caption(
    "Backends available in this install: "
    + ", ".join(f"`{b}`" for b in __import__("quanta").list_backends())
)
