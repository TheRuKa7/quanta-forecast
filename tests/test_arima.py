"""ARIMA backend via statsmodels.

Skipped when statsmodels isn't installed.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("statsmodels")

from quanta.data.loaders import load_airline_passengers
from quanta.eval.metrics import mae
from quanta.models.arima import ARIMAForecaster
from quanta.models.naive import SeasonalNaiveForecaster


def test_arima_fits_and_predicts_shape() -> None:
    ts = load_airline_passengers()
    m = ARIMAForecaster(order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    m.fit(ts)
    fc = m.predict(24)
    assert fc.point.shape == (24,)
    assert fc.model_name == "arima"


def test_arima_quantiles_bracket_point() -> None:
    ts = load_airline_passengers()
    m = ARIMAForecaster(order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)).fit(ts)
    fcq = m.predict_quantiles(12, quantiles=(0.1, 0.5, 0.9))
    assert fcq.quantiles is not None
    lo = fcq.quantiles[0.1]
    md = fcq.quantiles[0.5]
    hi = fcq.quantiles[0.9]
    assert np.all(lo <= md + 1e-6)
    assert np.all(md <= hi + 1e-6)


def test_arima_competitive_with_seasonal_naive_on_airline() -> None:
    ts = load_airline_passengers()
    train = ts.values[:-24]
    test = ts.values[-24:]
    from quanta.base import TimeSeries

    train_ts = TimeSeries(
        values=train, index=ts.index[:-24], freq=ts.freq, name=ts.name
    )
    a = (
        ARIMAForecaster(order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
        .fit(train_ts)
        .predict(24)
        .point
    )
    s = SeasonalNaiveForecaster(season=12).fit(train_ts).predict(24).point
    # SARIMA should be at least within 40% of seasonal naive on MAE — this
    # is a soft bound because small-sample ARIMA fits vary; we just want to
    # catch a catastrophic regression (e.g. all-zeros forecast).
    assert mae(test, a) < 1.4 * mae(test, s)
