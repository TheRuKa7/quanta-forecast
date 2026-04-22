"""Baseline forecasters — naive, seasonal_naive, mean, drift."""
from __future__ import annotations

import numpy as np
import pytest

from quanta.base import TimeSeries
from quanta.models.naive import (
    DriftForecaster,
    MeanForecaster,
    NaiveForecaster,
    SeasonalNaiveForecaster,
)


def test_naive_returns_last_value() -> None:
    m = NaiveForecaster().fit([1.0, 2.0, 3.0, 42.0])
    fc = m.predict(5)
    np.testing.assert_array_equal(fc.point, [42.0] * 5)


def test_seasonal_naive_repeats_last_cycle() -> None:
    m = SeasonalNaiveForecaster(season=3).fit([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
    fc = m.predict(5)
    # Last cycle = [40, 50, 60] → forecast = [40, 50, 60, 40, 50]
    np.testing.assert_array_equal(fc.point, [40.0, 50.0, 60.0, 40.0, 50.0])


def test_seasonal_naive_rejects_short_series() -> None:
    m = SeasonalNaiveForecaster(season=7)
    # After coerce to TimeSeries, len < season should raise.
    with pytest.raises(ValueError, match=">= season"):
        m.fit(list(range(5)))


def test_mean_returns_training_mean() -> None:
    m = MeanForecaster().fit([2.0, 4.0, 6.0])
    fc = m.predict(3)
    np.testing.assert_array_equal(fc.point, [4.0, 4.0, 4.0])


def test_drift_linear_extrapolation() -> None:
    # y = t (slope 1, last = 9, n=10) → forecast_h = 9 + 1*h
    m = DriftForecaster().fit(list(range(10)))
    fc = m.predict(3)
    np.testing.assert_array_equal(fc.point, [10.0, 11.0, 12.0])


def test_forecast_index_is_integer_when_input_is_rangeindex() -> None:
    m = NaiveForecaster().fit([1.0, 2.0, 3.0])
    fc = m.predict(2)
    # RangeIndex projected forward.
    assert list(fc.index) == [3, 4]


def test_forecast_index_is_dated_when_input_is_datetime() -> None:
    import pandas as pd

    ts = TimeSeries.from_array([1.0, 2.0, 3.0], start="2024-01-01", freq="D")
    fc = NaiveForecaster().fit(ts).predict(2)
    assert list(fc.index) == [pd.Timestamp("2024-01-04"), pd.Timestamp("2024-01-05")]


def test_naive_supports_quantiles_is_false() -> None:
    assert NaiveForecaster().supports_quantiles is False
