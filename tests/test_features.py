"""Lag / rolling / calendar feature construction."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quanta.base import TimeSeries
from quanta.features.lag import LagFeatureBuilder, make_lag_frame


def test_lag_frame_has_expected_columns() -> None:
    ts = TimeSeries.from_array(np.arange(30, dtype=float), start="2024-01-01", freq="D")
    builder = LagFeatureBuilder(lags=(1, 7), rolling_windows=(3,), calendar=True)
    df = builder.build(ts)
    assert "lag_1" in df.columns
    assert "lag_7" in df.columns
    assert "rollmean_3" in df.columns
    assert "rollstd_3" in df.columns
    assert "dayofweek" in df.columns
    assert "month" in df.columns


def test_lag_values_are_correct() -> None:
    ts = TimeSeries.from_array(np.arange(10, dtype=float), start="2024-01-01", freq="D")
    df = make_lag_frame(ts, lags=(1, 2))
    assert pd.isna(df["lag_1"].iloc[0])
    assert df["lag_1"].iloc[1] == 0.0
    assert df["lag_1"].iloc[5] == 4.0
    assert df["lag_2"].iloc[2] == 0.0


def test_rolling_excludes_current_row() -> None:
    """Rolling is computed on shift(1) — a row's rolling mean must not include y."""
    ts = TimeSeries.from_array(np.arange(10, dtype=float), start="2024-01-01", freq="D")
    df = LagFeatureBuilder(
        lags=(1,), rolling_windows=(3,), calendar=False
    ).build(ts)
    # At index 4 (value=4), rollmean_3 must be mean of {y[1], y[2], y[3]} = 2.0.
    assert df["rollmean_3"].iloc[4] == 2.0


def test_calendar_skipped_on_range_index() -> None:
    ts = TimeSeries.from_array(np.arange(10, dtype=float))  # range index
    df = LagFeatureBuilder(calendar=True).build(ts)
    assert "dayofweek" not in df.columns


def test_invalid_lags_rejected() -> None:
    with pytest.raises(ValueError, match="lags"):
        LagFeatureBuilder(lags=(0, 1)).build(
            TimeSeries.from_array([1.0, 2.0, 3.0])
        )


def test_feature_columns_matches_built_frame() -> None:
    ts = TimeSeries.from_array(np.arange(20, dtype=float), start="2024-01-01", freq="D")
    builder = LagFeatureBuilder(lags=(1, 7), rolling_windows=(3,), calendar=True)
    df = builder.build(ts)
    cols = builder.feature_columns(ts)
    # Every declared feature must exist in the frame.
    for c in cols:
        assert c in df.columns
