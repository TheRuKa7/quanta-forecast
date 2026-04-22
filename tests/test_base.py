"""Base contract: TimeSeries validation, ForecastOutput guarantees, fitted-state."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quanta.base import (
    DEFAULT_QUANTILES,
    BaseForecaster,
    ForecastOutput,
    TimeSeries,
)


class _Const(BaseForecaster):
    """Minimal subclass used to exercise the base class plumbing."""

    name = "const"

    def _fit(self, series):
        self._last = float(series.values[-1])

    def _predict(self, horizon):
        return np.full(horizon, self._last)


def test_timeseries_from_array_range_index() -> None:
    ts = TimeSeries.from_array([1.0, 2.0, 3.0])
    assert len(ts) == 3
    assert isinstance(ts.index, pd.RangeIndex)
    assert ts.freq is None


def test_timeseries_from_array_with_freq_gives_datetime_index() -> None:
    ts = TimeSeries.from_array([1, 2, 3], start="2024-01-01", freq="D")
    assert isinstance(ts.index, pd.DatetimeIndex)
    assert ts.freq == "D"


def test_timeseries_rejects_non_1d() -> None:
    with pytest.raises(ValueError, match="1-D"):
        TimeSeries(values=np.zeros((3, 3)), index=pd.RangeIndex(3))


def test_timeseries_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError, match="length mismatch"):
        TimeSeries(values=np.zeros(5), index=pd.RangeIndex(4))


def test_future_index_datetime() -> None:
    ts = TimeSeries.from_array([1, 2, 3], start="2024-01-01", freq="D")
    idx = ts.future_index(3)
    assert list(idx) == [
        pd.Timestamp("2024-01-04"),
        pd.Timestamp("2024-01-05"),
        pd.Timestamp("2024-01-06"),
    ]


def test_future_index_range() -> None:
    ts = TimeSeries.from_array([10.0, 11.0, 12.0])
    idx = ts.future_index(2)
    assert list(idx) == [3, 4]


def test_future_index_datetime_requires_freq() -> None:
    # Intentionally construct a DatetimeIndex without setting freq.
    ts = TimeSeries(
        values=np.array([1.0, 2.0]),
        index=pd.DatetimeIndex(["2024-01-01", "2024-01-02"]),
        freq=None,
    )
    with pytest.raises(ValueError, match="freq must be set"):
        ts.future_index(1)


def test_forecast_output_validates_horizon() -> None:
    with pytest.raises(ValueError, match="doesn't match horizon"):
        ForecastOutput(
            point=np.zeros(3),
            index=pd.RangeIndex(3),
            model_name="x",
            horizon=4,
        )


def test_forecast_output_validates_quantile_range() -> None:
    with pytest.raises(ValueError, match=r"in .0, 1"):
        ForecastOutput(
            point=np.zeros(3),
            index=pd.RangeIndex(3),
            model_name="x",
            horizon=3,
            quantiles={1.5: np.zeros(3)},
        )


def test_forecast_output_to_frame_has_point_and_quantiles() -> None:
    out = ForecastOutput(
        point=np.arange(3, dtype=float),
        index=pd.RangeIndex(3),
        model_name="x",
        horizon=3,
        quantiles={0.1: np.zeros(3), 0.9: np.ones(3) * 9},
    )
    df = out.to_frame()
    assert list(df.columns) == ["point", "q10", "q90"]
    assert df.shape == (3, 3)


def test_predict_before_fit_raises() -> None:
    m = _Const()
    with pytest.raises(RuntimeError, match="call fit"):
        m.predict(3)


def test_predict_quantiles_raises_when_unsupported(short_ts) -> None:
    m = _Const().fit(short_ts)
    with pytest.raises(NotImplementedError, match="does not support quantile"):
        m.predict_quantiles(3)


def test_default_quantiles_shape() -> None:
    assert DEFAULT_QUANTILES == (0.1, 0.5, 0.9)


def test_fit_accepts_numpy(short_ts) -> None:
    m = _Const().fit(np.array([1.0, 2.0, 3.0]))
    fc = m.predict(2)
    assert fc.point.tolist() == [3.0, 3.0]


def test_save_and_load_roundtrip(tmp_path, short_ts) -> None:
    m = _Const().fit(short_ts)
    path = m.save(tmp_path / "model.pkl")
    loaded = BaseForecaster.load(path)
    assert type(loaded) is _Const
    # The loaded model must still predict the same value.
    np.testing.assert_allclose(loaded.predict(3).point, m.predict(3).point)


def test_fit_needs_at_least_two_observations() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        _Const().fit(np.array([42.0]))
