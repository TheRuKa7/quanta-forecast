"""Lag + rolling + calendar feature construction for the LightGBM backend.

Design notes:

* The builder is stateless: it takes the full series and config and returns
  a dense frame. That frame is trained on (with NaN rows dropped) for
  supervised fitting, and then re-fed row-by-row at predict time for
  recursive forecasting.
* We use ``pd.Series.shift`` for lags rather than manual indexing because
  shift is defined around a DatetimeIndex and correctly propagates NaNs
  for the head of the series — no off-by-one hazards.
* Rolling windows are always computed on ``shift(1)`` of the series so a
  row's rolling mean never contains the target (no leakage).
* Calendar features (dayofweek, month, quarter) are emitted only when the
  index is a ``DatetimeIndex``; integer-indexed series skip them.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from quanta.base import TimeSeries

__all__ = ["LagFeatureBuilder", "make_lag_frame"]


@dataclass(frozen=True)
class LagFeatureBuilder:
    """Configuration for lag / rolling / calendar feature construction.

    ``lags`` and ``rolling_windows`` are lists of positive integers measured
    in observations (not wall-clock time). ``calendar`` toggles the datetime
    features; it's a no-op on integer-indexed series.
    """

    lags: Sequence[int] = field(default_factory=lambda: (1, 7, 14, 28))
    rolling_windows: Sequence[int] = field(default_factory=lambda: (7, 28))
    calendar: bool = True

    def build(self, ts: TimeSeries) -> pd.DataFrame:
        """Return a DataFrame with columns ``y`` plus every engineered feature.

        The caller is responsible for dropping NaN rows (the head of the
        series) before fitting, but NOT before making predictions — at
        predict time we need the last row with its features filled in.
        """
        if any(lag < 1 for lag in self.lags):
            raise ValueError("lags must be >= 1")
        if any(w < 1 for w in self.rolling_windows):
            raise ValueError("rolling_windows must be >= 1")

        df = pd.DataFrame({"y": ts.values}, index=ts.index)
        for lag in self.lags:
            df[f"lag_{lag}"] = df["y"].shift(lag)
        for window in self.rolling_windows:
            df[f"rollmean_{window}"] = df["y"].shift(1).rolling(window).mean()
            df[f"rollstd_{window}"] = df["y"].shift(1).rolling(window).std()
        if self.calendar and isinstance(ts.index, pd.DatetimeIndex):
            df["dayofweek"] = ts.index.dayofweek
            df["month"] = ts.index.month
            df["quarter"] = ts.index.quarter
            df["is_month_start"] = ts.index.is_month_start.astype(np.int8)
            df["is_month_end"] = ts.index.is_month_end.astype(np.int8)
        return df

    def feature_columns(self, ts: TimeSeries) -> list[str]:
        """Column order used for the X matrix at fit + predict time."""
        cols = [f"lag_{lag}" for lag in self.lags]
        for window in self.rolling_windows:
            cols.append(f"rollmean_{window}")
            cols.append(f"rollstd_{window}")
        if self.calendar and isinstance(ts.index, pd.DatetimeIndex):
            cols += ["dayofweek", "month", "quarter", "is_month_start", "is_month_end"]
        return cols


def make_lag_frame(ts: TimeSeries, lags: Sequence[int] = (1, 7, 14)) -> pd.DataFrame:
    """Convenience: build a lag-only frame for ad-hoc use."""
    return LagFeatureBuilder(lags=lags, rolling_windows=(), calendar=False).build(ts)
