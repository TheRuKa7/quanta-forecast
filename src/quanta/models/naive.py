"""Always-available baseline forecasters.

Baselines matter. M4 / M5 experience is that a well-chosen naive method often
beats a poorly-tuned deep model, and the MASE metric is defined relative to
a seasonal naive. These live in the core (no extras) so every install can
run backtests, CI smoke tests, and MASE.
"""
from __future__ import annotations

import numpy as np

from quanta.base import BaseForecaster, TimeSeries

__all__ = [
    "NaiveForecaster",
    "SeasonalNaiveForecaster",
    "MeanForecaster",
    "DriftForecaster",
]


class NaiveForecaster(BaseForecaster):
    """Forecast = last observed value. The M-competition entry-level bar."""

    name = "naive"

    def __init__(self) -> None:
        super().__init__()
        self._last: float = 0.0

    def _fit(self, series: TimeSeries) -> None:
        self._last = float(series.values[-1])

    def _predict(self, horizon: int) -> np.ndarray:
        return np.full(horizon, self._last, dtype=np.float64)


class SeasonalNaiveForecaster(BaseForecaster):
    """Forecast = value from ``season`` steps ago, repeated.

    ``season`` is the seasonal period in observations (7 for daily data with
    weekly seasonality; 12 for monthly with yearly seasonality).
    """

    name = "seasonal_naive"

    def __init__(self, season: int = 7) -> None:
        super().__init__()
        if season < 1:
            raise ValueError("season must be >= 1")
        self.season = season
        self._last_cycle: np.ndarray = np.empty(0)

    def _fit(self, series: TimeSeries) -> None:
        if len(series) < self.season:
            raise ValueError(
                f"seasonal_naive needs >= season={self.season} observations, "
                f"got {len(series)}"
            )
        self._last_cycle = np.asarray(series.values[-self.season :], dtype=np.float64)

    def _predict(self, horizon: int) -> np.ndarray:
        reps = (horizon // self.season) + 1
        tiled = np.tile(self._last_cycle, reps)
        return tiled[:horizon]


class MeanForecaster(BaseForecaster):
    """Forecast = mean of training series. Floor for flat/trendless data."""

    name = "mean"

    def __init__(self) -> None:
        super().__init__()
        self._mu: float = 0.0

    def _fit(self, series: TimeSeries) -> None:
        self._mu = float(np.mean(series.values))

    def _predict(self, horizon: int) -> np.ndarray:
        return np.full(horizon, self._mu, dtype=np.float64)


class DriftForecaster(BaseForecaster):
    """Hyndman's drift method: last value + average per-step change.

    Equivalent to extrapolating the line through the first and last points.
    A surprisingly competitive baseline for trending series.
    """

    name = "drift"

    def __init__(self) -> None:
        super().__init__()
        self._last: float = 0.0
        self._slope: float = 0.0

    def _fit(self, series: TimeSeries) -> None:
        v = series.values
        if len(v) < 2:
            raise ValueError("drift needs >= 2 observations")
        self._last = float(v[-1])
        self._slope = float((v[-1] - v[0]) / (len(v) - 1))

    def _predict(self, horizon: int) -> np.ndarray:
        steps = np.arange(1, horizon + 1, dtype=np.float64)
        return self._last + self._slope * steps
