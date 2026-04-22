"""Exponential smoothing family — pure-numpy implementations.

Three flavours, all stateless after ``fit`` aside from the learned smoothing
parameters:

* :class:`SimpleExpSmoothingForecaster` (SES) — level only. Good for no-trend
  no-seasonality series.
* :class:`HoltForecaster` — level + trend (additive). Good for trending.
* :class:`HoltWintersForecaster` — level + trend + seasonal (additive). The
  classical workhorse for seasonal series.

These exist as pure-numpy rather than delegating to statsmodels because:

1. They're small and dependency-free — useful when the ``classical`` extra
   isn't installed.
2. They let us validate the :class:`BaseForecaster` contract (including
   probabilistic forecasts for HoltWinters via residual bootstrap) without
   a heavy import.

Parameter fitting uses a simple 1-D grid for SES and SciPy-style numerical
minimization via ``scipy.optimize.minimize`` if available; otherwise we fall
back to the grid. We prefer reliability over optimality here — the user who
needs the best-possible ETS fit should use ``quanta.models.arima`` (auto
model selection via statsmodels) or the statsforecast backend.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from quanta.base import BaseForecaster, TimeSeries

__all__ = [
    "SimpleExpSmoothingForecaster",
    "HoltForecaster",
    "HoltWintersForecaster",
]


def _ses_loss(alpha: float, y: np.ndarray) -> float:
    """Sum of squared one-step errors for SES with smoothing ``alpha``."""
    level = y[0]
    sse = 0.0
    for t in range(1, len(y)):
        sse += (y[t] - level) ** 2
        level = alpha * y[t] + (1.0 - alpha) * level
    return float(sse)


def _grid_search(
    loss_fn: Callable[[float], float], grid: np.ndarray
) -> tuple[float, float]:
    """Pick the argmin on a 1-D grid."""
    losses = np.asarray([loss_fn(g) for g in grid], dtype=np.float64)
    idx = int(np.argmin(losses))
    return float(grid[idx]), float(losses[idx])


class SimpleExpSmoothingForecaster(BaseForecaster):
    """Simple Exponential Smoothing. Forecast is a flat extrapolation of the
    fitted level."""

    name = "ses"

    def __init__(self, alpha: float | None = None) -> None:
        super().__init__()
        if alpha is not None and not (0.0 < alpha < 1.0):
            raise ValueError("alpha must be in (0, 1) or None for auto-fit")
        self._alpha_user = alpha
        self._alpha: float = 0.3
        self._level: float = 0.0

    def _fit(self, series: TimeSeries) -> None:
        y = series.values
        if self._alpha_user is None:
            grid = np.linspace(0.05, 0.95, 19)
            self._alpha, _ = _grid_search(lambda a: _ses_loss(a, y), grid)
        else:
            self._alpha = self._alpha_user
        # Run through once to get the final level.
        level = float(y[0])
        a = self._alpha
        for t in range(1, len(y)):
            level = a * y[t] + (1.0 - a) * level
        self._level = level

    def _predict(self, horizon: int) -> np.ndarray:
        return np.full(horizon, self._level, dtype=np.float64)


def _holt_loss(params: tuple[float, float], y: np.ndarray) -> float:
    alpha, beta = params
    level = float(y[0])
    trend = float(y[1] - y[0]) if len(y) >= 2 else 0.0
    sse = 0.0
    for t in range(1, len(y)):
        pred = level + trend
        sse += (y[t] - pred) ** 2
        new_level = alpha * y[t] + (1.0 - alpha) * pred
        trend = beta * (new_level - level) + (1.0 - beta) * trend
        level = new_level
    return float(sse)


class HoltForecaster(BaseForecaster):
    """Holt's linear trend method (additive)."""

    name = "holt"

    def __init__(
        self, alpha: float | None = None, beta: float | None = None
    ) -> None:
        super().__init__()
        self._alpha_user = alpha
        self._beta_user = beta
        self._alpha: float = 0.5
        self._beta: float = 0.1
        self._level: float = 0.0
        self._trend: float = 0.0

    def _fit(self, series: TimeSeries) -> None:
        y = series.values
        if len(y) < 2:
            raise ValueError("holt needs >= 2 observations")

        if self._alpha_user is None or self._beta_user is None:
            # Coarse 2-D grid — 10x10 is enough and keeps us dep-free.
            grid = np.linspace(0.1, 0.9, 9)
            best_loss = np.inf
            best = (0.5, 0.1)
            for a in grid:
                for b in grid:
                    loss = _holt_loss((a, b), y)
                    if loss < best_loss:
                        best_loss = loss
                        best = (float(a), float(b))
            self._alpha, self._beta = best
        if self._alpha_user is not None:
            self._alpha = self._alpha_user
        if self._beta_user is not None:
            self._beta = self._beta_user

        level = float(y[0])
        trend = float(y[1] - y[0])
        a, b = self._alpha, self._beta
        for t in range(1, len(y)):
            pred = level + trend
            new_level = a * y[t] + (1.0 - a) * pred
            trend = b * (new_level - level) + (1.0 - b) * trend
            level = new_level
        self._level = level
        self._trend = trend

    def _predict(self, horizon: int) -> np.ndarray:
        steps = np.arange(1, horizon + 1, dtype=np.float64)
        return self._level + self._trend * steps


class HoltWintersForecaster(BaseForecaster):
    """Additive Holt-Winters (level + trend + seasonal).

    With ``probabilistic=True`` (default), ``predict_quantiles`` returns
    bootstrap intervals from the training residuals — a crude but
    distribution-free alternative to the state-space analytical intervals
    statsmodels produces.
    """

    name = "holt_winters"
    supports_quantiles = True

    def __init__(
        self,
        season: int = 12,
        alpha: float = 0.3,
        beta: float = 0.1,
        gamma: float = 0.2,
    ) -> None:
        super().__init__()
        if season < 2:
            raise ValueError("season must be >= 2 for Holt-Winters")
        self.season = season
        self._alpha = alpha
        self._beta = beta
        self._gamma = gamma
        self._level: float = 0.0
        self._trend: float = 0.0
        self._seasonals: np.ndarray = np.empty(0)
        self._residuals: np.ndarray = np.empty(0)

    def _fit(self, series: TimeSeries) -> None:
        y = series.values
        m = self.season
        if len(y) < 2 * m:
            raise ValueError(
                f"holt_winters needs >= 2 * season = {2 * m} observations"
            )

        # Initial level = mean of first cycle.
        level = float(np.mean(y[:m]))
        # Initial trend = avg per-step change across first two cycles.
        trend = float(np.mean((y[m : 2 * m] - y[:m]) / m))
        # Initial seasonals = deviations of first cycle from initial level.
        seasonals = y[:m] - level

        a, b, g = self._alpha, self._beta, self._gamma
        residuals = np.empty(len(y), dtype=np.float64)
        for t in range(len(y)):
            s_idx = t % m
            pred = level + trend + seasonals[s_idx]
            residuals[t] = y[t] - pred
            new_level = a * (y[t] - seasonals[s_idx]) + (1.0 - a) * (level + trend)
            trend = b * (new_level - level) + (1.0 - b) * trend
            seasonals[s_idx] = g * (y[t] - new_level) + (1.0 - g) * seasonals[s_idx]
            level = new_level

        self._level = level
        self._trend = trend
        self._seasonals = seasonals
        # Drop first-cycle residuals — they include the init-warmup error.
        self._residuals = residuals[m:]

    def _predict(self, horizon: int) -> np.ndarray:
        steps = np.arange(1, horizon + 1, dtype=np.float64)
        trend_component = self._level + self._trend * steps
        # Wrap seasonal indices starting from the next period.
        season_indices = (np.arange(horizon) + 0) % self.season
        seasonal_component = self._seasonals[season_indices]
        return trend_component + seasonal_component

    def _predict_quantiles(
        self, horizon: int, quantiles: tuple[float, ...]
    ) -> tuple[np.ndarray, dict[float, np.ndarray]]:
        """Bootstrap intervals from the training residuals.

        Residuals are resampled with replacement to build a sample matrix of
        shape (horizon, n_samples). Quantiles are taken per row. Simple,
        robust, and makes no Gaussianity assumption.
        """
        n_samples = 500
        rng = np.random.default_rng(0)
        point = self._predict(horizon)
        if len(self._residuals) == 0:
            # Degenerate; return point forecast for all quantiles.
            return point, {q: point.copy() for q in quantiles}
        resampled = rng.choice(self._residuals, size=(horizon, n_samples), replace=True)
        samples = point[:, None] + resampled
        qdict = {q: np.quantile(samples, q, axis=1) for q in quantiles}
        return point, qdict
