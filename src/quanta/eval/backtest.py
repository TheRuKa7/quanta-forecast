"""Rolling-origin backtest.

The industry-standard forecasting evaluation is rolling-origin cross-validation:
pick a minimum training window, then at each fold expand (or slide) the
training set by ``step`` and forecast ``horizon`` steps. The per-fold errors
are aggregated into a single summary. This protects against the
pseudo-generalization you get from a single train/test split.

We always run the backtest sequentially: the fold order matters (each fold's
training set is a prefix of the next), and the compute is dominated by the
per-fold ``fit`` anyway — parallelism would buy little.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from quanta.base import BaseForecaster, TimeSeries
from quanta.eval.metrics import mae, mase, rmse, smape

__all__ = ["BacktestResult", "rolling_origin_backtest"]


#: Factory signature: no args, returns a fresh un-fitted forecaster.
#: Using a factory (not a pre-built instance) lets each fold train from
#: a clean slate — critical when the model has mutable internal state.
ForecasterFactory = Callable[[], BaseForecaster]


@dataclass(frozen=True)
class BacktestResult:
    """Per-fold errors + summary stats across folds."""

    model_name: str
    folds: int
    horizon: int
    per_fold: pd.DataFrame  #: rows = folds, cols = metric names
    forecasts: list[np.ndarray] = field(default_factory=list)
    actuals: list[np.ndarray] = field(default_factory=list)

    def summary(self) -> pd.Series:
        """Mean across folds for each metric."""
        return self.per_fold.mean(numeric_only=True)


def rolling_origin_backtest(
    factory: ForecasterFactory,
    series: TimeSeries,
    *,
    horizon: int,
    min_train: int,
    step: int = 1,
    max_folds: int | None = None,
    expanding: bool = True,
    season: int = 1,
    extra_metrics: Sequence[tuple[str, Callable[[np.ndarray, np.ndarray], float]]] = (),
) -> BacktestResult:
    """Run a rolling-origin backtest.

    Parameters
    ----------
    factory:
        Zero-arg callable returning a fresh forecaster. Called once per fold.
    series:
        The full series. Must have ``len(series) >= min_train + horizon``.
    horizon:
        Forecast horizon per fold.
    min_train:
        Size of the training window at fold 0.
    step:
        How many observations to advance the split between folds.
    max_folds:
        Cap on fold count; ``None`` = run every possible fold.
    expanding:
        ``True`` = expanding window (train grows each fold).
        ``False`` = sliding window (train stays ``min_train`` wide).
    season:
        Seasonal period for MASE. Pass 1 if the series isn't seasonal.
    extra_metrics:
        Additional ``(name, fn(y_true, y_pred) -> float)`` pairs to compute
        alongside the defaults (MAE, RMSE, sMAPE, MASE).

    Returns
    -------
    BacktestResult with per-fold and summary metrics.
    """
    n = len(series)
    if min_train < 2:
        raise ValueError("min_train must be >= 2")
    if n < min_train + horizon:
        raise ValueError(
            f"series too short: need >= {min_train + horizon} points, got {n}"
        )
    if step < 1:
        raise ValueError("step must be >= 1")

    rows: list[dict[str, float | int]] = []
    forecasts: list[np.ndarray] = []
    actuals: list[np.ndarray] = []
    model_name = factory().name  # cheap sanity-check of the factory

    origin = min_train
    fold = 0
    while origin + horizon <= n:
        if max_folds is not None and fold >= max_folds:
            break
        train_start = 0 if expanding else origin - min_train
        train_values = series.values[train_start:origin]
        train_index = series.index[train_start:origin]
        train_ts = TimeSeries(
            values=train_values,
            index=train_index,
            freq=series.freq,
            name=series.name,
        )
        actual = series.values[origin : origin + horizon]

        model = factory()
        model.fit(train_ts)
        forecast = model.predict(horizon).point

        row: dict[str, float | int] = {
            "fold": fold,
            "train_end": int(origin),
            "mae": mae(actual, forecast),
            "rmse": rmse(actual, forecast),
            "smape": smape(actual, forecast),
        }
        try:
            row["mase"] = mase(actual, forecast, train_values, season=season)
        except ValueError:
            # Degenerate training scale; leave it NaN rather than crash.
            row["mase"] = float("nan")
        for metric_name, fn in extra_metrics:
            row[metric_name] = float(fn(actual, forecast))
        rows.append(row)
        forecasts.append(forecast)
        actuals.append(actual)
        origin += step
        fold += 1

    per_fold = pd.DataFrame(rows).set_index("fold")
    return BacktestResult(
        model_name=model_name,
        folds=len(rows),
        horizon=horizon,
        per_fold=per_fold,
        forecasts=forecasts,
        actuals=actuals,
    )
