"""LightGBM backend using a recursive multi-step forecasting strategy.

Strategy trade-off:

* **Recursive**: train a single 1-step model, then feed predictions back in
  to generate longer horizons. Pros: one model, infinite horizon. Cons:
  errors compound.
* **Direct**: train one model per horizon step. Pros: each step optimized
  independently. Cons: H models to train, no reuse.

We pick **recursive** for the base class because it's the simpler default
and it composes cleanly with the :class:`LagFeatureBuilder`. A direct
variant (:class:`DirectLightGBMForecaster`) subclasses to train H models.

Quantile forecasting is supported via LightGBM's native ``objective="quantile"``
— we train one model per requested quantile. That's honest probabilistic
output at the cost of one train per q.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from quanta.base import BaseForecaster, TimeSeries
from quanta.features.lag import LagFeatureBuilder

__all__ = ["LightGBMForecaster", "DirectLightGBMForecaster"]


_DEFAULT_PARAMS: dict[str, Any] = {
    "objective": "regression",
    "metric": "mae",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 5,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "bagging_freq": 5,
    "verbose": -1,
}


def _import_lightgbm():
    try:
        import lightgbm as lgb  # noqa: PLC0415
    except ImportError as e:  # pragma: no cover - dev-only path
        raise ImportError(
            "LightGBMForecaster requires lightgbm. Install the `ml` extra: "
            "`pip install 'quanta-forecast[ml]'`."
        ) from e
    return lgb


class LightGBMForecaster(BaseForecaster):
    """Recursive-strategy LightGBM regressor.

    Parameters
    ----------
    features:
        ``LagFeatureBuilder`` instance controlling what features get
        engineered. A sensible default covers weekly + 4-weekly lags.
    params:
        LightGBM params merged over the built-in defaults.
    num_boost_round:
        Number of boosting rounds. Keep small by default — forecasting
        datasets are usually small and overfit easily.
    """

    name = "lightgbm"
    supports_quantiles = True

    def __init__(
        self,
        features: LagFeatureBuilder | None = None,
        params: dict[str, Any] | None = None,
        num_boost_round: int = 200,
    ) -> None:
        super().__init__()
        self.features = features or LagFeatureBuilder()
        self.params = {**_DEFAULT_PARAMS, **(params or {})}
        self.num_boost_round = num_boost_round
        self._model: Any = None  # lightgbm.Booster
        self._quantile_models: dict[float, Any] = {}
        self._feature_cols: list[str] = []
        self._history: np.ndarray = np.empty(0)
        self._history_index: pd.DatetimeIndex | pd.RangeIndex | None = None
        self._freq: str | None = None

    def _fit(self, series: TimeSeries) -> None:
        lgb = _import_lightgbm()
        frame = self.features.build(series)
        self._feature_cols = self.features.feature_columns(series)
        train = frame.dropna()
        if len(train) < 10:
            raise ValueError(
                f"lightgbm: after lag construction only {len(train)} rows "
                "remain; increase training length or reduce lag depth"
            )
        X = train[self._feature_cols].to_numpy(dtype=np.float64)
        y = train["y"].to_numpy(dtype=np.float64)
        dtrain = lgb.Dataset(X, label=y)
        self._model = lgb.train(
            params=self.params,
            train_set=dtrain,
            num_boost_round=self.num_boost_round,
        )
        # Keep history for recursive prediction.
        self._history = np.asarray(series.values, dtype=np.float64).copy()
        self._history_index = series.index
        self._freq = series.freq

    def _predict(self, horizon: int) -> np.ndarray:
        assert self._model is not None
        preds = np.empty(horizon, dtype=np.float64)
        working_values = self._history.copy()
        working_index = self._history_index
        assert working_index is not None
        for h in range(horizon):
            # Extend by one synthetic step so the feature builder has an index.
            if isinstance(working_index, pd.DatetimeIndex):
                assert self._freq is not None
                next_ts = working_index[-1] + pd.tseries.frequencies.to_offset(
                    self._freq
                )
                new_index = working_index.append(pd.DatetimeIndex([next_ts]))
            else:
                new_index = pd.RangeIndex(start=0, stop=len(working_values) + 1)
            # Extend values with a NaN placeholder — the row we want to predict.
            extended_values = np.concatenate([working_values, [np.nan]])
            synthetic_ts = TimeSeries(
                values=np.where(np.isnan(extended_values), 0.0, extended_values),
                index=new_index,
                freq=self._freq,
                name="_predict",
            )
            frame = self.features.build(synthetic_ts)
            X_last = frame[self._feature_cols].iloc[[-1]].to_numpy(dtype=np.float64)
            yhat = float(self._model.predict(X_last)[0])
            preds[h] = yhat
            # Feed prediction back as next observation.
            working_values = np.concatenate([working_values, [yhat]])
            working_index = new_index
        return preds

    def _predict_quantiles(
        self, horizon: int, quantiles: tuple[float, ...]
    ) -> tuple[np.ndarray, dict[float, np.ndarray]]:
        """Quantile regression: one LightGBM model per quantile level.

        We train on demand the first time a quantile is requested, then
        cache the booster. That keeps the constructor cheap and avoids
        training quantile models when only point forecasts are needed.
        """
        lgb = _import_lightgbm()
        point = self._predict(horizon)
        qdict: dict[float, np.ndarray] = {}
        # We need a training frame; rebuild from the stored history.
        assert self._history_index is not None
        train_ts = TimeSeries(
            values=self._history,
            index=self._history_index,
            freq=self._freq,
            name="_qtrain",
        )
        frame = self.features.build(train_ts).dropna()
        X = frame[self._feature_cols].to_numpy(dtype=np.float64)
        y = frame["y"].to_numpy(dtype=np.float64)
        for q in quantiles:
            if q not in self._quantile_models:
                q_params = {**self.params, "objective": "quantile", "alpha": float(q)}
                q_params.pop("metric", None)
                dtrain = lgb.Dataset(X, label=y)
                self._quantile_models[q] = lgb.train(
                    params=q_params,
                    train_set=dtrain,
                    num_boost_round=self.num_boost_round,
                )
            qdict[q] = self._predict_with_model(horizon, self._quantile_models[q])
        return point, qdict

    def _predict_with_model(self, horizon: int, model: Any) -> np.ndarray:
        """Recursive predict against an arbitrary fitted booster.

        Used by :meth:`_predict_quantiles` to reuse the recursion loop for
        each quantile-specific model.
        """
        preds = np.empty(horizon, dtype=np.float64)
        working_values = self._history.copy()
        working_index = self._history_index
        assert working_index is not None
        for h in range(horizon):
            if isinstance(working_index, pd.DatetimeIndex):
                assert self._freq is not None
                next_ts = working_index[-1] + pd.tseries.frequencies.to_offset(
                    self._freq
                )
                new_index = working_index.append(pd.DatetimeIndex([next_ts]))
            else:
                new_index = pd.RangeIndex(start=0, stop=len(working_values) + 1)
            extended_values = np.concatenate([working_values, [np.nan]])
            synthetic_ts = TimeSeries(
                values=np.where(np.isnan(extended_values), 0.0, extended_values),
                index=new_index,
                freq=self._freq,
                name="_predict",
            )
            frame = self.features.build(synthetic_ts)
            X_last = frame[self._feature_cols].iloc[[-1]].to_numpy(dtype=np.float64)
            yhat = float(model.predict(X_last)[0])
            preds[h] = yhat
            working_values = np.concatenate([working_values, [yhat]])
            working_index = new_index
        return preds


class DirectLightGBMForecaster(LightGBMForecaster):
    """Direct-strategy variant: one model per horizon step.

    Trades H models for H trains — pays off on long horizons where error
    compounding in the recursive strategy becomes the dominant loss.
    """

    name = "lightgbm_direct"
    supports_quantiles = False

    def __init__(
        self,
        horizon_train: int = 14,
        features: LagFeatureBuilder | None = None,
        params: dict[str, Any] | None = None,
        num_boost_round: int = 200,
    ) -> None:
        super().__init__(
            features=features, params=params, num_boost_round=num_boost_round
        )
        if horizon_train < 1:
            raise ValueError("horizon_train must be >= 1")
        self.horizon_train = horizon_train
        self._direct_models: list[Any] = []

    def _fit(self, series: TimeSeries) -> None:
        lgb = _import_lightgbm()
        frame = self.features.build(series)
        self._feature_cols = self.features.feature_columns(series)
        base = frame.dropna().copy()
        if len(base) < 10 + self.horizon_train:
            raise ValueError(
                f"lightgbm_direct: need >= {10 + self.horizon_train} usable rows, "
                f"got {len(base)}"
            )
        # Build per-step targets via shift(-h).
        self._direct_models = []
        for h in range(1, self.horizon_train + 1):
            target = base["y"].shift(-h + 1)  # h=1 → same row; h=2 → next row
            if h > 1:
                # Align: drop last (h-1) rows where the shifted target is NaN.
                target = target.iloc[: -(h - 1)] if h > 1 else target
                feat = base.iloc[: -(h - 1)][self._feature_cols]
            else:
                feat = base[self._feature_cols]
            X = feat.to_numpy(dtype=np.float64)
            y = target.to_numpy(dtype=np.float64)
            mask = np.isfinite(y)
            dtrain = lgb.Dataset(X[mask], label=y[mask])
            booster = lgb.train(
                params=self.params,
                train_set=dtrain,
                num_boost_round=self.num_boost_round,
            )
            self._direct_models.append(booster)
        self._history = np.asarray(series.values, dtype=np.float64).copy()
        self._history_index = series.index
        self._freq = series.freq

    def _predict(self, horizon: int) -> np.ndarray:
        if horizon > self.horizon_train:
            raise ValueError(
                f"lightgbm_direct was trained for horizon_train={self.horizon_train}; "
                f"got request for horizon={horizon}"
            )
        # Build features once for the final known row.
        assert self._history_index is not None
        train_ts = TimeSeries(
            values=self._history,
            index=self._history_index,
            freq=self._freq,
            name="_predict",
        )
        frame = self.features.build(train_ts)
        X_last = frame[self._feature_cols].iloc[[-1]].to_numpy(dtype=np.float64)
        preds = np.empty(horizon, dtype=np.float64)
        for h in range(horizon):
            preds[h] = float(self._direct_models[h].predict(X_last)[0])
        return preds
