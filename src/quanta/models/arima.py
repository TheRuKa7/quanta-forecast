"""ARIMA backend via ``statsmodels``.

``statsmodels.tsa.arima.model.ARIMA`` is the most broadly compatible ARIMA
implementation in the Python ecosystem. It gives us analytical prediction
intervals out of the box, which we surface via :meth:`predict_quantiles`.

We do *not* implement Hyndman-Khandakar auto-ARIMA in-house — if you want
that, use the Nixtla ``statsforecast`` backend (lazy-imported from
``quanta.models.classical`` in the ``[classical]`` extra). The fixed ``order``
API here is appropriate for users who know the ARIMA(p,d,q) / SARIMA they
want and for CI tests that need reproducibility.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

from quanta.base import BaseForecaster, TimeSeries

__all__ = ["ARIMAForecaster"]


class ARIMAForecaster(BaseForecaster):
    """Fixed-order (S)ARIMA with analytical prediction intervals.

    Parameters
    ----------
    order:
        ``(p, d, q)`` for the non-seasonal component.
    seasonal_order:
        ``(P, D, Q, s)`` for the seasonal component; ``s=0`` disables it.
    trend:
        Passed straight through to statsmodels — ``'c'`` (constant),
        ``'t'`` (linear), ``'ct'``, or ``None``.
    """

    name = "arima"
    supports_quantiles = True

    def __init__(
        self,
        order: tuple[int, int, int] = (1, 1, 1),
        seasonal_order: tuple[int, int, int, int] = (0, 0, 0, 0),
        trend: str | None = None,
    ) -> None:
        super().__init__()
        self.order = order
        self.seasonal_order = seasonal_order
        self.trend = trend
        self._res = None  # statsmodels results wrapper
        self._resid_std: float = 0.0

    def _fit(self, series: TimeSeries) -> None:
        try:
            from statsmodels.tsa.arima.model import ARIMA as _SMArima
        except ImportError as e:  # pragma: no cover - dev-only path
            raise ImportError(
                "ARIMAForecaster requires statsmodels. Install the "
                "`classical` extra: `pip install 'quanta-forecast[classical]'`."
            ) from e
        model = _SMArima(
            series.values,
            order=self.order,
            seasonal_order=self.seasonal_order,
            trend=self.trend,
        )
        # ``disp=False`` silences the convergence printouts that statsmodels
        # emits by default — fine for library use, noisy in CI.
        self._res = model.fit()
        resid = np.asarray(self._res.resid, dtype=np.float64)
        # Guard against all-NaN residuals at the series head (common for d>0).
        finite = resid[np.isfinite(resid)]
        self._resid_std = float(np.std(finite)) if len(finite) > 1 else 0.0

    def _predict(self, horizon: int) -> np.ndarray:
        assert self._res is not None
        point = np.asarray(self._res.forecast(steps=horizon), dtype=np.float64)
        return point

    def _predict_quantiles(
        self, horizon: int, quantiles: tuple[float, ...]
    ) -> tuple[np.ndarray, dict[float, np.ndarray]]:
        """Use statsmodels' ``get_forecast`` for analytical intervals.

        ``get_forecast`` returns a ``PredictionResults`` whose ``conf_int``
        takes an ``alpha``. Building quantiles from a sweep of alpha values
        is straightforward — we just pick each quantile's two-sided bound.
        """
        assert self._res is not None
        pred = self._res.get_forecast(steps=horizon)
        mean = np.asarray(pred.predicted_mean, dtype=np.float64)
        se = np.asarray(pred.se_mean, dtype=np.float64)
        qdict: dict[float, np.ndarray] = {}
        for q in quantiles:
            z = norm.ppf(q)
            qdict[q] = mean + z * se
        return mean, qdict

    # statsmodels Results objects are large and don't always pickle cleanly
    # across versions. Override save/load to stash only what we need.
    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        # Drop the heavy results wrapper; persist the fitted params instead.
        if self._res is not None:
            state["_res_params"] = np.asarray(self._res.params, dtype=np.float64)
            state["_res_fittedvalues"] = None  # reconstructable from params
        state["_res"] = None
        return state
