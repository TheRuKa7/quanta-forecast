"""quanta-forecast — unified multi-model time-series forecasting.

Public API — everything imported here is stable across 0.x releases:

>>> from quanta import create, TimeSeries, make_synthetic_seasonal
>>> ts = make_synthetic_seasonal(n=120, period=12)
>>> m = create("holt_winters", season=12)
>>> m.fit(ts)
>>> fc = m.predict(horizon=12)
>>> fc.point.shape
(12,)

Heavy-dependency backends (``lightgbm``, ``tft``, ``chronos``, ``timesfm``)
are lazily imported by the registry; importing ``quanta`` itself is cheap.
"""
from __future__ import annotations

from quanta.base import (
    DEFAULT_QUANTILES,
    BaseForecaster,
    ForecastOutput,
    Forecaster,
    TimeSeries,
)
from quanta.data.loaders import (
    load_airline_passengers,
    load_csv,
    make_synthetic_seasonal,
    make_synthetic_trend,
)
from quanta.registry import create, is_available, list_backends, register

__version__ = "0.1.0"

__all__ = [
    "BaseForecaster",
    "DEFAULT_QUANTILES",
    "ForecastOutput",
    "Forecaster",
    "TimeSeries",
    "__version__",
    "create",
    "is_available",
    "list_backends",
    "load_airline_passengers",
    "load_csv",
    "make_synthetic_seasonal",
    "make_synthetic_trend",
    "register",
]
