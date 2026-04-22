"""Shared pytest fixtures."""
from __future__ import annotations

import numpy as np
import pytest

from quanta.data.loaders import make_synthetic_seasonal, make_synthetic_trend


@pytest.fixture()
def seasonal_ts():
    """365-day weekly-seasonal series — canonical for SeasonalNaive + ARIMA."""
    return make_synthetic_seasonal(n=365, period=7, seed=42)


@pytest.fixture()
def trend_ts():
    """200-day linear trend — canonical for naive/drift comparisons."""
    return make_synthetic_trend(n=200, seed=42)


@pytest.fixture()
def short_ts():
    """24-point series for contract tests that don't care about signal."""
    rng = np.random.default_rng(0)
    values = rng.normal(50, 2, size=24)
    from quanta.base import TimeSeries

    return TimeSeries.from_array(values, start="2024-01-01", freq="D")
