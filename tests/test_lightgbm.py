"""LightGBM backends — recursive + direct.

Skipped when lightgbm isn't installed.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("lightgbm")

from quanta.data.loaders import make_synthetic_seasonal
from quanta.features.lag import LagFeatureBuilder
from quanta.models.ml import DirectLightGBMForecaster, LightGBMForecaster


def test_lightgbm_fits_and_predicts_shape() -> None:
    ts = make_synthetic_seasonal(n=200, period=7, seed=0)
    m = LightGBMForecaster(num_boost_round=50)
    m.fit(ts)
    fc = m.predict(14)
    assert fc.point.shape == (14,)
    assert fc.model_name == "lightgbm"
    assert np.all(np.isfinite(fc.point))


def test_lightgbm_with_custom_features() -> None:
    ts = make_synthetic_seasonal(n=200, period=7, seed=0)
    features = LagFeatureBuilder(
        lags=(1, 2, 7), rolling_windows=(7,), calendar=True
    )
    m = LightGBMForecaster(features=features, num_boost_round=50)
    m.fit(ts)
    fc = m.predict(7)
    assert fc.point.shape == (7,)


def test_lightgbm_quantiles_bracket_point() -> None:
    ts = make_synthetic_seasonal(n=250, period=7, seed=1)
    m = LightGBMForecaster(num_boost_round=80).fit(ts)
    fcq = m.predict_quantiles(14, quantiles=(0.1, 0.5, 0.9))
    assert fcq.quantiles is not None
    lo = fcq.quantiles[0.1]
    hi = fcq.quantiles[0.9]
    # 10th <= 90th at every step (allow tiny numerical slack from separate
    # boosters).
    assert np.all(lo <= hi + 1e-6)


def test_lightgbm_direct_respects_horizon_cap() -> None:
    ts = make_synthetic_seasonal(n=200, period=7, seed=0)
    m = DirectLightGBMForecaster(horizon_train=7, num_boost_round=50).fit(ts)
    fc = m.predict(7)
    assert fc.point.shape == (7,)
    with pytest.raises(ValueError, match="horizon_train"):
        m.predict(14)


def test_lightgbm_rejects_short_series() -> None:
    """After lag-1..28 is cut, a 10-point series has no training rows."""
    from quanta.base import TimeSeries

    ts = TimeSeries.from_array(np.arange(10, dtype=float), start="2024-01-01", freq="D")
    m = LightGBMForecaster()
    with pytest.raises(ValueError, match="lightgbm"):
        m.fit(ts)
