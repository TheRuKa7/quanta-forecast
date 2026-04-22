"""Exponential smoothing family — SES, Holt, Holt-Winters."""
from __future__ import annotations

import numpy as np
import pytest

from quanta.data.loaders import make_synthetic_seasonal, make_synthetic_trend
from quanta.eval.metrics import mae
from quanta.models.naive import NaiveForecaster
from quanta.models.smoothing import (
    HoltForecaster,
    HoltWintersForecaster,
    SimpleExpSmoothingForecaster,
)


def test_ses_alpha_validation() -> None:
    with pytest.raises(ValueError, match="alpha"):
        SimpleExpSmoothingForecaster(alpha=0.0)


def test_ses_forecasts_flat() -> None:
    m = SimpleExpSmoothingForecaster(alpha=0.3).fit([1.0, 2.0, 3.0, 4.0, 5.0])
    fc = m.predict(3)
    # All values equal (flat extrapolation).
    assert fc.point[0] == pytest.approx(fc.point[1])
    assert fc.point[1] == pytest.approx(fc.point[2])


def test_holt_extrapolates_trend() -> None:
    ts = make_synthetic_trend(n=100, slope=1.0, noise=0.01, seed=0)
    m = HoltForecaster().fit(ts)
    fc = m.predict(5)
    # Trend series — later forecasts must exceed earlier ones.
    assert fc.point[-1] > fc.point[0]


def test_holt_beats_naive_on_trend() -> None:
    ts = make_synthetic_trend(n=100, slope=0.5, noise=0.1, seed=1)
    train, test = ts.values[:80], ts.values[80:]
    # Manual split — the fixture already wraps one, but simpler here.
    from quanta.base import TimeSeries

    train_ts = TimeSeries(
        values=train, index=ts.index[:80], freq=ts.freq, name=ts.name
    )
    holt = HoltForecaster().fit(train_ts).predict(20).point
    naive = NaiveForecaster().fit(train_ts).predict(20).point
    # Holt captures the trend; naive flat-lines. MAE for Holt should be lower.
    assert mae(test, holt) < mae(test, naive)


def test_holt_winters_requires_two_cycles() -> None:
    with pytest.raises(ValueError, match=">= 2 .* season"):
        HoltWintersForecaster(season=12).fit(list(range(20)))


def test_holt_winters_follows_seasonal_pattern() -> None:
    ts = make_synthetic_seasonal(
        n=140, period=7, amplitude=5.0, trend_slope=0.0, noise=0.01, seed=0
    )
    m = HoltWintersForecaster(season=7).fit(ts)
    fc = m.predict(14)
    # Two full cycles of forecast — the two halves should be ~equal.
    np.testing.assert_allclose(fc.point[:7], fc.point[7:14], atol=0.5)


def test_holt_winters_quantiles_bracket_point() -> None:
    ts = make_synthetic_seasonal(n=140, period=7, seed=0)
    m = HoltWintersForecaster(season=7).fit(ts)
    fcq = m.predict_quantiles(7, quantiles=(0.1, 0.5, 0.9))
    assert fcq.quantiles is not None
    # 10th percentile <= median <= 90th at every step (weakly).
    lo = fcq.quantiles[0.1]
    md = fcq.quantiles[0.5]
    hi = fcq.quantiles[0.9]
    assert np.all(lo <= md + 1e-9)
    assert np.all(md <= hi + 1e-9)
