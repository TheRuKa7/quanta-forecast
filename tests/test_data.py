"""Data loaders + synthetic generators."""
from __future__ import annotations

import numpy as np
import pandas as pd

from quanta.data.loaders import (
    load_airline_passengers,
    load_csv,
    make_synthetic_seasonal,
    make_synthetic_trend,
)


def test_airline_passengers_has_canonical_shape() -> None:
    ts = load_airline_passengers()
    assert len(ts) == 144  # 12 years x 12 months
    assert ts.freq == "MS"
    # First month is the published value.
    assert int(ts.values[0]) == 112
    assert int(ts.values[-1]) == 432


def test_synthetic_trend_is_monotone_on_average() -> None:
    ts = make_synthetic_trend(n=200, slope=1.0, noise=0.01, seed=1)
    # Near-noiseless: last must exceed first by ~slope*(n-1).
    expected = 1.0 * (len(ts) - 1)
    assert abs((ts.values[-1] - ts.values[0]) - expected) < 2.0


def test_synthetic_seasonal_has_period() -> None:
    # With no noise and no trend, same-day-of-week values should coincide.
    tight = make_synthetic_seasonal(
        n=140, period=7, amplitude=5.0, trend_slope=0.0, noise=0.0, seed=0
    )
    # v[t] and v[t+7] should be equal modulo the tiny offset from trend=0.
    np.testing.assert_allclose(tight.values[7:14], tight.values[0:7], atol=1e-10)


def test_load_csv_roundtrip(tmp_path) -> None:
    df = pd.DataFrame(
        {
            "ds": pd.date_range("2024-01-01", periods=30, freq="D"),
            "y": np.arange(30, dtype=float),
        }
    )
    path = tmp_path / "series.csv"
    df.to_csv(path, index=False)
    ts = load_csv(path)
    assert len(ts) == 30
    assert ts.freq == "D"
    assert ts.values[-1] == 29.0


def test_load_csv_missing_columns(tmp_path) -> None:
    import pytest

    df = pd.DataFrame({"date": [1, 2], "value": [1, 2]})
    path = tmp_path / "bad.csv"
    df.to_csv(path, index=False)
    with pytest.raises(KeyError, match="missing required"):
        load_csv(path)
