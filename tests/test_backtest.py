"""Rolling-origin backtest harness."""
from __future__ import annotations

import pytest

from quanta.data.loaders import make_synthetic_trend
from quanta.eval.backtest import rolling_origin_backtest
from quanta.models.naive import NaiveForecaster


def test_backtest_produces_expected_fold_count() -> None:
    ts = make_synthetic_trend(n=100, seed=0)
    result = rolling_origin_backtest(
        NaiveForecaster,
        ts,
        horizon=5,
        min_train=50,
        step=5,
        max_folds=None,
    )
    # Folds where origin + horizon <= 100, starting at 50, step 5.
    # Origins: 50, 55, 60, 65, 70, 75, 80, 85, 90, 95 — last valid is 95.
    # (95 + 5 = 100, OK). So 10 folds.
    assert result.folds == 10
    assert result.horizon == 5
    assert result.model_name == "naive"
    assert set(result.per_fold.columns) >= {"mae", "rmse", "smape"}


def test_backtest_summary_returns_series() -> None:
    ts = make_synthetic_trend(n=80, seed=0)
    result = rolling_origin_backtest(
        NaiveForecaster, ts, horizon=3, min_train=40, max_folds=5
    )
    summary = result.summary()
    assert "mae" in summary.index
    assert summary["mae"] >= 0.0


def test_backtest_rejects_short_series() -> None:
    ts = make_synthetic_trend(n=10, seed=0)
    with pytest.raises(ValueError, match="too short"):
        rolling_origin_backtest(NaiveForecaster, ts, horizon=5, min_train=20)


def test_backtest_expanding_vs_sliding_same_length() -> None:
    ts = make_synthetic_trend(n=60, seed=0)
    exp = rolling_origin_backtest(
        NaiveForecaster, ts, horizon=3, min_train=30, step=3, expanding=True
    )
    slid = rolling_origin_backtest(
        NaiveForecaster, ts, horizon=3, min_train=30, step=3, expanding=False
    )
    # Both should produce the same number of folds — only the train window
    # strategy differs.
    assert exp.folds == slid.folds


def test_max_folds_caps_iteration() -> None:
    ts = make_synthetic_trend(n=200, seed=0)
    result = rolling_origin_backtest(
        NaiveForecaster, ts, horizon=5, min_train=100, step=1, max_folds=3
    )
    assert result.folds == 3
