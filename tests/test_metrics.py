"""Point + probabilistic metrics."""
from __future__ import annotations

import numpy as np
import pytest

from quanta.eval.metrics import (
    coverage,
    crps_ensemble,
    mae,
    mape,
    mase,
    pinball_loss,
    rmse,
    smape,
)


def test_mae_zero_for_perfect_forecast() -> None:
    y = np.array([1.0, 2.0, 3.0])
    assert mae(y, y) == 0.0


def test_rmse_is_sqrt_mse() -> None:
    y = np.array([0.0, 0.0])
    p = np.array([3.0, 4.0])
    # MSE = (9 + 16) / 2 = 12.5; sqrt = ~3.5355
    assert rmse(y, p) == pytest.approx(np.sqrt(12.5))


def test_mape_percent_units() -> None:
    y = np.array([100.0, 200.0])
    p = np.array([110.0, 180.0])  # 10% and 10% errors
    assert mape(y, p) == pytest.approx(10.0)


def test_smape_bounded_range() -> None:
    y = np.array([0.0, 100.0])
    p = np.array([100.0, 0.0])
    # Both errors are 200% — sMAPE returns 200.0.
    assert smape(y, p) == pytest.approx(200.0)


def test_mase_naive_benchmark() -> None:
    y_train = np.arange(1, 21, dtype=float)  # 1..20, naive-1 error = 1 per step
    y_true = np.array([21.0, 22.0])
    y_pred = np.array([20.0, 21.0])  # always off by 1 → MASE = 1.0
    assert mase(y_true, y_pred, y_train, season=1) == pytest.approx(1.0)


def test_mase_degenerate_train_raises() -> None:
    with pytest.raises(ValueError, match="denominator"):
        mase(np.array([1.0]), np.array([1.0]), np.array([5.0, 5.0, 5.0]), season=1)


def test_pinball_symmetric_at_median() -> None:
    y = np.array([1.0, 2.0, 3.0])
    p = np.array([2.0, 2.0, 2.0])
    # At q=0.5 pinball is MAE / 2.
    assert pinball_loss(y, p, 0.5) == pytest.approx(mae(y, p) / 2)


def test_pinball_reject_invalid_q() -> None:
    y = np.array([1.0])
    with pytest.raises(ValueError, match="q must"):
        pinball_loss(y, y, 0.0)


def test_coverage_all_inside() -> None:
    y = np.array([1.0, 2.0, 3.0])
    lo = np.array([0.0, 0.0, 0.0])
    hi = np.array([10.0, 10.0, 10.0])
    assert coverage(y, lo, hi) == 1.0


def test_coverage_half_inside() -> None:
    y = np.array([1.0, 2.0, 3.0, 4.0])
    lo = np.array([-10, -10, 5, 5], dtype=float)
    hi = np.array([10, 10, 10, 10], dtype=float)
    # y=1 in, y=2 in, y=3 below lo=5 → out, y=4 below lo=5 → out
    assert coverage(y, lo, hi) == 0.5


def test_crps_ensemble_zero_for_delta_at_truth() -> None:
    y = np.array([1.0, 2.0])
    samples = np.tile(y[:, None], (1, 10))  # samples == truth, delta mass
    assert crps_ensemble(y, samples) == pytest.approx(0.0, abs=1e-10)


def test_crps_ensemble_positive_for_biased() -> None:
    y = np.array([1.0])
    samples = np.array([[10.0] * 10])
    assert crps_ensemble(y, samples) > 0.0


def test_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        mae(np.array([1.0, 2.0]), np.array([1.0]))
