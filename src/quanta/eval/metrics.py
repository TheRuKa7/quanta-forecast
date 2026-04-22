"""Point + probabilistic forecast metrics.

All functions accept numpy arrays and return a scalar float. They never
mutate their inputs. The distinction between MAPE (symmetric? no; blows up
near zero) and sMAPE (symmetric; bounded) matters for competition-grade
comparisons — we provide both so callers can pick what their domain expects.

MASE (Mean Absolute Scaled Error) is the M-competition favourite because
it's scale-free and robust to zeros. It requires the training series to
compute the seasonal naive denominator — that's why its signature differs
from the others.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "coverage",
    "crps_ensemble",
    "mae",
    "mape",
    "mase",
    "pinball_loss",
    "rmse",
    "smape",
]

_EPS = 1e-12


def _check_shapes(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: y_true={y_true.shape} y_pred={y_pred.shape}")
    if y_true.ndim != 1:
        raise ValueError("metrics expect 1-D arrays")


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error."""
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    _check_shapes(yt, yp)
    return float(np.mean(np.abs(yt - yp)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error."""
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    _check_shapes(yt, yp)
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute percentage error. Undefined near zero — use sMAPE if
    your series can hit zero."""
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    _check_shapes(yt, yp)
    denom = np.clip(np.abs(yt), _EPS, None)
    return float(np.mean(np.abs(yt - yp) / denom) * 100.0)


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Symmetric MAPE (percent, range [0, 200])."""
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    _check_shapes(yt, yp)
    denom = np.clip((np.abs(yt) + np.abs(yp)), _EPS, None)
    return float(np.mean(2.0 * np.abs(yt - yp) / denom) * 100.0)


def mase(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: np.ndarray,
    *,
    season: int = 1,
) -> float:
    """Mean Absolute Scaled Error. ``season`` should equal the seasonal
    period used by the naive benchmark (1 for plain naive, m for seasonal
    naive with period m)."""
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    ytr = np.asarray(y_train, dtype=np.float64)
    _check_shapes(yt, yp)
    if ytr.ndim != 1 or len(ytr) <= season:
        raise ValueError(
            f"need at least {season + 1} training points for MASE at season={season}"
        )
    naive_errors = np.abs(ytr[season:] - ytr[:-season])
    scale = float(np.mean(naive_errors))
    if scale < _EPS:
        raise ValueError(
            "MASE denominator is ~0; training series has no variation at this "
            "seasonal lag. Use a different season or a different metric."
        )
    return float(np.mean(np.abs(yt - yp)) / scale)


def pinball_loss(y_true: np.ndarray, y_pred_q: np.ndarray, q: float) -> float:
    """Pinball (quantile) loss for a single quantile level ``q`` in (0, 1)."""
    if not (0.0 < q < 1.0):
        raise ValueError(f"q must be in (0, 1), got {q}")
    yt = np.asarray(y_true, dtype=np.float64)
    yq = np.asarray(y_pred_q, dtype=np.float64)
    _check_shapes(yt, yq)
    diff = yt - yq
    return float(np.mean(np.maximum(q * diff, (q - 1.0) * diff)))


def coverage(
    y_true: np.ndarray, y_lower: np.ndarray, y_upper: np.ndarray
) -> float:
    """Empirical coverage — fraction of truth values inside [lower, upper].

    For a well-calibrated 80% central interval this should be ~0.8.
    """
    yt = np.asarray(y_true, dtype=np.float64)
    yl = np.asarray(y_lower, dtype=np.float64)
    yu = np.asarray(y_upper, dtype=np.float64)
    _check_shapes(yt, yl)
    _check_shapes(yt, yu)
    if np.any(yu < yl):
        raise ValueError("y_upper must be >= y_lower elementwise")
    return float(np.mean((yt >= yl) & (yt <= yu)))


def crps_ensemble(y_true: np.ndarray, samples: np.ndarray) -> float:
    """CRPS estimated from an ensemble/sample matrix of shape (horizon, n_samples).

    Uses the sorted-sample estimator — O(n_samples log n_samples) per step.
    """
    yt = np.asarray(y_true, dtype=np.float64)
    s = np.asarray(samples, dtype=np.float64)
    if s.ndim != 2 or s.shape[0] != yt.shape[0]:
        raise ValueError(
            f"samples must have shape (len(y_true), n_samples), got {s.shape}"
        )
    if yt.ndim != 1:
        raise ValueError("y_true must be 1-D")
    n_samples = s.shape[1]
    s_sorted = np.sort(s, axis=1)
    # Term 1: E|X - y|
    term1 = np.mean(np.abs(s_sorted - yt[:, None]), axis=1)
    # Term 2: 0.5 * E|X - X'|  via sorted trick.
    idx = np.arange(n_samples)
    # sum_{i<j} (s[j] - s[i]) = sum_i s[i] * (2 i - n + 1)  for sorted ascending
    weights = (2 * idx - n_samples + 1).astype(np.float64)
    term2 = np.sum(weights * s_sorted, axis=1) / (n_samples * n_samples)
    crps = term1 - term2
    return float(np.mean(crps))
