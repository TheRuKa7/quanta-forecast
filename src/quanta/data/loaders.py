"""Loaders + synthetic generators.

The synthetic generators (``make_synthetic_trend``, ``make_synthetic_seasonal``,
``load_airline_passengers``) exist so tests and examples can run without any
network access or external files. The real-data loaders (``load_csv``) keep
the network path optional — they never run at import time.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from quanta.base import TimeSeries

__all__ = [
    "load_csv",
    "load_airline_passengers",
    "make_synthetic_trend",
    "make_synthetic_seasonal",
]


def load_csv(
    path: str | Path,
    *,
    date_col: str = "ds",
    value_col: str = "y",
    freq: str | None = None,
    name: str = "y",
) -> TimeSeries:
    """Load a CSV with ``date_col`` and ``value_col``.

    The column names default to Prophet's ``ds``/``y`` convention because
    it's what most public forecasting datasets use. ``freq`` is inferred
    from the parsed index when not provided.
    """
    df = pd.read_csv(path)
    if date_col not in df.columns or value_col not in df.columns:
        raise KeyError(
            f"CSV at {path} missing required columns "
            f"{date_col!r} and/or {value_col!r}; found {list(df.columns)}"
        )
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).set_index(date_col)
    if freq is None:
        freq = pd.infer_freq(df.index)
    return TimeSeries(
        values=df[value_col].to_numpy(dtype=np.float64),
        index=pd.DatetimeIndex(df.index),
        freq=freq,
        name=name,
    )


def load_airline_passengers() -> TimeSeries:
    """The classic Box-Jenkins monthly airline series (1949–1960).

    Numbers are the published ones and ship in-repo rather than being
    downloaded — 144 values is cheap to keep as a literal, and it removes
    a network dependency from tests and examples.
    """
    values = [
        112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118,
        115, 126, 141, 135, 125, 149, 170, 170, 158, 133, 114, 140,
        145, 150, 178, 163, 172, 178, 199, 199, 184, 162, 146, 166,
        171, 180, 193, 181, 183, 218, 230, 242, 209, 191, 172, 194,
        196, 196, 236, 235, 229, 243, 264, 272, 237, 211, 180, 201,
        204, 188, 235, 227, 234, 264, 302, 293, 259, 229, 203, 229,
        242, 233, 267, 269, 270, 315, 364, 347, 312, 274, 237, 278,
        284, 277, 317, 313, 318, 374, 413, 405, 355, 306, 271, 306,
        315, 301, 356, 348, 355, 422, 465, 467, 404, 347, 305, 336,
        340, 318, 362, 348, 363, 435, 491, 505, 404, 359, 310, 337,
        360, 342, 406, 396, 420, 472, 548, 559, 463, 407, 362, 405,
        417, 391, 419, 461, 472, 535, 622, 606, 508, 461, 390, 432,
    ]
    idx = pd.date_range(start="1949-01-01", periods=len(values), freq="MS")
    return TimeSeries(
        values=np.asarray(values, dtype=np.float64),
        index=idx,
        freq="MS",
        name="airline_passengers",
    )


def make_synthetic_trend(
    n: int = 200,
    *,
    slope: float = 0.5,
    intercept: float = 10.0,
    noise: float = 1.0,
    freq: str = "D",
    start: str = "2024-01-01",
    seed: int = 0,
) -> TimeSeries:
    """Linear trend + Gaussian noise. Useful for naive/drift sanity checks."""
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=np.float64)
    values = intercept + slope * t + rng.normal(0.0, noise, size=n)
    return TimeSeries(
        values=values,
        index=pd.date_range(start=start, periods=n, freq=freq),
        freq=freq,
        name="trend",
    )


def make_synthetic_seasonal(
    n: int = 365,
    *,
    period: int = 7,
    amplitude: float = 5.0,
    trend_slope: float = 0.1,
    noise: float = 0.5,
    freq: str = "D",
    start: str = "2024-01-01",
    seed: int = 0,
) -> TimeSeries:
    """Sinusoidal seasonal component + linear trend + noise.

    Default produces a weekly-seasonal daily series — the canonical toy for
    seasonal-naive and ARIMA smoke tests.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=np.float64)
    seasonal = amplitude * np.sin(2 * np.pi * t / period)
    trend = trend_slope * t
    noise_arr = rng.normal(0.0, noise, size=n)
    values = 50.0 + trend + seasonal + noise_arr
    return TimeSeries(
        values=values,
        index=pd.date_range(start=start, periods=n, freq=freq),
        freq=freq,
        name="seasonal",
    )
