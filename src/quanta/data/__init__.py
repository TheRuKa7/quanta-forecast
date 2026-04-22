"""Data containers + loaders for quanta-forecast."""
from __future__ import annotations

from quanta.base import TimeSeries
from quanta.data.loaders import (
    load_airline_passengers,
    load_csv,
    make_synthetic_seasonal,
    make_synthetic_trend,
)

__all__ = [
    "TimeSeries",
    "load_airline_passengers",
    "load_csv",
    "make_synthetic_seasonal",
    "make_synthetic_trend",
]
