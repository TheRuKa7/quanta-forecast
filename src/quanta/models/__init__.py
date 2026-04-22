"""Forecasting backends.

Modules in this package all subclass :class:`quanta.base.BaseForecaster`.
The registry in :mod:`quanta.registry` wires them up by name so the CLI and
the HTTP surface can dispatch without importing every backend eagerly (the
deep / foundation backends pull heavy deps that may not be installed).
"""
from __future__ import annotations

# Always-available backends — safe to import at package load.
from quanta.models.naive import (
    DriftForecaster,
    MeanForecaster,
    NaiveForecaster,
    SeasonalNaiveForecaster,
)
from quanta.models.smoothing import (
    HoltForecaster,
    HoltWintersForecaster,
    SimpleExpSmoothingForecaster,
)

__all__ = [
    "DriftForecaster",
    "HoltForecaster",
    "HoltWintersForecaster",
    "MeanForecaster",
    "NaiveForecaster",
    "SeasonalNaiveForecaster",
    "SimpleExpSmoothingForecaster",
]
