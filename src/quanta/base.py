"""Unified Forecaster protocol — the organizing abstraction for all tiers."""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from pydantic import BaseModel, ConfigDict


class ForecastOutput(BaseModel):
    """Point + probabilistic forecast output shared across all backends."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    mean: np.ndarray
    quantiles: dict[float, np.ndarray] | None = None  # e.g. {0.1: arr, 0.5: arr, 0.9: arr}
    model_name: str
    horizon: int


@runtime_checkable
class Forecaster(Protocol):
    """Every backend implements this contract — classical, ML, deep, foundation."""

    name: str

    def fit(self, series: np.ndarray) -> "Forecaster":
        """Fit on a univariate series. Return self for chaining."""
        ...

    def predict(self, horizon: int, num_samples: int = 100) -> ForecastOutput:
        """Produce a point + probabilistic forecast `horizon` steps ahead."""
        ...
