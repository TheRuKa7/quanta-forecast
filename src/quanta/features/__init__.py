"""Feature engineering for the ML (LightGBM) backend.

Foundation and deep models consume raw series; classical models handle their
own feature construction internally. Only the ML tier needs explicit lag
engineering, so everything lives in a single module.
"""
from __future__ import annotations

from quanta.features.lag import LagFeatureBuilder, make_lag_frame

__all__ = ["LagFeatureBuilder", "make_lag_frame"]
