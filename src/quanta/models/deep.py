"""Deep-learning forecaster backends — lazy-imported.

The Temporal Fusion Transformer (TFT) and friends live in ``darts`` /
``neuralforecast``, both of which pull in ``torch`` + ``pytorch-lightning``.
That's >1 GB of wheels we don't want to require for a simple naive forecast.
This module defers every import into ``_fit`` so the backend is only paid
for when actually used.

Install the ``deep`` extra first: ``pip install 'quanta-forecast[deep]'``.

Registered names:

* ``tft`` — Temporal Fusion Transformer via darts
* ``nbeats`` — N-BEATS via darts
* ``nhits`` — N-HiTS via darts

Each is thin — darts already wraps the PyTorch training loop; we just
normalize the contract.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from quanta.base import BaseForecaster, TimeSeries

__all__ = ["NBEATSForecaster", "NHiTSForecaster", "TFTForecaster"]


def _require_darts():
    """Import darts or raise a clear install hint.

    Deferred so users who never touch the deep models don't pay the
    import cost (darts drags in torch + lightning).
    """
    try:
        import darts  # noqa: F401
        from darts import TimeSeries as DartsTimeSeries
    except ImportError as e:  # pragma: no cover - dev-only path
        raise ImportError(
            "Deep backends (tft/nbeats/nhits) require darts. Install the "
            "`deep` extra: `pip install 'quanta-forecast[deep]'`."
        ) from e
    return DartsTimeSeries


def _to_darts_ts(series: TimeSeries):
    """Convert a quanta TimeSeries to a ``darts.TimeSeries``."""
    DartsTimeSeries = _require_darts()
    return DartsTimeSeries.from_times_and_values(series.index, series.values)


class _DartsBackedForecaster(BaseForecaster):
    """Shared plumbing for darts-wrapped deep models."""

    supports_quantiles = True

    def __init__(
        self,
        input_chunk_length: int = 24,
        output_chunk_length: int = 12,
        n_epochs: int = 20,
        random_state: int = 0,
        **extra: Any,
    ) -> None:
        super().__init__()
        self.input_chunk_length = input_chunk_length
        self.output_chunk_length = output_chunk_length
        self.n_epochs = n_epochs
        self.random_state = random_state
        self.extra = extra
        self._model: Any = None

    def _build_model(self) -> Any:  # pragma: no cover - requires darts
        raise NotImplementedError

    def _fit(self, series: TimeSeries) -> None:  # pragma: no cover - requires darts
        darts_ts = _to_darts_ts(series)
        self._model = self._build_model()
        self._model.fit(darts_ts)

    def _predict(self, horizon: int) -> np.ndarray:  # pragma: no cover
        assert self._model is not None
        forecast = self._model.predict(n=horizon)
        return np.asarray(forecast.values().flatten(), dtype=np.float64)

    def _predict_quantiles(  # pragma: no cover - requires darts
        self, horizon: int, quantiles: tuple[float, ...]
    ) -> tuple[np.ndarray, dict[float, np.ndarray]]:
        assert self._model is not None
        forecast = self._model.predict(n=horizon, num_samples=500)
        point = np.asarray(forecast.quantile(0.5).values().flatten(), dtype=np.float64)
        qdict = {
            q: np.asarray(forecast.quantile(q).values().flatten(), dtype=np.float64)
            for q in quantiles
        }
        return point, qdict


class TFTForecaster(_DartsBackedForecaster):
    """Temporal Fusion Transformer (Lim et al. 2021) via darts."""

    name = "tft"

    def _build_model(self) -> Any:  # pragma: no cover - requires darts
        from darts.models import TFTModel

        return TFTModel(
            input_chunk_length=self.input_chunk_length,
            output_chunk_length=self.output_chunk_length,
            n_epochs=self.n_epochs,
            random_state=self.random_state,
            add_relative_index=True,  # works without covariates
            **self.extra,
        )


class NBEATSForecaster(_DartsBackedForecaster):
    """N-BEATS (Oreshkin et al. 2019) via darts."""

    name = "nbeats"
    supports_quantiles = False  # darts NBEATSModel is deterministic by default

    def _build_model(self) -> Any:  # pragma: no cover - requires darts
        from darts.models import NBEATSModel

        return NBEATSModel(
            input_chunk_length=self.input_chunk_length,
            output_chunk_length=self.output_chunk_length,
            n_epochs=self.n_epochs,
            random_state=self.random_state,
            **self.extra,
        )

    def _predict_quantiles(  # pragma: no cover - requires darts
        self, horizon: int, quantiles: tuple[float, ...]
    ) -> tuple[np.ndarray, dict[float, np.ndarray]]:
        raise NotImplementedError("nbeats is deterministic; use tft for quantiles")


class NHiTSForecaster(_DartsBackedForecaster):
    """N-HiTS (Challu et al. 2022) via darts."""

    name = "nhits"
    supports_quantiles = False

    def _build_model(self) -> Any:  # pragma: no cover - requires darts
        from darts.models import NHiTSModel

        return NHiTSModel(
            input_chunk_length=self.input_chunk_length,
            output_chunk_length=self.output_chunk_length,
            n_epochs=self.n_epochs,
            random_state=self.random_state,
            **self.extra,
        )

    def _predict_quantiles(  # pragma: no cover - requires darts
        self, horizon: int, quantiles: tuple[float, ...]
    ) -> tuple[np.ndarray, dict[float, np.ndarray]]:
        raise NotImplementedError("nhits is deterministic; use tft for quantiles")
