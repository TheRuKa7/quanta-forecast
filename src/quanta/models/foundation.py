"""Zero-shot foundation-model forecasters — lazy-imported.

Two flavours, both loaded from their upstream packages:

* :class:`ChronosForecaster` — Amazon Chronos (T5-based, 2024). Zero-shot
  probabilistic. Ships in a ``t5-tiny`` variant we default to so CPU-only
  installs stay useful.
* :class:`TimesFMForecaster` — Google TimesFM (decoder-only, 2024).
  Zero-shot point + quantile forecasts.

Both require torch + the package-specific install. ``_fit`` is a no-op
except for storing the series — these models don't train at fit time.
``predict`` is where the forward pass happens.

Design notes:

* We cache the model instance as a class-level singleton keyed by
  checkpoint name, so repeated instantiations don't re-download weights.
* HuggingFace cache is used by default (``~/.cache/huggingface``), so
  after the first download subsequent calls are offline-capable.
* All imports are deferred into ``_fit`` — the foundation tier weighs
  hundreds of megabytes and users shouldn't pay for it unless they ask.
"""
from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

from quanta.base import BaseForecaster, TimeSeries

__all__ = ["ChronosForecaster", "TimesFMForecaster"]


class ChronosForecaster(BaseForecaster):
    """Amazon Chronos zero-shot forecaster via ``chronos-forecasting``.

    Parameters
    ----------
    model_name:
        HuggingFace model ID. Defaults to ``amazon/chronos-t5-tiny``
        (~8M params, CPU-friendly). Swap for ``-small``, ``-base``, or
        ``-large`` for better accuracy at more compute.
    device:
        Torch device string. ``"cpu"``, ``"cuda"``, ``"mps"``.
    num_samples:
        Samples to draw per step for probabilistic output.
    """

    name = "chronos"
    supports_quantiles = True

    #: Class-level cache keyed by model_name — avoids re-downloading weights
    #: when users instantiate the same model multiple times in one process.
    _pipeline_cache: ClassVar[dict[str, Any]] = {}

    def __init__(
        self,
        model_name: str = "amazon/chronos-t5-tiny",
        device: str = "cpu",
        num_samples: int = 100,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.device = device
        self.num_samples = num_samples
        self._context: np.ndarray = np.empty(0)

    def _load_pipeline(self) -> Any:  # pragma: no cover - requires network
        if self.model_name in self._pipeline_cache:
            return self._pipeline_cache[self.model_name]
        try:
            import torch
            from chronos import ChronosPipeline
        except ImportError as e:
            raise ImportError(
                "ChronosForecaster requires chronos-forecasting + torch. "
                "Install the `foundation` extra: "
                "`pip install 'quanta-forecast[foundation]'`."
            ) from e
        pipeline = ChronosPipeline.from_pretrained(
            self.model_name,
            device_map=self.device,
            torch_dtype=torch.float32,
        )
        self._pipeline_cache[self.model_name] = pipeline
        return pipeline

    def _fit(self, series: TimeSeries) -> None:  # pragma: no cover - requires network
        # Chronos is zero-shot — no training, just snapshot the context.
        self._context = np.asarray(series.values, dtype=np.float32)

    def _predict(self, horizon: int) -> np.ndarray:  # pragma: no cover
        import torch

        pipeline = self._load_pipeline()
        ctx = torch.tensor(self._context, dtype=torch.float32)
        samples = pipeline.predict(
            context=ctx, prediction_length=horizon, num_samples=self.num_samples
        )
        # samples shape: (batch=1, num_samples, horizon)
        arr = samples.numpy() if hasattr(samples, "numpy") else np.asarray(samples)
        return np.median(arr[0], axis=0).astype(np.float64)

    def _predict_quantiles(  # pragma: no cover
        self, horizon: int, quantiles: tuple[float, ...]
    ) -> tuple[np.ndarray, dict[float, np.ndarray]]:
        import torch

        pipeline = self._load_pipeline()
        ctx = torch.tensor(self._context, dtype=torch.float32)
        samples = pipeline.predict(
            context=ctx, prediction_length=horizon, num_samples=self.num_samples
        )
        arr = samples.numpy() if hasattr(samples, "numpy") else np.asarray(samples)
        arr0 = arr[0]  # (num_samples, horizon)
        point = np.median(arr0, axis=0).astype(np.float64)
        qdict = {
            q: np.quantile(arr0, q, axis=0).astype(np.float64) for q in quantiles
        }
        return point, qdict


class TimesFMForecaster(BaseForecaster):
    """Google TimesFM zero-shot forecaster.

    Parameters
    ----------
    model_name:
        HuggingFace repo. Defaults to ``google/timesfm-1.0-200m-pytorch``.
    context_len:
        Context length for TimesFM. Default 512 matches the public weights.
    horizon_len:
        Max horizon per forward pass. Longer horizons chunk internally.
    """

    name = "timesfm"
    supports_quantiles = True

    _model_cache: ClassVar[dict[str, Any]] = {}

    def __init__(
        self,
        model_name: str = "google/timesfm-1.0-200m-pytorch",
        context_len: int = 512,
        horizon_len: int = 128,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.context_len = context_len
        self.horizon_len = horizon_len
        self._context: np.ndarray = np.empty(0)
        self._freq: int = 0  # 0 = high frequency (daily/hourly); see timesfm docs

    def _load_model(self) -> Any:  # pragma: no cover - requires network
        if self.model_name in self._model_cache:
            return self._model_cache[self.model_name]
        try:
            import timesfm
        except ImportError as e:
            raise ImportError(
                "TimesFMForecaster requires timesfm. Install the `foundation` "
                "extra: `pip install 'quanta-forecast[foundation]'` (timesfm "
                "may require a separate wheel — see upstream install notes)."
            ) from e
        tfm = timesfm.TimesFm(
            hparams=timesfm.TimesFmHparams(
                backend="cpu",
                per_core_batch_size=32,
                horizon_len=self.horizon_len,
                context_len=self.context_len,
            ),
            checkpoint=timesfm.TimesFmCheckpoint(huggingface_repo_id=self.model_name),
        )
        self._model_cache[self.model_name] = tfm
        return tfm

    def _fit(self, series: TimeSeries) -> None:  # pragma: no cover - requires network
        self._context = np.asarray(series.values, dtype=np.float32)

    def _predict(self, horizon: int) -> np.ndarray:  # pragma: no cover
        tfm = self._load_model()
        point, _ = tfm.forecast([self._context], freq=[self._freq])
        arr = np.asarray(point[0], dtype=np.float64)
        return arr[:horizon]

    def _predict_quantiles(  # pragma: no cover
        self, horizon: int, quantiles: tuple[float, ...]
    ) -> tuple[np.ndarray, dict[float, np.ndarray]]:
        tfm = self._load_model()
        point, qvals = tfm.forecast([self._context], freq=[self._freq])
        mean = np.asarray(point[0], dtype=np.float64)[:horizon]
        # timesfm emits a fixed quantile set (typically 10 levels); interpolate.
        q_levels = np.linspace(0.1, 0.9, qvals[0].shape[-1])
        qmatrix = np.asarray(qvals[0], dtype=np.float64)[:horizon]
        qdict: dict[float, np.ndarray] = {}
        for q in quantiles:
            idx = int(np.argmin(np.abs(q_levels - q)))
            qdict[q] = qmatrix[:, idx]
        return mean, qdict
