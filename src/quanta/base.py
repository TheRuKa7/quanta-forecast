"""Unified Forecaster contract — the organizing abstraction for all tiers.

Every backend (naive, ARIMA, LightGBM, TFT, Chronos, TimesFM) implements the
same four-method surface: ``fit``, ``predict``, ``predict_quantiles``, and the
class-level ``save`` / ``load`` pair. This is what lets the registry, the CLI,
and the HTTP layer treat them as interchangeable.

Why an ABC instead of a pure ``typing.Protocol``:

* ABCs let us register subclasses and enforce the contract at construction
  time via ``__init_subclass__``. That gives a clearer error for partial
  implementations than a Protocol ever will at runtime.
* ABCs carry default implementations (``save``, ``load``, ``predict_quantiles``)
  that most backends want to inherit unchanged.
* The matching ``Forecaster`` ``Protocol`` remains below for structural typing —
  use it for function signatures that don't care about inheritance.

The in-memory ``TimeSeries`` container and the ``ForecastOutput`` result are
kept as lightweight frozen dataclasses rather than pydantic models: every
backend touches them on the hot path, and pydantic validation adds overhead
that we don't need for trusted, in-process data.
"""
from __future__ import annotations

import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, Self, runtime_checkable

import numpy as np
import pandas as pd

# Default quantile set for probabilistic forecasts. Covers median + an 80%
# central interval — matches the conventional M-competition reporting.
DEFAULT_QUANTILES: tuple[float, ...] = (0.1, 0.5, 0.9)


@dataclass(frozen=True)
class TimeSeries:
    """Univariate time series with an explicit index.

    ``values`` and ``index`` must be the same length. ``freq`` is a pandas
    frequency string (``'D'``, ``'H'``, ``'MS'``, ...); callers that build a
    series from a numpy array without a meaningful index can use
    :meth:`from_array` and get a RangeIndex + ``freq=None``.
    """

    values: np.ndarray
    index: pd.DatetimeIndex | pd.RangeIndex
    freq: str | None = None
    name: str = "y"

    def __post_init__(self) -> None:
        arr = np.asarray(self.values, dtype=np.float64)
        if arr.ndim != 1:
            raise ValueError(f"TimeSeries.values must be 1-D, got shape {arr.shape}")
        if len(arr) != len(self.index):
            raise ValueError(
                f"values/index length mismatch: {len(arr)} vs {len(self.index)}"
            )
        # frozen=True means we can't just assign; use object.__setattr__.
        object.__setattr__(self, "values", arr)

    @classmethod
    def from_array(
        cls,
        values: np.ndarray | list[float],
        *,
        start: str | pd.Timestamp | None = None,
        freq: str | None = None,
        name: str = "y",
    ) -> TimeSeries:
        """Build a TimeSeries from a raw array; index is inferred."""
        arr = np.asarray(values, dtype=np.float64)
        idx: pd.DatetimeIndex | pd.RangeIndex
        if start is None or freq is None:
            idx = pd.RangeIndex(len(arr))
        else:
            idx = pd.date_range(start=start, periods=len(arr), freq=freq)
        return cls(values=arr, index=idx, freq=freq, name=name)

    @classmethod
    def from_series(cls, series: pd.Series, *, name: str | None = None) -> TimeSeries:
        """Build from a pandas Series (``series.index`` becomes ``index``)."""
        idx = series.index
        freq = getattr(idx, "freqstr", None) or pd.infer_freq(idx) if isinstance(
            idx, pd.DatetimeIndex
        ) else None
        return cls(
            values=series.to_numpy(dtype=np.float64),
            index=idx,
            freq=freq,
            name=name or str(series.name or "y"),
        )

    def to_series(self) -> pd.Series:
        return pd.Series(self.values, index=self.index, name=self.name)

    def __len__(self) -> int:
        return len(self.values)

    def future_index(self, horizon: int) -> pd.DatetimeIndex | pd.RangeIndex:
        """Index for the next ``horizon`` steps after the last observation.

        Handles both datetime and integer indices. For datetime indices we
        require ``self.freq`` to be set so the cadence is unambiguous.
        """
        if isinstance(self.index, pd.DatetimeIndex):
            if self.freq is None:
                raise ValueError(
                    "TimeSeries.freq must be set to project a DatetimeIndex forward"
                )
            start = self.index[-1] + pd.tseries.frequencies.to_offset(self.freq)
            return pd.date_range(start=start, periods=horizon, freq=self.freq)
        start_int = int(self.index[-1]) + 1
        return pd.RangeIndex(start=start_int, stop=start_int + horizon)


@dataclass(frozen=True)
class ForecastOutput:
    """The result of :meth:`Forecaster.predict`.

    ``point`` is the mean/median forecast; ``quantiles`` is an optional
    dict of quantile level -> array for probabilistic forecasts. Backends
    that don't produce uncertainty leave ``quantiles`` at ``None``.
    """

    point: np.ndarray
    index: pd.DatetimeIndex | pd.RangeIndex
    model_name: str
    horizon: int
    quantiles: dict[float, np.ndarray] | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        arr = np.asarray(self.point, dtype=np.float64)
        if arr.shape != (self.horizon,):
            raise ValueError(
                f"point shape {arr.shape} doesn't match horizon {self.horizon}"
            )
        if len(self.index) != self.horizon:
            raise ValueError(
                f"index length {len(self.index)} doesn't match horizon {self.horizon}"
            )
        object.__setattr__(self, "point", arr)
        if self.quantiles is not None:
            for q, qarr in self.quantiles.items():
                if not (0.0 < q < 1.0):
                    raise ValueError(f"quantile level must be in (0, 1), got {q}")
                qa = np.asarray(qarr, dtype=np.float64)
                if qa.shape != (self.horizon,):
                    raise ValueError(
                        f"quantile {q} shape {qa.shape} doesn't match horizon {self.horizon}"
                    )

    def to_frame(self) -> pd.DataFrame:
        """Return a DataFrame with ``point`` and each quantile as a column."""
        data: dict[str, np.ndarray] = {"point": self.point}
        if self.quantiles:
            for q, arr in sorted(self.quantiles.items()):
                data[f"q{int(q * 100):02d}"] = arr
        return pd.DataFrame(data, index=self.index)


class BaseForecaster(ABC):
    """Abstract base for every backend.

    Subclasses override :meth:`_fit` and :meth:`_predict`. The public
    :meth:`fit` and :meth:`predict` handle input coercion, state tracking
    (``_is_fitted``), and building the ``ForecastOutput`` container.

    Probabilistic forecasting is opt-in per backend via the class-level
    ``supports_quantiles`` flag. Backends that set it to ``True`` must also
    override :meth:`_predict_quantiles`; the default raises
    ``NotImplementedError``.
    """

    #: Human-readable registry key (``"naive"``, ``"arima"``, ``"lightgbm"``...).
    name: str = "base"

    #: Whether :meth:`predict_quantiles` is a real implementation.
    supports_quantiles: bool = False

    def __init__(self) -> None:
        self._is_fitted: bool = False
        self._train: TimeSeries | None = None

    # --- lifecycle --------------------------------------------------------

    def fit(self, series: TimeSeries | np.ndarray | list[float]) -> Self:
        """Fit on a univariate series and return ``self``."""
        ts = _coerce(series)
        if len(ts) < 2:
            raise ValueError(f"{self.name}: need at least 2 observations to fit")
        self._train = ts
        self._fit(ts)
        self._is_fitted = True
        return self

    def predict(self, horizon: int) -> ForecastOutput:
        """Produce a point forecast for the next ``horizon`` steps."""
        self._require_fitted()
        if horizon < 1:
            raise ValueError("horizon must be >= 1")
        assert self._train is not None  # for type checker
        point = np.asarray(self._predict(horizon), dtype=np.float64)
        if point.shape != (horizon,):
            raise RuntimeError(
                f"{self.name}._predict returned shape {point.shape}, expected ({horizon},)"
            )
        return ForecastOutput(
            point=point,
            index=self._train.future_index(horizon),
            model_name=self.name,
            horizon=horizon,
        )

    def predict_quantiles(
        self, horizon: int, quantiles: tuple[float, ...] = DEFAULT_QUANTILES
    ) -> ForecastOutput:
        """Probabilistic forecast. Backends that can't produce uncertainty
        raise ``NotImplementedError`` — callers should check
        ``supports_quantiles`` first."""
        self._require_fitted()
        if horizon < 1:
            raise ValueError("horizon must be >= 1")
        if not self.supports_quantiles:
            raise NotImplementedError(
                f"{self.name} does not support quantile forecasts; "
                "check Forecaster.supports_quantiles before calling."
            )
        assert self._train is not None
        point, qdict = self._predict_quantiles(horizon, tuple(quantiles))
        return ForecastOutput(
            point=np.asarray(point, dtype=np.float64),
            index=self._train.future_index(horizon),
            model_name=self.name,
            horizon=horizon,
            quantiles=qdict,
        )

    # --- persistence ------------------------------------------------------

    def save(self, path: str | Path) -> Path:
        """Pickle the fitted model to disk.

        Subclasses with un-picklable state (live torch modules, open sockets)
        should override this and `load`.
        """
        self._require_fitted()
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("wb") as f:
            pickle.dump(self, f)
        return p

    @classmethod
    def load(cls, path: str | Path) -> BaseForecaster:
        with Path(path).open("rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, BaseForecaster):
            raise TypeError(f"loaded object is {type(obj)!r}, not a BaseForecaster")
        return obj

    # --- hooks ------------------------------------------------------------

    @abstractmethod
    def _fit(self, series: TimeSeries) -> None:
        """Backend-specific fit. Called with a validated ``TimeSeries``."""

    @abstractmethod
    def _predict(self, horizon: int) -> np.ndarray:
        """Backend-specific point forecast. Return shape ``(horizon,)``."""

    def _predict_quantiles(
        self, horizon: int, quantiles: tuple[float, ...]
    ) -> tuple[np.ndarray, dict[float, np.ndarray]]:
        """Probabilistic hook. Only required when ``supports_quantiles=True``."""
        raise NotImplementedError

    # --- helpers ----------------------------------------------------------

    def _require_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError(f"{self.name}: call fit() before predict()")


def _coerce(series: TimeSeries | np.ndarray | list[float] | pd.Series) -> TimeSeries:
    """Best-effort coercion used by ``BaseForecaster.fit``."""
    if isinstance(series, TimeSeries):
        return series
    if isinstance(series, pd.Series):
        return TimeSeries.from_series(series)
    return TimeSeries.from_array(np.asarray(series, dtype=np.float64))


@runtime_checkable
class Forecaster(Protocol):
    """Structural-typing twin of :class:`BaseForecaster`.

    Useful as a type hint when you want to accept anything that behaves like
    a forecaster without caring about the concrete class.
    """

    name: str
    supports_quantiles: bool

    def fit(self, series: TimeSeries | np.ndarray | list[float]) -> Forecaster: ...
    def predict(self, horizon: int) -> ForecastOutput: ...
    def predict_quantiles(
        self, horizon: int, quantiles: tuple[float, ...] = ...
    ) -> ForecastOutput: ...
