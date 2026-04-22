"""FastAPI surface for quanta-forecast.

Endpoints:

* ``GET /healthz`` — liveness + version
* ``GET /backends`` — enumerate registered backends
* ``POST /forecast`` — fit-and-forecast from a JSON payload

Design:

* **Stateless** — every request fits from scratch and returns the forecast.
  This is the right shape for a library with heterogeneous backends: the
  caller knows their history; we have no persistent store. A persistent
  ``/models/{id}`` surface can be layered on later without breaking this.
* **Synchronous** — ARIMA + LightGBM fit in milliseconds on the sizes
  users will POST. Async + a job queue would be overkill here; we can add
  it when a foundation-model POST takes too long.
* **Pydantic request models** — lax coercion (accepts list/str for the
  index) so clients written in any language don't have to match a narrow
  spec.
"""
from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from quanta import __version__
from quanta.base import TimeSeries
from quanta.registry import create, list_backends

app = FastAPI(title="quanta-forecast", version=__version__)


class ForecastRequest(BaseModel):
    """Payload for ``POST /forecast``.

    Either ``timestamps`` or ``start`` + ``freq`` must be provided so we can
    reconstruct the index. Pure-``values`` requests get a RangeIndex and a
    RangeIndex-based forecast back.
    """

    model: str = Field(..., description="backend name; see /backends")
    values: list[float] = Field(..., min_length=2)
    horizon: int = Field(..., ge=1, le=1024)
    timestamps: list[str] | None = None
    start: str | None = None
    freq: str | None = None
    quantiles: list[float] | None = None
    params: dict[str, Any] = Field(default_factory=dict)

    @field_validator("quantiles")
    @classmethod
    def _valid_quantiles(cls, v: list[float] | None) -> list[float] | None:
        if v is None:
            return None
        for q in v:
            if not (0.0 < q < 1.0):
                raise ValueError(f"quantile must be in (0,1), got {q}")
        return v


class ForecastResponse(BaseModel):
    model: str
    horizon: int
    timestamps: list[str]
    point: list[float]
    quantiles: dict[str, list[float]] | None = None


def _build_ts(req: ForecastRequest) -> TimeSeries:
    """Construct a TimeSeries from the request, picking the best available index."""
    n = len(req.values)
    if req.timestamps is not None:
        if len(req.timestamps) != n:
            raise HTTPException(
                status_code=400,
                detail=f"timestamps length {len(req.timestamps)} != values length {n}",
            )
        idx = pd.DatetimeIndex(pd.to_datetime(req.timestamps))
        freq = req.freq or pd.infer_freq(idx)
        return TimeSeries(
            values=np.asarray(req.values, dtype=np.float64),
            index=idx,
            freq=freq,
        )
    if req.start is not None and req.freq is not None:
        idx = pd.date_range(start=req.start, periods=n, freq=req.freq)
        return TimeSeries(
            values=np.asarray(req.values, dtype=np.float64),
            index=idx,
            freq=req.freq,
        )
    return TimeSeries.from_array(np.asarray(req.values, dtype=np.float64))


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {
        "status": "ok",
        "version": __version__,
        "ts": datetime.now(UTC).isoformat(),
    }


@app.get("/")
async def root() -> dict[str, str]:
    return {
        "service": "quanta-forecast",
        "version": __version__,
        "repo": "https://github.com/TheRuKa7/quanta-forecast",
    }


@app.get("/backends")
async def backends() -> dict[str, list[str]]:
    """List every registered backend name."""
    return {"backends": list_backends()}


@app.post("/forecast", response_model=ForecastResponse)
async def forecast(req: ForecastRequest) -> ForecastResponse:
    """Fit-and-predict.

    On unknown backend we return 400. On missing optional dep (e.g. torch
    for Chronos) the lazy import inside the factory raises ImportError,
    which we translate to 503 Service Unavailable — from the client's
    perspective the backend exists, it's just not deployed here.
    """
    ts = _build_ts(req)
    try:
        model = create(req.model, **req.params)
    except KeyError:
        raise HTTPException(
            status_code=400,
            detail=f"unknown model {req.model!r}; see /backends",
        )
    try:
        model.fit(ts)
    except ImportError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    want_quantiles = req.quantiles is not None and model.supports_quantiles
    if want_quantiles:
        assert req.quantiles is not None
        fc = model.predict_quantiles(
            horizon=req.horizon, quantiles=tuple(req.quantiles)
        )
        qdict = (
            {f"q{int(q * 100):02d}": arr.tolist() for q, arr in fc.quantiles.items()}
            if fc.quantiles is not None
            else None
        )
    else:
        fc = model.predict(horizon=req.horizon)
        qdict = None

    return ForecastResponse(
        model=fc.model_name,
        horizon=fc.horizon,
        timestamps=[str(t) for t in fc.index],
        point=fc.point.tolist(),
        quantiles=qdict,
    )
