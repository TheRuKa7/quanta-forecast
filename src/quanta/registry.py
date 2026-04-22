"""Name → forecaster class dispatch.

Kept intentionally small: a string-keyed dict of factories. The CLI and the
HTTP API go through this registry so adding a new backend is a single line.

Lazy registration matters: the naive/smoothing/arima/lightgbm backends
import cleanly at package load, but the deep and foundation backends pull
torch and must be imported lazily. We register a *factory* rather than a
class reference, and the factory does the import — so a simple
``registry.create("tft")`` call is what triggers the heavy import, not
``import quanta``.
"""
from __future__ import annotations

from collections.abc import Callable

from quanta.base import BaseForecaster

__all__ = ["create", "is_available", "list_backends", "register"]


#: Factory = zero-arg callable that returns a fresh BaseForecaster instance.
Factory = Callable[..., BaseForecaster]

_REGISTRY: dict[str, Factory] = {}


def register(name: str, factory: Factory) -> None:
    """Register ``factory`` under ``name``. Overwrites on duplicate."""
    if not name:
        raise ValueError("name must be non-empty")
    _REGISTRY[name] = factory


def list_backends() -> list[str]:
    """Return registered names in alphabetical order."""
    return sorted(_REGISTRY)


def is_available(name: str) -> bool:
    """Whether a factory is registered for ``name``.

    Availability here means "registered"; heavy backends are registered
    eagerly but their imports happen on first ``create`` call — so a
    ``True`` here doesn't guarantee the optional extra is installed.
    """
    return name in _REGISTRY


def create(name: str, **kwargs) -> BaseForecaster:
    """Instantiate the backend registered under ``name``.

    ``**kwargs`` are forwarded to the factory; that's how the CLI passes
    things like ``season=12`` or ``order=(2,1,2)``.
    """
    if name not in _REGISTRY:
        raise KeyError(
            f"unknown backend {name!r}; available: {list_backends()}"
        )
    return _REGISTRY[name](**kwargs)


# --- eager registrations (always available) ------------------------------

def _register_eager() -> None:
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

    register("naive", NaiveForecaster)
    register("seasonal_naive", SeasonalNaiveForecaster)
    register("mean", MeanForecaster)
    register("drift", DriftForecaster)
    register("ses", SimpleExpSmoothingForecaster)
    register("holt", HoltForecaster)
    register("holt_winters", HoltWintersForecaster)


# --- lazy registrations (heavy deps) -------------------------------------

def _register_lazy() -> None:
    """Register factories whose class imports pull in heavy deps.

    The factory does the import lazily: the ``quanta`` top-level import
    never touches statsmodels / lightgbm / torch. Users get a clear
    ``ImportError`` on first ``create`` call if the extra isn't installed.
    """

    def _make_arima(**kwargs) -> BaseForecaster:
        from quanta.models.arima import ARIMAForecaster

        return ARIMAForecaster(**kwargs)

    def _make_lightgbm(**kwargs) -> BaseForecaster:
        from quanta.models.ml import LightGBMForecaster

        return LightGBMForecaster(**kwargs)

    def _make_lightgbm_direct(**kwargs) -> BaseForecaster:
        from quanta.models.ml import DirectLightGBMForecaster

        return DirectLightGBMForecaster(**kwargs)

    def _make_tft(**kwargs) -> BaseForecaster:
        from quanta.models.deep import TFTForecaster

        return TFTForecaster(**kwargs)

    def _make_nbeats(**kwargs) -> BaseForecaster:
        from quanta.models.deep import NBEATSForecaster

        return NBEATSForecaster(**kwargs)

    def _make_nhits(**kwargs) -> BaseForecaster:
        from quanta.models.deep import NHiTSForecaster

        return NHiTSForecaster(**kwargs)

    def _make_chronos(**kwargs) -> BaseForecaster:
        from quanta.models.foundation import ChronosForecaster

        return ChronosForecaster(**kwargs)

    def _make_timesfm(**kwargs) -> BaseForecaster:
        from quanta.models.foundation import TimesFMForecaster

        return TimesFMForecaster(**kwargs)

    register("arima", _make_arima)
    register("lightgbm", _make_lightgbm)
    register("lightgbm_direct", _make_lightgbm_direct)
    register("tft", _make_tft)
    register("nbeats", _make_nbeats)
    register("nhits", _make_nhits)
    register("chronos", _make_chronos)
    register("timesfm", _make_timesfm)


_register_eager()
_register_lazy()
