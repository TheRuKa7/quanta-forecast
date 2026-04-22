"""Backend registry + lazy-import behaviour."""
from __future__ import annotations

import pytest

from quanta.registry import create, is_available, list_backends, register
from quanta.models.naive import NaiveForecaster


def test_core_backends_always_available() -> None:
    names = list_backends()
    for name in (
        "naive",
        "seasonal_naive",
        "mean",
        "drift",
        "ses",
        "holt",
        "holt_winters",
    ):
        assert name in names


def test_optional_backends_registered_eagerly() -> None:
    """Registration is lazy at the *class import* level, not at *registration*
    — the factories are registered even when the heavy deps aren't installed."""
    names = list_backends()
    for name in ("arima", "lightgbm", "tft", "chronos", "timesfm"):
        assert name in names
        assert is_available(name)


def test_create_returns_expected_type() -> None:
    m = create("naive")
    assert isinstance(m, NaiveForecaster)


def test_create_passes_kwargs() -> None:
    from quanta.models.naive import SeasonalNaiveForecaster

    m = create("seasonal_naive", season=30)
    assert isinstance(m, SeasonalNaiveForecaster)
    assert m.season == 30


def test_create_unknown_backend_raises() -> None:
    with pytest.raises(KeyError, match="unknown backend"):
        create("does_not_exist")


def test_register_empty_name_rejected() -> None:
    with pytest.raises(ValueError):
        register("", lambda: NaiveForecaster())


def test_register_overwrite_is_allowed() -> None:
    """Re-registering replaces — useful for users swapping in mocks."""
    original = create("naive")
    assert type(original) is NaiveForecaster

    class _Fake(NaiveForecaster):
        name = "naive"

    register("naive", _Fake)
    try:
        replaced = create("naive")
        assert isinstance(replaced, _Fake)
    finally:
        # Restore so subsequent tests see the real class.
        register("naive", NaiveForecaster)
