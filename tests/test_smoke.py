"""Smoke tests for the scaffold."""
from __future__ import annotations

from fastapi.testclient import TestClient

from quanta import __version__
from quanta.api.main import app

client = TestClient(app)


def test_healthz() -> None:
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json()["version"] == __version__


def test_import_base() -> None:
    from quanta.base import Forecaster, ForecastOutput
    assert Forecaster is not None
    assert ForecastOutput is not None
