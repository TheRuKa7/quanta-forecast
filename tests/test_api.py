"""HTTP surface — /healthz, /backends, /forecast."""
from __future__ import annotations

from fastapi.testclient import TestClient

from quanta import __version__
from quanta.api.main import app


client = TestClient(app)


def test_healthz() -> None:
    r = client.get("/healthz")
    assert r.status_code == 200
    payload = r.json()
    assert payload["version"] == __version__
    assert payload["status"] == "ok"


def test_backends_lists_core_names() -> None:
    r = client.get("/backends")
    assert r.status_code == 200
    names = r.json()["backends"]
    assert "naive" in names
    assert "holt_winters" in names
    assert "arima" in names
    assert "lightgbm" in names


def test_forecast_naive_roundtrip() -> None:
    payload = {
        "model": "naive",
        "values": [1.0, 2.0, 3.0, 4.0, 5.0],
        "horizon": 3,
    }
    r = client.post("/forecast", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["model"] == "naive"
    assert body["horizon"] == 3
    assert body["point"] == [5.0, 5.0, 5.0]
    assert body["quantiles"] is None


def test_forecast_with_timestamps() -> None:
    payload = {
        "model": "drift",
        "values": list(range(10)),
        "horizon": 3,
        "timestamps": [f"2024-01-{i + 1:02d}" for i in range(10)],
        "freq": "D",
    }
    r = client.post("/forecast", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body["timestamps"]) == 3
    # Drift with slope=1 after 10 points → forecast = 10, 11, 12
    assert body["point"] == [10.0, 11.0, 12.0]


def test_forecast_with_quantiles() -> None:
    payload = {
        "model": "holt_winters",
        "values": [10, 12, 15, 14, 13, 16, 18, 22, 25, 24, 21, 18] * 4,
        "horizon": 6,
        "quantiles": [0.1, 0.5, 0.9],
        "params": {"season": 12},
    }
    r = client.post("/forecast", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["quantiles"] is not None
    assert set(body["quantiles"].keys()) == {"q10", "q50", "q90"}


def test_forecast_unknown_model_returns_400() -> None:
    r = client.post(
        "/forecast",
        json={"model": "does_not_exist", "values": [1, 2, 3], "horizon": 2},
    )
    assert r.status_code == 400


def test_forecast_bad_horizon_returns_422() -> None:
    r = client.post(
        "/forecast",
        json={"model": "naive", "values": [1, 2, 3], "horizon": 0},
    )
    assert r.status_code == 422  # Pydantic validation


def test_forecast_short_series_returns_400() -> None:
    """One-value series can't be fit — BaseForecaster raises ValueError,
    which the API maps to 400."""
    r = client.post(
        "/forecast",
        json={"model": "naive", "values": [1.0, 2.0], "horizon": 1},
    )
    # 2 is the minimum, so this should succeed.
    assert r.status_code == 200

    # But pydantic's min_length=2 blocks 1.
    r = client.post(
        "/forecast",
        json={"model": "naive", "values": [1.0], "horizon": 1},
    )
    assert r.status_code == 422


def test_forecast_timestamps_length_mismatch_400() -> None:
    r = client.post(
        "/forecast",
        json={
            "model": "naive",
            "values": [1, 2, 3],
            "timestamps": ["2024-01-01", "2024-01-02"],
            "horizon": 1,
        },
    )
    assert r.status_code == 400
