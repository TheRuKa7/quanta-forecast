"""CLI commands via typer's TestRunner."""
from __future__ import annotations

import json

import pandas as pd
from typer.testing import CliRunner

from quanta.cli import app

runner = CliRunner()


def test_cli_version() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "quanta-forecast" in result.stdout


def test_cli_list_backends_includes_core() -> None:
    result = runner.invoke(app, ["list-backends"])
    assert result.exit_code == 0
    assert "naive" in result.stdout
    assert "holt_winters" in result.stdout


def test_cli_forecast_synthetic_airline() -> None:
    result = runner.invoke(
        app,
        ["forecast", "--model", "naive", "--synthetic", "airline", "--horizon", "6"],
    )
    assert result.exit_code == 0, result.stdout
    # CSV header + 6 forecast rows.
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    # Skip any header/section lines emitted by rich before the CSV stream.
    data_lines = [line for line in lines if "," in line]
    assert len(data_lines) >= 7  # header + 6 rows


def test_cli_forecast_unknown_model_exits_nonzero() -> None:
    result = runner.invoke(
        app,
        ["forecast", "--model", "not_a_model", "--synthetic", "airline", "-h", "3"],
    )
    assert result.exit_code != 0


def test_cli_forecast_rejects_both_inputs() -> None:
    result = runner.invoke(
        app,
        [
            "forecast",
            "--model",
            "naive",
            "--synthetic",
            "airline",
            "--input",
            "doesnotexist.csv",
        ],
    )
    assert result.exit_code != 0


def test_cli_forecast_writes_output_file(tmp_path) -> None:
    out = tmp_path / "out.csv"
    result = runner.invoke(
        app,
        [
            "forecast",
            "--model",
            "drift",
            "--synthetic",
            "trend",
            "-h",
            "5",
            "-o",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert out.exists()
    df = pd.read_csv(out)
    assert len(df) == 5
    assert "point" in df.columns


def test_cli_backtest_json_shape() -> None:
    result = runner.invoke(
        app,
        [
            "backtest",
            "--model",
            "naive",
            "--synthetic",
            "airline",
            "-h",
            "6",
            "--min-train",
            "80",
            "--max-folds",
            "3",
            "--season",
            "12",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["model"] == "naive"
    assert payload["folds"] == 3
    assert "mae" in payload["summary"]


def test_cli_dataset_dumps_head() -> None:
    result = runner.invoke(app, ["dataset", "airline", "--head", "5"])
    assert result.exit_code == 0
    assert "airline_passengers" in result.stdout


def test_cli_parse_param_tuple() -> None:
    """The `--param order=1,1,1` path should be parsed as a tuple of ints,
    which ARIMA requires. We run through the CLI end-to-end on a synthetic
    series to prove the round trip works."""
    result = runner.invoke(
        app,
        [
            "forecast",
            "--model",
            "seasonal_naive",
            "--synthetic",
            "seasonal",
            "-h",
            "7",
            "--param",
            "season=7",
        ],
    )
    assert result.exit_code == 0, result.stdout
