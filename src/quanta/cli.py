"""quanta CLI — forecast, backtest, and inspect from the terminal.

Commands:

* ``quanta version`` — print version
* ``quanta list-backends`` — enumerate registered backends
* ``quanta forecast`` — fit + predict, emit CSV to stdout or path
* ``quanta backtest`` — rolling-origin CV, emit per-fold + summary tables
* ``quanta dataset`` — dump one of the built-in synthetic datasets

All commands accept ``--input PATH`` for a CSV with ``ds,y`` columns, or
``--synthetic airline|trend|seasonal`` to use one of the bundled series.
That duality means the CLI is usable out of the box with zero setup.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from quanta import (
    __version__,
    create,
    list_backends,
    load_airline_passengers,
    load_csv,
    make_synthetic_seasonal,
    make_synthetic_trend,
)
from quanta.base import TimeSeries
from quanta.eval.backtest import rolling_origin_backtest

app = typer.Typer(
    name="quanta",
    help="Multi-model time-series forecasting toolkit.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()
err_console = Console(stderr=True, style="bold red")


def _load_input(
    input_path: Optional[Path], synthetic: Optional[str]
) -> TimeSeries:
    """Resolve the ``--input`` / ``--synthetic`` knob into a :class:`TimeSeries`.

    Exactly one must be provided. Synthetic names are deliberately
    restricted so typos don't silently fall back to airline passengers.
    """
    if (input_path is None) == (synthetic is None):
        raise typer.BadParameter(
            "provide exactly one of --input or --synthetic"
        )
    if input_path is not None:
        return load_csv(input_path)
    assert synthetic is not None
    table = {
        "airline": load_airline_passengers,
        "seasonal": lambda: make_synthetic_seasonal(n=365, period=7, seed=0),
        "trend": lambda: make_synthetic_trend(n=200, seed=0),
    }
    if synthetic not in table:
        raise typer.BadParameter(
            f"unknown synthetic dataset {synthetic!r}; "
            f"choose one of {sorted(table)}"
        )
    return table[synthetic]()


def _parse_kwargs(pairs: list[str]) -> dict[str, object]:
    """Parse ``--param k=v`` pairs into a dict with int/float coercion."""
    out: dict[str, object] = {}
    for pair in pairs:
        if "=" not in pair:
            raise typer.BadParameter(
                f"bad --param {pair!r}; expected k=v"
            )
        key, value = pair.split("=", 1)
        # Try int → float → str.
        try:
            out[key] = int(value)
            continue
        except ValueError:
            pass
        try:
            out[key] = float(value)
            continue
        except ValueError:
            pass
        # Tuples like "1,1,1" → (1,1,1)
        if "," in value:
            try:
                out[key] = tuple(
                    int(v) if v.strip().lstrip("-").isdigit() else float(v)
                    for v in value.split(",")
                )
                continue
            except ValueError:
                pass
        out[key] = value
    return out


@app.command()
def version() -> None:
    """Print the installed version."""
    console.print(f"quanta-forecast [bold cyan]v{__version__}[/]")


@app.command("list-backends")
def list_backends_cmd() -> None:
    """Enumerate registered forecaster backends."""
    table = Table(title="quanta backends", show_lines=False)
    table.add_column("name", style="cyan")
    table.add_column("tier", style="green")

    tiers = {
        "naive": "baseline",
        "seasonal_naive": "baseline",
        "mean": "baseline",
        "drift": "baseline",
        "ses": "classical",
        "holt": "classical",
        "holt_winters": "classical",
        "arima": "classical (extra)",
        "lightgbm": "ml (extra)",
        "lightgbm_direct": "ml (extra)",
        "tft": "deep (extra)",
        "nbeats": "deep (extra)",
        "nhits": "deep (extra)",
        "chronos": "foundation (extra)",
        "timesfm": "foundation (extra)",
    }
    for backend in list_backends():
        table.add_row(backend, tiers.get(backend, "?"))
    console.print(table)


@app.command()
def forecast(
    model: str = typer.Option(
        ..., "--model", "-m", help="backend name (see `list-backends`)"
    ),
    horizon: int = typer.Option(14, "--horizon", "-h"),
    input_path: Optional[Path] = typer.Option(
        None, "--input", "-i", help="CSV with `ds,y` columns"
    ),
    synthetic: Optional[str] = typer.Option(
        None,
        "--synthetic",
        help="use a built-in series: airline | seasonal | trend",
    ),
    quantiles: bool = typer.Option(
        False, "--quantiles/--no-quantiles", help="emit q10/q50/q90 columns"
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="write CSV here; default = stdout"
    ),
    param: list[str] = typer.Option(
        [], "--param", "-p", help="model kwarg, e.g. --param season=12"
    ),
) -> None:
    """Fit a model and forecast ``horizon`` steps ahead."""
    ts = _load_input(input_path, synthetic)
    kwargs = _parse_kwargs(param)
    m = create(model, **kwargs)
    m.fit(ts)
    if quantiles and m.supports_quantiles:
        fc = m.predict_quantiles(horizon=horizon)
    else:
        fc = m.predict(horizon=horizon)
        if quantiles:
            err_console.print(
                f"[yellow]warning[/] {model} has no quantile support; "
                "emitting point only"
            )
    df = fc.to_frame()
    df.index.name = "ds"
    if output is None:
        df.to_csv(sys.stdout)
    else:
        df.to_csv(output)
        console.print(f"[green]wrote[/] {output} ({len(df)} rows)")


@app.command()
def backtest(
    model: str = typer.Option(..., "--model", "-m"),
    horizon: int = typer.Option(14, "--horizon", "-h"),
    min_train: int = typer.Option(
        60, "--min-train", help="size of first training window"
    ),
    step: int = typer.Option(1, "--step", help="fold stride"),
    max_folds: Optional[int] = typer.Option(None, "--max-folds"),
    expanding: bool = typer.Option(
        True, "--expanding/--sliding", help="window strategy"
    ),
    season: int = typer.Option(1, "--season", help="MASE seasonal period"),
    input_path: Optional[Path] = typer.Option(None, "--input", "-i"),
    synthetic: Optional[str] = typer.Option(None, "--synthetic"),
    param: list[str] = typer.Option([], "--param", "-p"),
    json_out: bool = typer.Option(
        False, "--json", help="emit machine-readable JSON instead of a table"
    ),
) -> None:
    """Rolling-origin backtest on the selected series."""
    ts = _load_input(input_path, synthetic)
    kwargs = _parse_kwargs(param)

    def factory():
        return create(model, **kwargs)

    result = rolling_origin_backtest(
        factory,
        ts,
        horizon=horizon,
        min_train=min_train,
        step=step,
        max_folds=max_folds,
        expanding=expanding,
        season=season,
    )
    summary = result.summary().to_dict()
    if json_out:
        payload = {
            "model": result.model_name,
            "folds": result.folds,
            "horizon": result.horizon,
            "summary": summary,
            "per_fold": result.per_fold.reset_index().to_dict(orient="records"),
        }
        console.print_json(data=payload)
        return

    table = Table(
        title=f"backtest {result.model_name} "
        f"(folds={result.folds}, horizon={result.horizon})"
    )
    table.add_column("metric", style="cyan")
    table.add_column("mean across folds", justify="right")
    for metric_name, value in summary.items():
        table.add_row(metric_name, f"{value:.4f}")
    console.print(table)


@app.command()
def dataset(
    name: str = typer.Argument(
        ..., help="airline | seasonal | trend — one of the built-in series"
    ),
    head: int = typer.Option(10, "--head", help="print first N rows"),
) -> None:
    """Inspect a built-in dataset."""
    ts = _load_input(None, name)
    df = ts.to_series().rename(ts.name).to_frame()
    df.index.name = "ds"
    console.print(
        f"[bold]{ts.name}[/] — len={len(ts)}, freq={ts.freq}, "
        f"start={ts.index[0]}, end={ts.index[-1]}"
    )
    console.print(df.head(head))


if __name__ == "__main__":
    app()
