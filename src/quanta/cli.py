"""quanta CLI — forecast from the terminal."""
from __future__ import annotations

import typer
from rich.console import Console

from quanta import __version__

app = typer.Typer(
    name="quanta",
    help="Multi-model time-series forecasting toolkit.",
    no_args_is_help=True,
)
console = Console()


@app.command()
def version() -> None:
    """Print version and exit."""
    console.print(f"quanta-forecast [bold cyan]v{__version__}[/]")


@app.command()
def forecast(
    dataset: str = typer.Option(..., "--dataset", "-d", help="m4_daily | aapl | energy"),
    model: str = typer.Option("chronos", "--model", "-m", help="classical | ml | deep | chronos"),
    horizon: int = typer.Option(14, "--horizon", "-h"),
) -> None:
    """Run a forecast. NOTE: full implementation ships in P1–P4 (see docs/PLAN.md)."""
    console.print(f"[yellow]stub[/] forecast dataset={dataset} model={model} horizon={horizon}")


if __name__ == "__main__":
    app()
