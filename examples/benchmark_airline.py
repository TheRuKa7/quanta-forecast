"""Benchmark every non-foundation backend on the airline passengers series.

Run:

    python examples/benchmark_airline.py

Produces a single table comparing per-fold MAE/RMSE/sMAPE/MASE across
seasonal_naive, drift, holt_winters, arima, and lightgbm. Foundation
backends (chronos, timesfm) and deep backends (tft, nbeats, nhits) are
skipped because their extras require a multi-GB install — add them by
extending ``BACKENDS`` below.
"""
from __future__ import annotations

import pandas as pd

from quanta.data.loaders import load_airline_passengers
from quanta.eval.backtest import rolling_origin_backtest
from quanta.registry import create


#: (display_name, factory). Each factory must return a fresh model.
BACKENDS: list[tuple[str, callable]] = [
    ("seasonal_naive(12)", lambda: create("seasonal_naive", season=12)),
    ("drift", lambda: create("drift")),
    ("holt_winters(12)", lambda: create("holt_winters", season=12)),
    ("arima(1,1,1)(1,1,1,12)", lambda: create(
        "arima", order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)
    )),
    ("lightgbm", lambda: create("lightgbm", num_boost_round=200)),
]


def main() -> None:
    ts = load_airline_passengers()
    rows: list[dict[str, float | str]] = []
    for name, factory in BACKENDS:
        result = rolling_origin_backtest(
            factory,
            ts,
            horizon=12,
            min_train=72,  # 6 years of data
            step=6,
            season=12,
        )
        summary = result.summary()
        rows.append(
            {
                "model": name,
                "folds": result.folds,
                "mae": summary["mae"],
                "rmse": summary["rmse"],
                "smape": summary["smape"],
                "mase": summary.get("mase", float("nan")),
            }
        )

    df = pd.DataFrame(rows).set_index("model").round(3)
    # Sort by MASE (best benchmark-relative metric).
    df = df.sort_values("mase")
    print("\nAirline passengers — rolling-origin backtest, h=12")
    print(df.to_string())


if __name__ == "__main__":
    main()
