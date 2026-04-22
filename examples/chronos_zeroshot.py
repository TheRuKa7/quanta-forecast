"""Zero-shot forecasting with Amazon Chronos.

Chronos is a T5-family foundation model for time series: no training
required, just feed it a context window and ask for a forecast. This
example:

1. Loads the bundled airline-passengers series (144 monthly points).
2. Holds out the last 12 months.
3. Asks ``amazon/chronos-t5-tiny`` (~8M params, CPU-friendly) for a
   zero-shot 12-month forecast + 80 percent prediction interval.
4. Compares the Chronos forecast against a seasonal-naive baseline using
   MAE / RMSE / sMAPE / MASE / coverage@80.

Run:

    # Standard run (downloads the Chronos weights on first call):
    uv run python examples/chronos_zeroshot.py

    # CI smoke test — skips the forward pass when the foundation extra
    # is not installed, so the script exits 0 on a minimal install:
    uv run python examples/chronos_zeroshot.py --smoke

Install:

    pip install 'quanta-forecast[foundation]'

The first run downloads ~30 MB of weights into ``~/.cache/huggingface``.
Subsequent runs are offline-capable.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from quanta import TimeSeries, create, load_airline_passengers
from quanta.eval.metrics import coverage, mae, mase, rmse, smape


def _fmt(x: float) -> str:
    return f"{x:>8.3f}" if np.isfinite(x) else "     nan"


def main(smoke: bool = False) -> int:
    series = load_airline_passengers()
    horizon = 12

    # Hold out the last year.
    train_values = series.values[:-horizon]
    train_index = series.index[:-horizon]
    train = TimeSeries(
        values=train_values, index=train_index, freq=series.freq, name=series.name
    )
    actual = series.values[-horizon:]

    print(f"Dataset: airline passengers | train={len(train)} | horizon={horizon}")
    print("-" * 60)

    # --- Seasonal-naive baseline (always available, no deps) ------------
    naive = create("seasonal_naive", season=12)
    naive.fit(train)
    naive_out = naive.predict(horizon=horizon)
    naive_row = {
        "MAE": mae(actual, naive_out.point),
        "RMSE": rmse(actual, naive_out.point),
        "sMAPE": smape(actual, naive_out.point),
        "MASE": mase(actual, naive_out.point, train_values, season=12),
    }

    # --- Chronos zero-shot ----------------------------------------------
    try:
        chronos = create(
            "chronos",
            model_name="amazon/chronos-t5-tiny",
            device="cpu",
            num_samples=50,
        )
        print("Fitting Chronos (lazy-loading weights on first call)...")
        chronos.fit(train)
        print("Predicting (point + 80% interval)...")
        ch_out = chronos.predict_quantiles(horizon=horizon, quantiles=(0.1, 0.5, 0.9))
    except (ImportError, ModuleNotFoundError) as e:
        print(f"chronos unavailable — foundation extra not installed: {e}")
        if smoke:
            print("smoke-test mode: printing naive baseline only and exiting 0")
            print()
            header = f"{'model':<22} {'MAE':>8} {'RMSE':>8} {'sMAPE':>8} {'MASE':>8}"
            print(header)
            print("-" * len(header))
            _print_row("seasonal_naive(12)", naive_row, coverage_val=None)
            return 0
        return 1

    ch_row = {
        "MAE": mae(actual, ch_out.point),
        "RMSE": rmse(actual, ch_out.point),
        "sMAPE": smape(actual, ch_out.point),
        "MASE": mase(actual, ch_out.point, train_values, season=12),
    }
    cov80 = coverage(actual, ch_out.quantiles[0.1], ch_out.quantiles[0.9])

    # --- Report ---------------------------------------------------------
    print()
    header = f"{'model':<22} {'MAE':>8} {'RMSE':>8} {'sMAPE':>8} {'MASE':>8} {'Cov@80':>8}"
    print(header)
    print("-" * len(header))
    _print_row("seasonal_naive(12)", naive_row, coverage_val=None)
    _print_row("chronos-t5-tiny", ch_row, coverage_val=cov80)

    print()
    print("Interpretation:")
    print("  - MASE < 1 means the model beats a seasonal-naive baseline.")
    print("  - Coverage close to 0.80 means the interval is well-calibrated.")
    print("  - This is zero-shot: Chronos was NEVER trained on this series.")
    return 0


def _print_row(name: str, row: dict, coverage_val: float | None) -> None:
    cov_s = _fmt(coverage_val) if coverage_val is not None else "       -"
    print(
        f"{name:<22} {_fmt(row['MAE'])} {_fmt(row['RMSE'])} "
        f"{_fmt(row['sMAPE'])} {_fmt(row['MASE'])} {cov_s}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="If foundation extra not installed, exit 0 instead of 1.",
    )
    args = parser.parse_args()
    sys.exit(main(smoke=args.smoke))
