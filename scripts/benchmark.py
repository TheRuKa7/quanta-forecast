"""Cross-backend benchmark that writes a markdown report into docs/.

This is the script that produces ``docs/BENCHMARKS.md`` from a live run,
so the published numbers always match what the code currently does. Run:

    uv run python scripts/benchmark.py

The script:

1. Loads the bundled airline-passengers series (144 monthly points).
2. Runs a rolling-origin backtest at horizon=12, min_train=72, step=6
   for every non-foundation backend installed.
3. Writes a markdown table + interpretation into ``docs/BENCHMARKS.md``.

Foundation backends (chronos, timesfm) are attempted but gracefully
skipped if their extra isn't installed — the report notes this so a
reader knows which tier was measured.
"""
from __future__ import annotations

import platform
import sys
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from quanta import __version__ as quanta_version
from quanta import create, load_airline_passengers
from quanta.base import TimeSeries
from quanta.eval.backtest import rolling_origin_backtest
from quanta.eval.metrics import coverage

#: (display name, factory, supports_quantiles).
BACKENDS: list[tuple[str, Callable, bool]] = [
    ("seasonal_naive(12)", lambda: create("seasonal_naive", season=12), False),
    ("drift", lambda: create("drift"), False),
    ("holt_winters(12)", lambda: create("holt_winters", season=12), True),
    (
        "arima(1,1,1)(1,1,1,12)",
        lambda: create("arima", order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)),
        True,
    ),
    ("lightgbm", lambda: create("lightgbm", num_boost_round=200), True),
]

#: Foundation backends attempted optionally — missing extra => report skip.
FOUNDATION_BACKENDS: list[tuple[str, Callable, bool]] = [
    (
        "chronos-t5-tiny (zero-shot)",
        lambda: create(
            "chronos", model_name="amazon/chronos-t5-tiny", num_samples=50
        ),
        True,
    ),
]


def _try_factory(factory: Callable) -> tuple[bool, str]:
    """Attempt the full fit+predict path to surface lazy-import failures."""
    try:
        m = factory()
        # Smoke-fit on a toy to make sure _fit's imports resolve.
        toy = load_airline_passengers()
        m.fit(toy)
        m.predict(horizon=3)
        return True, ""
    except (ImportError, ModuleNotFoundError) as e:
        return False, str(e)
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def _single_holdout_coverage(
    factory: Callable, series: TimeSeries, horizon: int
) -> float | None:
    """Coverage@80 on a single last-horizon holdout for quantile-capable models."""
    try:
        train = TimeSeries(
            values=series.values[:-horizon],
            index=series.index[:-horizon],
            freq=series.freq,
            name=series.name,
        )
        model = factory()
        model.fit(train)
        out = model.predict_quantiles(horizon=horizon, quantiles=(0.1, 0.9))
        return float(
            coverage(series.values[-horizon:], out.quantiles[0.1], out.quantiles[0.9])
        )
    except Exception:
        return None


def _benchmark_row(name: str, factory: Callable, series: TimeSeries, supports_q: bool):
    t0 = time.perf_counter()
    result = rolling_origin_backtest(
        factory,
        series,
        horizon=12,
        min_train=72,
        step=6,
        season=12,
    )
    elapsed = time.perf_counter() - t0
    summary = result.summary()

    cov80 = None
    if supports_q:
        cov80 = _single_holdout_coverage(factory, series, horizon=12)

    return {
        "model": name,
        "folds": result.folds,
        "mae": float(summary["mae"]),
        "rmse": float(summary["rmse"]),
        "smape": float(summary["smape"]),
        "mase": float(summary.get("mase", float("nan"))),
        "coverage_80": cov80 if cov80 is not None else float("nan"),
        "seconds": elapsed,
    }


def _format_table(df: pd.DataFrame) -> str:
    """Render a markdown table. Pandas' to_markdown needs tabulate; keep it
    vanilla so the script has no extra dep."""
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    rows = []
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, float):
                cells.append("nan" if not np.isfinite(v) else f"{v:.3f}")
            else:
                cells.append(str(v))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, sep, *rows])


def main(out_path: Path | None = None) -> int:
    series = load_airline_passengers()
    print(f"airline-passengers: {len(series)} obs @ freq={series.freq}")

    rows = []
    skipped: list[tuple[str, str]] = []

    all_backends = BACKENDS + FOUNDATION_BACKENDS
    for name, factory, supports_q in all_backends:
        ok, reason = _try_factory(factory)
        if not ok:
            print(f"  skip {name}: {reason}")
            skipped.append((name, reason))
            continue
        print(f"  run  {name} ...", end="", flush=True)
        row = _benchmark_row(name, factory, series, supports_q)
        print(f" {row['seconds']:.2f}s")
        rows.append(row)

    df = pd.DataFrame(rows).set_index("model")
    df = df.sort_values("mase")

    # --- Render markdown report ----------------------------------------
    out_path = out_path or Path(__file__).resolve().parent.parent / "docs" / "BENCHMARKS.md"
    ts = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")

    table_df = df[["folds", "mae", "rmse", "smape", "mase", "coverage_80", "seconds"]].copy()
    table_df.index.name = "model"
    table_df = table_df.reset_index()

    md_parts: list[str] = []
    md_parts.append("# Benchmarks")
    md_parts.append("")
    md_parts.append(
        "> Generated by `scripts/benchmark.py` — the numbers below come from a live run, "
        "so they always match the current code. Re-run the script any time the implementation "
        "changes."
    )
    md_parts.append("")
    md_parts.append("## Setup")
    md_parts.append("")
    md_parts.append(f"- **Dataset:** Box-Jenkins airline passengers, {len(series)} monthly obs (1949-1960)")
    md_parts.append("- **Protocol:** rolling-origin backtest, horizon=12, min_train=72, step=6, season=12")
    md_parts.append("- **Metrics:** MAE / RMSE / sMAPE (%) / MASE (seasonal naive = 1) / Coverage@80 (single-holdout)")
    md_parts.append(f"- **quanta-forecast:** v{quanta_version}")
    md_parts.append(f"- **Python:** {platform.python_version()} on {platform.system()} {platform.machine()}")
    md_parts.append(f"- **Generated:** {ts}")
    md_parts.append("")
    md_parts.append("## Results — airline passengers, h=12")
    md_parts.append("")
    md_parts.append(_format_table(table_df))
    md_parts.append("")

    md_parts.append("## How to read this")
    md_parts.append("")
    md_parts.append("- **MASE < 1** means the model beats a seasonal-naive baseline on average absolute error.")
    md_parts.append("- **sMAPE** is in percent, range 0-200; lower is better.")
    md_parts.append("- **Coverage@80** is a single-holdout sanity check — well-calibrated intervals land near 0.80. Models without native quantile output show `nan`.")
    md_parts.append("- **seconds** is the total wall-clock time for all folds, including fit + predict.")
    md_parts.append("")

    if skipped:
        md_parts.append("## Skipped backends")
        md_parts.append("")
        md_parts.append("These backends weren't measured because their optional extra wasn't installed on the machine that generated this report:")
        md_parts.append("")
        for name, reason in skipped:
            md_parts.append(f"- `{name}` — {reason}")
        md_parts.append("")
        md_parts.append("Install the matching extra (`[deep]`, `[foundation]`) and re-run to include them.")
        md_parts.append("")

    md_parts.append("## Notes")
    md_parts.append("")
    md_parts.append("- The airline series is short (144 obs) and strongly seasonal — a setting where classical methods (ARIMA, Holt-Winters) have a structural advantage over ML/foundation models that rely on larger context windows. Don't read these numbers as a global ranking; they're a sanity check that the unified API works end-to-end.")
    md_parts.append("- Running this benchmark on your own data is the right thing to do. The repo ships `scripts/benchmark.py` as a template — point it at `load_csv(...)` with your series and the protocol above will apply verbatim.")
    md_parts.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(md_parts), encoding="utf-8")
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
