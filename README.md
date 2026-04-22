# quanta-forecast

> **Multi-model time-series forecasting toolkit.** Classical, ML, deep, and foundation-model forecasters behind one unified API, with backtesting, probabilistic forecasts, and a Streamlit demo.

[![CI](https://github.com/TheRuKa7/quanta-forecast/actions/workflows/ci.yml/badge.svg)](https://github.com/TheRuKa7/quanta-forecast/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Built by [Rushil Kaul](https://github.com/TheRuKa7) — a portfolio rebuild of a 2022 stock-prediction experiment, re-architected as a general forecasting library with foundation-model support (Chronos, TimesFM).

> **Not financial advice.** The stock-price demo exists for methodology illustration only. Markets are near-efficient; this repo makes no claim of trading alpha.

## Backends

Every backend implements the same `BaseForecaster` contract: `fit(series) -> self`,
`predict(horizon) -> ForecastOutput`, optional `predict_quantiles(...)`. Quantile
support is advertised via `supports_quantiles`.

| Tier | Registry keys | Package (extra) |
|------|---------------|-----------------|
| **Baseline** | `naive`, `seasonal_naive`, `mean`, `drift` | core (no extra) |
| **Classical** | `arima`, `ses`, `holt`, `holt_winters` | `statsmodels` (`[classical]`) |
| **ML** | `lightgbm`, `lightgbm_direct` | `lightgbm` (`[ml]`) |
| **Deep** | `nbeats`, `tft` | `darts`, `torch` (`[deep]`) |
| **Foundation (zero-shot)** | `chronos`, `timesfm` | `chronos-forecasting`, `timesfm` (`[foundation]`) |

Install a tier by picking its extra:

```bash
pip install "quanta-forecast[classical,ml]"      # everyday stack
pip install "quanta-forecast[foundation]"        # Chronos / TimesFM
pip install "quanta-forecast[deep]"              # TFT / N-BEATS via darts
pip install "quanta-forecast[demo]"              # adds Streamlit + FastAPI
```

## Highlights

- **Unified `Forecaster` interface** — fit/predict/backtest across every tier
- **Rolling-origin backtest** — expanding or sliding window, MAE / RMSE / sMAPE / MASE per fold
- **Probabilistic forecasts** — pinball loss, CRPS, coverage, plus native quantile heads in LightGBM and Chronos
- **Lazy imports** — torch, darts, chronos never load unless you actually instantiate a model that needs them
- **Streamlit demo** — pick a dataset, pick a model, see a probabilistic fan chart
- **FastAPI service** — `POST /forecast` for production use
- **Typer CLI** — `quanta forecast`, `quanta backtest`, `quanta list-backends`, `quanta dataset`

## Quickstart

```bash
# Install (dev + classical + ml)
uv sync --extra dev --extra classical --extra ml

# Forecast the bundled airline-passengers series with SARIMA
uv run quanta forecast \
  --dataset airline \
  --backend arima \
  --horizon 12 \
  --param order="(1,1,1)" \
  --param seasonal_order="(1,1,1,12)"

# Backtest LightGBM with a 12-step rolling horizon
uv run quanta backtest \
  --dataset airline \
  --backend lightgbm \
  --horizon 12 \
  --min-train 60 \
  --step 12 \
  --season 12

# Zero-shot forecast with Chronos (foundation extra required)
uv run python examples/chronos_zeroshot.py

# Streamlit demo
uv run streamlit run streamlit_app/app.py
```

## Python API

```python
from quanta import create, load_airline_passengers, rolling_origin_backtest

series = load_airline_passengers()

# Point forecast
model = create("arima", order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
model.fit(series)
out = model.predict(horizon=12)
print(out.point)          # np.ndarray of length 12
print(out.index)          # projected DatetimeIndex

# Probabilistic forecast
model_lgb = create("lightgbm")
model_lgb.fit(series)
out = model_lgb.predict_quantiles(horizon=12, quantiles=(0.1, 0.5, 0.9))
print(out.quantiles[0.1]) # lower band

# Rolling-origin backtest
result = rolling_origin_backtest(
    factory=lambda: create("arima", order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)),
    series=series,
    horizon=12,
    min_train=60,
    step=12,
    season=12,
)
print(result.per_fold)    # MAE / RMSE / sMAPE / MASE per fold
print(result.summary())   # mean across folds
```

## FastAPI service

```bash
uv run uvicorn quanta.api.main:app --reload
```

```bash
curl -X POST http://localhost:8000/forecast \
  -H 'Content-Type: application/json' \
  -d '{
    "backend": "seasonal_naive",
    "params": {"season": 12},
    "horizon": 12,
    "values": [112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118],
    "freq": "MS"
  }'
```

## Docs

- [docs/PLAN.md](./docs/PLAN.md) — phased implementation plan + current status
- [docs/PRD.md](./docs/PRD.md) — product requirements
- [docs/RESEARCH.md](./docs/RESEARCH.md) — SOTA forecasting landscape 2026
- [docs/THINK.md](./docs/THINK.md) — design rationale, stock-prediction skepticism, PM framing
- [docs/BENCHMARKS.md](./docs/BENCHMARKS.md) — cross-backend results on the airline dataset
- [docs/RFC.md](./docs/RFC.md) / [docs/DFD.md](./docs/DFD.md) — architecture notes
- [docs/USECASES.md](./docs/USECASES.md) — targeted scenarios

## Development

```bash
uv sync --extra dev --extra classical --extra ml
uv run ruff check src tests
uv run pytest -q --no-cov

# Re-run the benchmark to refresh docs/BENCHMARKS.md
uv run python scripts/benchmark.py
```

## License

MIT.
