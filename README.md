# quanta-forecast

> **Multi-model time-series forecasting toolkit.** Classical, ML, deep, and foundation-model forecasters behind one unified API, with backtesting, probabilistic forecasts, and a Streamlit demo.

[![CI](https://github.com/TheRuKa7/quanta-forecast/actions/workflows/ci.yml/badge.svg)](https://github.com/TheRuKa7/quanta-forecast/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Built by [Rushil Kaul](https://github.com/TheRuKa7) — a portfolio rebuild of a 2022 stock-prediction experiment, re-architected as a general forecasting library with foundation-model support (Chronos, TimesFM).

> **⚠️ Not financial advice.** The stock-price demo exists for methodology illustration only. Markets are near-efficient; this repo makes no claim of trading alpha.

## Backends

| Tier | Models | Via |
|------|--------|-----|
| Classical | ARIMA, SARIMA, Prophet, ETS | `statsforecast` |
| ML | LightGBM + lag features | `mlforecast` |
| Deep | TFT, N-BEATS, N-HiTS, PatchTST | `darts` |
| **Foundation (zero-shot)** | **Chronos**, TimesFM, Moirai | `transformers` |

## Highlights

- **Unified `Forecaster` interface** — fit/predict/backtest across every tier
- **Backtesting framework** — expanding window, rolling CV, per-horizon metrics (sMAPE, MASE, CRPS, pinball)
- **Streamlit demo** — upload CSV, pick models, get probabilistic forecasts with plotly charts
- **Three demo datasets** — stock OHLC (yfinance), M4 daily subset, energy load
- **Foundation-model hero demo** — zero-shot forecast with Chronos; no training required
- **FastAPI service** — `POST /forecast` for production use

## Docs

- [docs/RESEARCH.md](./docs/RESEARCH.md) — SOTA forecasting landscape 2026
- [docs/PLAN.md](./docs/PLAN.md) — phased implementation plan
- [docs/THINK.md](./docs/THINK.md) — design rationale, stock-prediction skepticism, PM framing
- [docs/BENCHMARKS.md](./docs/BENCHMARKS.md) — M4/ETT results across backends

## Quickstart

```bash
uv sync
uv run python -m quanta.data.loaders --dataset m4_daily
uv run python -m quanta --model chronos --horizon 30 --dataset aapl
uv run streamlit run streamlit_app/app.py
```

## License

MIT.
