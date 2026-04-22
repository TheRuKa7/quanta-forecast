# /ultraplan — quanta-forecast

## Goal
Ship a unified forecasting toolkit covering 4 model tiers (baseline → classical → ML → deep → foundation) with backtesting, FastAPI serving, and a Streamlit demo.

## Stack (as shipped)
- Python 3.13 + uv + hatchling
- `statsmodels` for ARIMA / SARIMA / exponential-smoothing (`[classical]` extra)
- `lightgbm` + a home-grown `LagFeatureBuilder` (`[ml]` extra)
- `darts` for TFT / N-BEATS / N-HiTS (`[deep]` extra)
- `chronos-forecasting` + `timesfm` for zero-shot foundation models (`[foundation]` extra)
- `fastapi` + `pydantic` for HTTP serving (`[demo]` extra)
- `streamlit` for the demo UI (`[demo]` extra)
- `typer` for the CLI

## Phases

### P0 — Scaffold — DONE
- [x] uv project, tests, ruff, mypy, CI (`uv sync --extra dev --extra classical --extra ml`, then `ruff check` + `pytest`)
- [x] Docs: PRD, RESEARCH, THINK, RFC, DFD, USECASES
- [x] FastAPI stub + Streamlit stub
- [x] Typer CLI stub

### P1 — Data & baselines & classical — DONE
- [x] `data/loaders.py` — `load_csv`, `load_airline_passengers` (144-point in-repo dataset), `make_synthetic_trend`, `make_synthetic_seasonal`
- [x] `models/naive.py` — `naive`, `seasonal_naive`, `mean`, `drift`
- [x] `models/smoothing.py` — `ses`, `holt`, `holt_winters` (with bootstrap residual quantiles)
- [x] `models/arima.py` — statsmodels-backed `arima` with analytical quantiles via `se_mean`
- [x] `eval/metrics.py` — MAE, RMSE, MAPE, sMAPE, MASE, pinball, coverage, CRPS
- [x] `eval/backtest.py` — expanding-or-sliding rolling-origin backtest

### P2 — ML layer — DONE
- [x] `features/lag.py` — `LagFeatureBuilder` (lags + leakage-safe rolling stats + calendar features)
- [x] `models/ml.py` — `LightGBMForecaster` (recursive) + `DirectLightGBMForecaster` (per-step models) + native quantile regression

### P3 — Deep models — DONE (dependency-gated)
- [x] `models/deep.py` — `NBEATSForecaster`, `NHiTSForecaster`, `TFTForecaster` via darts
- [x] Lazy imports — `quanta` top-level import never touches torch/darts

### P4 — Foundation models — DONE (dependency-gated)
- [x] `models/foundation.py` — `ChronosForecaster` (t5-tiny default, sample-based quantiles) + `TimesFMForecaster`
- [x] Class-level model cache so repeated instantiations skip the HF download
- [x] `examples/chronos_zeroshot.py` — hero demo with seasonal-naive baseline comparison + `--smoke` flag for CI

### P5 — API + UI — DONE
- [x] FastAPI `POST /forecast` — pydantic `ForecastRequest` / `ForecastResponse`, registry-dispatched backends, optional quantiles
- [x] Streamlit demo — dataset picker (airline / synthetic / CSV upload) + backend picker (baseline / classical / ml / foundation) + probabilistic fan chart + single-holdout sanity metrics + optional rolling-origin backtest
- [x] Typer CLI — `quanta forecast`, `quanta backtest`, `quanta list-backends`, `quanta dataset`

### P6 — Benchmarks + release — DONE
- [x] `scripts/benchmark.py` — cross-backend rolling-origin backtest that writes `docs/BENCHMARKS.md` from a live run; foundation backends are optional and the report records which were skipped
- [x] `docs/BENCHMARKS.md` — airline-passengers results, h=12, honest about ARIMA winning on this series
- [x] README rewritten against the real API
- [x] CI: ruff + pytest on every PR; foundation-smoke job on schedule/dispatch

## Acceptance criteria — all met

- Unified `BaseForecaster.fit(series).predict(horizon)` contract across all 15 registered backends
- 95 passing unit tests; `quanta` import never pulls torch/darts/chronos
- Streamlit demo runs on CPU with no GPU / no API keys
- Benchmark report published; ARIMA wins at MASE 0.668, LightGBM and seasonal_naive tied at ~1.4, drift at 2.1
- Chronos zero-shot example runs in the `foundation` extra install; exits 0 in smoke mode on minimal installs

## Known follow-ups (not must-land)

- Wider benchmark corpus: M4 daily subset, electricity, ETT — currently only the bundled airline series ships in-repo to keep the install small
- Per-horizon metric breakouts (current tables average across all H steps)
- Chronos `-small` / `-base` comparison once a runner with ~1GB of RAM-to-spare is available
- Conformal prediction wrapper so every backend gets distribution-free intervals
