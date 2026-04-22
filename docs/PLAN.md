# /ultraplan — quanta-forecast

## Goal
Ship a unified forecasting toolkit covering 4 model tiers (classical, ML, deep, foundation) with backtesting, FastAPI serving, and a Streamlit demo. ~10 working days to v1.0.

## Stack
- Python 3.13 + uv
- `statsforecast`, `mlforecast`, `neuralforecast` (Nixtla stack)
- `darts` (Unit8) for additional DL models
- `transformers` + `chronos-forecasting` for foundation models
- FastAPI for HTTP serving
- Streamlit + Plotly for demo UI
- Polars for data manipulation

## Phases

### P0 — Scaffold (Day 1)
- [x] uv project, tests, ruff, mypy, CI
- [x] Docs: RESEARCH, PLAN, THINK
- [x] FastAPI stub + Streamlit stub

### P1 — Data & classical (Days 2–3)
- [ ] `data/loaders.py` — M4 daily subset, yfinance OHLC, Monash electricity
- [ ] `models/classical.py` — ARIMA, Prophet, ETS via statsforecast
- [ ] `eval/metrics.py` — sMAPE, MASE, RMSE, CRPS, pinball, coverage
- [ ] `eval/backtest.py` — expanding window + rolling CV
- [ ] Unit tests + first benchmark table

### P2 — ML layer (Day 4)
- [ ] `models/ml.py` — LightGBM wrapper with lag feature generation
- [ ] Calendar/holiday features (workalendar)
- [ ] Add to benchmark table

### P3 — Deep models (Days 5–6)
- [ ] `models/deep.py` — darts wrappers for TFT, N-BEATS, N-HiTS, PatchTST
- [ ] Checkpoint caching (HF Hub or local)
- [ ] Small-scale training config for M4 daily
- [ ] Add to benchmark table

### P4 — Foundation models (Day 7)
- [ ] `models/foundation.py` — Chronos-tiny + TimesFM
- [ ] Zero-shot eval on M4 daily, electricity
- [ ] Document: this is the killer demo

### P5 — API + UI (Days 8–9)
- [ ] FastAPI `POST /forecast` — body: series + horizon + model choice
- [ ] Streamlit app: CSV upload → model picker → plot + metrics + download
- [ ] Demo GIF

### P6 — Benchmarks + release (Day 10)
- [ ] `scripts/benchmark.py` — one-shot run of all models on all datasets
- [ ] `docs/BENCHMARKS.md` — honest tables (including where foundation models lose)
- [ ] Release v1.0.0
- [ ] Link from `idas-scene-ai` "sibling repos" section

## Acceptance criteria
- ✅ Unified `Forecaster.fit(series).predict(h=14)` interface across all tiers
- ✅ Streamlit demo runs on CPU with no GPU / no API keys required
- ✅ Benchmark table published with honest losses as well as wins
- ✅ Foundation-model zero-shot demo works in < 10s on CPU
