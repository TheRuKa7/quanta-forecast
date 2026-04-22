# PRD — quanta-forecast

**Owner:** Rushil Kaul · **Status:** P0 scaffold complete · **Last updated:** 2026-04-22

## 1. TL;DR

A Python library that unifies **four forecaster families** (classical,
gradient-boosted, deep-learning, foundation) under one `Forecaster` protocol,
with honest back-testing, probabilistic outputs, and a Streamlit demo.

## 2. Problem

Forecasting codebases typically lock teams into one family (pmdarima, or
darts, or sktime, or a bespoke LSTM). Switching costs are high, and honest
cross-family bake-offs are rare. Meanwhile, zero-shot **foundation forecasters**
(Chronos, TimesFM, Moirai) changed the landscape in 2024–25 and most libraries
haven't fully absorbed them.

## 3. Goals

| G | Goal | Measure |
|---|------|---------|
| G1 | One `Forecaster` protocol across all families | 100% of built-ins conform; tested by `test_conformance.py` |
| G2 | Probabilistic output by default | Every forecaster returns quantiles (P10/P50/P90 min) |
| G3 | Honest back-testing | Rolling-origin CV; leak detector; no future-data features |
| G4 | Zero-shot foundation-model path | `quanta zero-shot` works without fitting on CPU |
| G5 | Ship a useful demo | Streamlit app deployable to HF Spaces in < 5 min |

## 4. Non-goals

- Trading alpha / portfolio optimisation
- Online learning (arriving-stream forecasting, incremental fit)
- Hierarchical reconciliation (maybe v2)
- Multi-horizon auto-ensembling beyond simple averaging

## 5. Users & stakeholders

See `USECASES.md` P1–P5. The "primary buyer" is a DS lead tired of stitching
four libraries together; the "loud user" is the skeptical quant; the "evangelist"
is the PM who can put probability cones in decks.

## 6. Functional requirements

| ID | Requirement | Priority |
|----|-------------|----------|
| F1 | `Forecaster` protocol with `fit(y, exog?) / predict(horizon) -> ForecastOutput` | P0 |
| F2 | Classical family: ARIMA, ETS, Theta, Seasonal-Naive | P1 |
| F3 | ML family: LightGBM with lag/rolling features; leak-safe feature builder | P1 |
| F4 | Deep family: PatchTST, N-BEATS, TFT (via PyTorch Lightning) | P2 |
| F5 | Foundation family: Chronos, TimesFM, Moirai zero-shot | P2 |
| F6 | Benchmark CLI (`quanta bench`) with rolling-origin CV | P1 |
| F7 | Leaderboard generator (leaderboard.parquet per SKU × model) | P2 |
| F8 | Streamlit demo (upload CSV, pick model, view quantile plot) | P2 |
| F9 | Typer CLI for all of the above | P0 |
| F10 | Leakage detector (fails CI if features use future rows) | P1 |

## 7. Non-functional requirements

| Category | Requirement |
|----------|-------------|
| Correctness | Conformance test suite proves every built-in forecaster honours the protocol |
| Honesty | Stock-data loader surfaces a mandatory disclaimer |
| Performance | Classical + ML runs on CPU; deep/foundation runs on CPU with warning if >30 s |
| Reproducibility | Every `fit` call records seed, package versions, data hash |
| Packaging | `uv sync --extra classical/ml/deep/foundation`: families are opt-in |
| Licensing | MIT repo; respect upstream licenses (Chronos/Amazon, TimesFM/Google) |

## 8. Success metrics

- **Primary:** number of distinct forecaster classes registered via `@register_forecaster`
  (indicator of ecosystem extensibility).
- **Secondary:** MASE/CRPS vs reference (M4 / Monash archive subset) for each family.
- **Evangelism:** Streamlit demo deploy on HF Spaces; bookmark-friendly URL.

## 9. Milestones

| Phase | Deliverable | ETA |
|-------|-------------|-----|
| P0 | Protocol, Typer CLI skeleton, conformance test harness | shipped |
| P1 | Classical + ML families, leak detector, `bench` CLI | +2 weeks |
| P2 | Deep + foundation families, leaderboard, Streamlit demo | +5 weeks |
| P3 | HF Spaces deploy, ensembling, docs site | +7 weeks |

## 10. Dependencies

- `statsforecast` (Nixtla) for classical family
- `lightgbm`
- `pytorch-lightning`, `neuralforecast` for deep family
- `chronos-forecasting` (Amazon), `timesfm` (Google), `uni2ts` (Salesforce) for foundation
- Streamlit, plotly (charts), typer (CLI)

## 11. Risks & mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| Foundation-model sizes blow CPU budget | High | Slow demo | Offer "small" variants only in default; large behind `--gpu` flag |
| Leakage slips in via subtle features | Med | Silent rot | Strict leak detector + unit tests; rolling-origin only |
| Protocol churns as families vary | Med | Breaking changes | SemVer on the protocol; deprecation windows |
| Users believe the stock demo predicts markets | High | Rep risk | Mandatory disclaimer + sample data labelled "toy" |
| License conflicts on foundation weights | Med | Distribution | Weights lazy-downloaded; licenses surfaced in `quanta info` |

## 12. Open questions

- Ship hierarchical reconciliation (MinT) in v1 or v2? Leaning v2.
- Whether to expose an auto-ML harness (`quanta auto`) or leave that as a recipe.
- Should `ForecastOutput` include quantile **samples** as well as quantile **levels**?
  Useful for downstream Monte Carlo sims; deferred to P3.
